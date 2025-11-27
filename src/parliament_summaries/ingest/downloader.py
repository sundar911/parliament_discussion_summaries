from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from ..config.models import RuntimeSettings, ScraperSettings
from ..utils.io import ensure_directory
from ..utils.logging import get_logger
from .catalog import DebateRecord

logger = get_logger(__name__)


class DebateDownloader:
    """Download debate PDFs and persist metadata locally."""

    def __init__(self, runtime: RuntimeSettings, scraper: ScraperSettings) -> None:
        self.runtime = runtime
        self.scraper = scraper
        ensure_directory(Path(runtime.raw_dir))
        self._client = httpx.AsyncClient(
            headers={"User-Agent": scraper.user_agent},
            timeout=scraper.timeout_seconds,
        )

    async def resolve_pdf_url(self, record: DebateRecord) -> Optional[str]:
        """Fetch the debate page and extract the PDF link."""
        page_url = httpx.URL(self.scraper.base_url + record.url)
        response = await self._client.get(page_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        meta_pdf = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if meta_pdf and meta_pdf.get("content"):
            return self._normalise_pdf_url(meta_pdf["content"], page_url)

        for anchor in soup.select("a[href$='.pdf']"):
            href = anchor.get("href", "")
            if href:
                return self._normalise_pdf_url(href, page_url)
        return None

    def _normalise_pdf_url(self, href: str, page_url: httpx.URL) -> str:
        """Convert hrefs (relative, internal IP, or http) to public HTTPS URL."""
        parsed = urlparse(href)

        if not parsed.netloc:
            return str(page_url.join(href))

        # Handle internal IP hosts by replacing with public base.
        if parsed.hostname and parsed.hostname.startswith("10."):
            base = httpx.URL(self.scraper.base_url)
            relative = parsed.path.lstrip("/")
            if parsed.query:
                relative = f"{relative}?{parsed.query}"
            joined = base.join(relative)
            return str(joined)

        # Ensure HTTPS for http links.
        if href.startswith("http://"):
            return "https://" + href[len("http://") :]

        return href

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _download_once(self, pdf_url: str, target_path: Path) -> None:
        async with self._client.stream("GET", pdf_url) as response:
            response.raise_for_status()
            with target_path.open("wb") as fh:
                async for chunk in response.aiter_bytes():
                    fh.write(chunk)

    async def download_record(self, record: DebateRecord) -> Path:
        """Download a single debate PDF, returning resulting path."""
        pdf_url = record.pdf_url or await self.resolve_pdf_url(record)
        if not pdf_url:
            raise ValueError(f"Could not locate PDF for debate {record.url}")

        filename = (
            f"{record.date}_{record.title}"
            .replace("/", "-")
            .replace(" ", "_")
            .replace(":", "-")
        )
        target_path = Path(self.runtime.raw_dir) / f"{filename}.pdf"

        if target_path.exists():
            logger.info("Skipping download; file already exists %s", target_path)
            return target_path

        logger.info("Downloading %s -> %s", pdf_url, target_path)
        await self._download_once(pdf_url, target_path)
        await asyncio.sleep(self.scraper.request_interval_seconds)
        return target_path

    async def bulk_download(self, records: Iterable[DebateRecord]) -> list[Path]:
        results: list[Path] = []
        for record in records:
            path = await self.download_record(record)
            results.append(path)
        return results

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "DebateDownloader":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
