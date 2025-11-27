from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

import httpx
import pendulum
from bs4 import BeautifulSoup

from ..config.models import ScraperSettings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DebateRecord:
    """Metadata describing a Lok Sabha debate entry on the portal."""

    title: str
    date: str
    session: Optional[str]
    url: str
    pdf_url: Optional[str] = None


class DebateCatalog:
    """Scrape debate metadata from the eParliament portal."""

    def __init__(self, settings: ScraperSettings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"User-Agent": settings.user_agent},
            timeout=settings.timeout_seconds,
        )
        self._page_size = 20
        self._cutoff_date = None
        if settings.years_back and settings.years_back > 0:
            self._cutoff_date = pendulum.now("UTC").subtract(years=settings.years_back).date()

    async def _fetch_page(self, offset: int) -> List[DebateRecord]:
        path = f"{self.settings.handle_path}?offset={offset}"
        response = await self._client.get(path)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        rows = soup.select("table tbody tr")
        if not rows:
            rows = soup.select("tr")

        records: List[DebateRecord] = []
        if not rows:
            logger.warning("No artifact entries found at offset %s", offset)
            return records

        for row in rows:
            record = self._parse_row(row)
            if record:
                records.append(record)

        if not records:
            logger.warning("Parsed zero debate records from offset %s", offset)
        return records

    def _parse_row(self, row) -> Optional[DebateRecord]:
        links = row.select("a[href]")
        if not links:
            return None

        detail_href: Optional[str] = None
        title_text: Optional[str] = None

        for link in links:
            href = link.get("href", "").strip()
            if not href:
                continue
            text = link.get_text(strip=True)
            if "?view_type=browse" in href or (text and text.lower().startswith("view")):
                detail_href = href.split("?")[0]
            elif href.startswith("/handle/") and not href.endswith(".pdf"):
                if not title_text:
                    title_text = text
                    if not detail_href:
                        detail_href = href

        if not detail_href:
            return None

        cells = [cell.get_text(strip=True) for cell in row.find_all("td")]
        date = cells[0] if cells else "Unknown"

        session: Optional[str] = None
        if len(cells) >= 3:
            # Assume second cell holds section/session information in table layout.
            session = cells[1] or None

        title = title_text or (cells[1] if len(cells) > 1 else "Untitled")

        return DebateRecord(
            title=title or "Untitled",
            date=date or "Unknown",
            session=session,
            url=detail_href,
        )

    def _parse_date(self, value: str) -> Optional[pendulum.Date]:
        if not value or value.lower() == "unknown":
            return None
        try:
            return pendulum.parse(value, strict=False).date()
        except Exception:
            return None

    async def iter_records(self) -> AsyncIterator[DebateRecord]:
        """Iterate over debate records within the configured time window."""
        offset = 0
        while True:
            try:
                page_records = await self._fetch_page(offset)
            except httpx.HTTPError as exc:
                logger.error("Failed to fetch listing at offset %s: %s", offset, exc)
                break

            if not page_records:
                break

            stop = False
            for record in page_records:
                if self._cutoff_date:
                    parsed_date = self._parse_date(record.date)
                    if parsed_date and parsed_date < self._cutoff_date:
                        stop = True
                        break
                yield record

            offset += self._page_size
            if stop:
                break

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "DebateCatalog":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


async def fetch_latest_records(settings: ScraperSettings, limit: int = 5) -> List[DebateRecord]:
    """Convenience helper returning the most recent debate records."""
    async with DebateCatalog(settings) as catalog:
        results: List[DebateRecord] = []
        async for record in catalog.iter_records():
            results.append(record)
            if limit and limit > 0 and len(results) >= limit:
                break
    return results
