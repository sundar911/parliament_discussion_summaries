from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import typer
import httpx

from .config.defaults import load_defaults
from .ingest.catalog import DebateRecord, fetch_latest_records
from .ingest.downloader import DebateDownloader
from .ingest.store import DebateMetadataStore
from .processing.pipeline import ProcessingPipeline
from .processing.text_extraction import PageBlock
from .processing.translation import TranslationService
from .topics.modeling import TopicModel
from .utils.io import ensure_directory
from .utils.logging import configure_logging, get_logger

app = typer.Typer(help="Parliament debate summarisation toolkit.")
logger = get_logger(__name__)


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


@app.command()
def show_config() -> None:
    """Print the current configuration defaults."""
    cfg = load_defaults()
    typer.echo(cfg.model_dump_json(indent=2))


@app.command()
def scrape(
    limit: int = typer.Option(50, help="Number of recent records to pull; specify 0 for no limit."),
    dry_run: bool = False,
    years_back: Optional[int] = typer.Option(None, help="Override default lookback window in years."),
) -> None:
    """Fetch debate metadata and optionally download PDFs."""
    cfg = load_defaults()
    if years_back is not None:
        cfg.scraper.years_back = years_back
    configure_logging(cfg.runtime.logs_dir)
    ensure_directory(Path(cfg.runtime.raw_dir))

    async def _run():
        try:
            records = await fetch_latest_records(cfg.scraper, limit=limit)
        except httpx.HTTPError as exc:
            logger.error("Network error while fetching listings: %s", exc)
            typer.echo("Failed to reach eParliament portal. Please check network/SSL settings.")
            return
        except Exception as exc:  # pragma: no cover
            logger.exception("Unexpected error during scraping: %s", exc)
            typer.echo("Unexpected error while scraping; see logs for details.")
            return

        if not records:
            typer.echo("No records found – the portal may have changed structure or blocked requests.")
            return

        ensure_directory(Path(cfg.runtime.data_root))
        metadata_path = Path(cfg.runtime.data_root) / "metadata.json"
        metadata = [asdict(record) for record in records]
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"Wrote metadata for {len(records)} debates to {metadata_path}")

        store = DebateMetadataStore(cfg.runtime)
        to_download: List[DebateRecord] = []
        for record in records:
            store.upsert(record)
            existing = store.get(record.url)
            if not existing or not existing.get("downloaded_path"):
                to_download.append(record)
            else:
                logger.info("Already have PDF for %s; skipping download", record.url)

        if dry_run:
            return

        if not to_download:
            typer.echo("No new PDFs to download; existing files are up to date.")
            return

        typer.echo(f"Downloading {len(to_download)} new PDFs...")

        async with DebateDownloader(cfg.runtime, cfg.scraper) as downloader:
            paths = await downloader.bulk_download(to_download)
            for record, path in zip(to_download, paths):
                store.upsert(record, downloaded_path=str(path))

    loop = get_event_loop()
    loop.run_until_complete(_run())


@app.command()
def process(pdf: Optional[Path] = typer.Option(None, help="Specific PDF to process.")) -> None:
    """Run text extraction + summarisation for downloaded PDFs."""
    cfg = load_defaults()
    configure_logging(cfg.runtime.logs_dir)
    ensure_directory(Path(cfg.runtime.processed_dir))

    pipeline = ProcessingPipeline(cfg.runtime, cfg.processing)

    if pdf:
        targets = [pdf]
    else:
        targets = sorted(Path(cfg.runtime.raw_dir).glob("*.pdf"))

    if not targets:
        typer.echo("No PDFs found. Run the scrape command first or specify --pdf.")
        raise typer.Exit(code=1)

    async def _run():
        for path in targets:
            await pipeline.process_pdf(path)

    loop = get_event_loop()
    loop.run_until_complete(_run())


@app.command()
def topics(source: Optional[Path] = typer.Option(None, help="Directory of processed JSON files.")) -> None:
    """Generate topic clusters from processed English documents."""
    cfg = load_defaults()
    configure_logging(cfg.runtime.logs_dir)

    default_base = Path(cfg.runtime.processed_dir) / "english"
    base = source or default_base
    if not base.exists():
        typer.echo(f"Topic source directory {base} does not exist.")
        raise typer.Exit(code=1)

    documents = []
    for json_path in sorted(base.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        pages = payload.get("pages", [])
        text = "\n".join(page.get("text", "") for page in pages)
        documents.append(text.strip())

    documents = [doc for doc in documents if doc]
    if not documents:
        typer.echo("No processed summaries found.")
        raise typer.Exit(code=1)

    topic_model = TopicModel(cfg.topics)
    assignments = topic_model.fit(documents)

    output_path = Path(cfg.runtime.data_root) / "topics.json"
    serialisable = [
        {"topic_id": assignment.topic_id, "label": assignment.label, "sentences": assignment.sentences}
        for assignment in assignments
    ]
    output_path.write_text(json.dumps(serialisable, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Wrote topic assignments to {output_path}")


def main():
    app()


if __name__ == "__main__":
    main()
