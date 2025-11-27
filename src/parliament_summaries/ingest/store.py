from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import sqlite_utils
from sqlite_utils.db import NotFoundError

from ..config.models import RuntimeSettings
from ..utils.io import ensure_directory
from ..utils.logging import get_logger
from .catalog import DebateRecord

logger = get_logger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DebateMetadataStore:
    """Persist debate metadata for resumable processing."""

    TABLE_NAME = "debates"

    def __init__(self, runtime: RuntimeSettings) -> None:
        ensure_directory(Path(runtime.data_root))
        self.db_path = Path(runtime.sqlite_path)
        self.db = sqlite_utils.Database(self.db_path)
        self._prepare()

    def _prepare(self) -> None:
        if self.TABLE_NAME in self.db.table_names():
            return
        logger.info("Initialising metadata database at %s", self.db_path)
        self.db[self.TABLE_NAME].create(
            {
                "url": str,
                "title": str,
                "date": str,
                "session": str,
                "pdf_url": str,
                "downloaded_path": str,
                "processed_path": str,
                "last_updated": str,
            },
            pk="url",
        )

    def upsert(self, record: DebateRecord, downloaded_path: Optional[str] = None) -> None:
        existing = self.get(record.url)
        existing_downloaded = existing.get("downloaded_path") if existing else None
        existing_processed = existing.get("processed_path") if existing else None

        resolved_downloaded = (
            downloaded_path if downloaded_path is not None else existing_downloaded
        )

        payload = {
            "url": record.url,
            "title": record.title,
            "date": record.date,
            "session": record.session,
            "pdf_url": record.pdf_url,
            "downloaded_path": resolved_downloaded,
            "processed_path": existing_processed,
            "last_updated": _utc_now(),
        }
        self.db[self.TABLE_NAME].upsert(payload, pk="url")

    def mark_processed(self, url: str, processed_path: str) -> None:
        self.db[self.TABLE_NAME].update(
            url,
            {
                "processed_path": processed_path,
                "last_updated": _utc_now(),
            },
        )

    def iter_pending_downloads(self) -> Iterable[dict]:
        return self.db[self.TABLE_NAME].rows_where("downloaded_path IS NULL")

    def iter_pending_processing(self) -> Iterable[dict]:
        return self.db[self.TABLE_NAME].rows_where(
            "downloaded_path IS NOT NULL AND processed_path IS NULL"
        )

    def get(self, url: str) -> Optional[dict]:
        try:
            return self.db[self.TABLE_NAME].get(url)
        except NotFoundError:
            return None
