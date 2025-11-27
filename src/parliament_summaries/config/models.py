from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RuntimeSettings(BaseModel):
    data_root: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    logs_dir: str = "logs"
    sqlite_path: str = "data/parliament.db"
    cache_dir: str = "~/.cache/parliament-summaries"


class ScraperSettings(BaseModel):
    base_url: str = "https://eparlib.sansad.in"
    handle_path: str = "/handle/123456789/7"
    user_agent: str = Field(
        default="ParliamentSummariesBot/0.1 (+https://github.com/user/parliament_discussion_summaries)"
    )
    request_interval_seconds: float = 1.5
    max_retries: int = 4
    timeout_seconds: int = 30
    years_back: int = 5


class ProcessingSettings(BaseModel):
    batch_size: int = 2
    max_pages: Optional[int] = None
    enable_ocr: bool = True
    ocr_dpi: int = 300
    ocr_languages: str = "eng+hin"
    language_detection: str = "fasttext"
    translation_model: str = "ai4bharat/indictrans2-hi-en"
    summarisation_model: str = "google/pegasus-xsum"
    max_summary_tokens: int = 256
    device_preference: str = "mps"


class TopicSettings(BaseModel):
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    cluster_method: str = "bertopic"
    target_topics: int = 10
    refresh_interval_days: int = 7


class AppConfig(BaseModel):
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    scraper: ScraperSettings = Field(default_factory=ScraperSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    topics: TopicSettings = Field(default_factory=TopicSettings)
