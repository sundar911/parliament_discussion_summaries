from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict

from ..config.models import ProcessingSettings, RuntimeSettings
from ..utils.io import ensure_directory
from ..utils.logging import get_logger
from .text_extraction import DocumentExtraction, TextExtractor
from .translation import TranslationService, TranslatedBlock

logger = get_logger(__name__)


class ProcessingPipeline:
    """Orchestrates PDF extraction and bilingual JSON generation."""

    def __init__(self, runtime: RuntimeSettings, settings: ProcessingSettings) -> None:
        self.runtime = runtime
        self.settings = settings

        base_dir = ensure_directory(Path(runtime.processed_dir))
        self._original_dir = ensure_directory(base_dir / "original")
        self._translated_dir = ensure_directory(base_dir / "english")

        self.extractor = TextExtractor(settings)
        self.translator = TranslationService(settings)

    async def _extract(self, pdf_path: Path) -> DocumentExtraction:
        return await asyncio.to_thread(self.extractor.extract, pdf_path)

    async def _translate(self, extraction: DocumentExtraction) -> Dict[int, TranslatedBlock]:
        translated_blocks = await asyncio.to_thread(self.translator.translate_blocks, extraction.blocks)
        return {block.page_number: block for block in translated_blocks}

    async def _write_json(self, path: Path, payload: Dict) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        await asyncio.to_thread(path.write_text, text, "utf-8")

    async def process_pdf(self, pdf_path: Path) -> Dict[str, Path]:
        extraction = await self._extract(pdf_path)
        translated_map = await self._translate(extraction)

        original_payload = {
            "source_pdf": pdf_path.name,
            "pages": [
                {
                    "page": block.page_number,
                    "language": block.language,
                    "text": block.text,
                }
                for block in extraction.blocks
            ],
        }

        translated_pages = []
        for block in extraction.blocks:
            translated = translated_map.get(block.page_number)
            translated_pages.append(
                {
                    "page": block.page_number,
                    "source_language": translated.source_language if translated else block.language,
                    "text": translated.translated_text if translated else block.text,
                }
            )

        translated_payload = {
            "source_pdf": pdf_path.name,
            "pages": translated_pages,
        }

        original_path = self._original_dir / f"{pdf_path.stem}.json"
        translated_path = self._translated_dir / f"{pdf_path.stem}.json"

        await asyncio.gather(
            self._write_json(original_path, original_payload),
            self._write_json(translated_path, translated_payload),
        )

        logger.info("Wrote processed outputs %s and %s", original_path, translated_path)
        return {"original": original_path, "english": translated_path}
