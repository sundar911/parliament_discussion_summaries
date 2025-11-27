from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from langdetect import detect_langs

from ..config.models import ProcessingSettings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PageBlock:
    page_number: int
    text: str
    language: Optional[str] = None


@dataclass
class DocumentExtraction:
    source_path: Path
    blocks: List[PageBlock]

    def to_json(self) -> str:
        return json.dumps(
            {
                "source_path": str(self.source_path),
                "blocks": [
                    {"page": block.page_number, "language": block.language, "text": block.text}
                    for block in self.blocks
                ],
            },
            ensure_ascii=False,
        )


class TextExtractor:
    """Extract page-wise text using OCR (via pypdfium2) or native PDF text."""

    def __init__(self, settings: ProcessingSettings) -> None:
        self.settings = settings
        self.ocr_languages = settings.ocr_languages or "eng+hin"
        self.ocr_dpi = settings.ocr_dpi or 300
        self._ocr_scale = max(self.ocr_dpi / 72.0, 1.0)

    def _detect_language(self, text: str) -> Optional[str]:
        snippet = text.strip().replace("\n", " ")
        if not snippet:
            return None
        try:
            return detect_langs(snippet)[0].lang
        except Exception:
            return None

    def _run_ocr(self, image, pdf_path: Path, page_number: int) -> str:
        text = ""
        try:
            text = pytesseract.image_to_string(image, lang=self.ocr_languages)
        except pytesseract.TesseractError as exc:
            logger.error("OCR failed on %s page %s: %s", pdf_path, page_number, exc)
            if "Failed loading language" in str(exc) and "hin" in self.ocr_languages:
                try:
                    text = pytesseract.image_to_string(image, lang="eng")
                except pytesseract.TesseractError as inner_exc:
                    logger.error(
                        "English-only OCR fallback failed on %s page %s: %s",
                        pdf_path,
                        page_number,
                        inner_exc,
                    )
                    text = ""
        finally:
            try:
                image.close()
            except Exception:
                pass

        return text.replace("\x0c", "").strip()

    def _extract_with_ocr(self, pdf_path: Path) -> DocumentExtraction:
        logger.info("Extracting (OCR) text from %s", pdf_path)
        blocks: List[PageBlock] = []

        try:
            document = pdfium.PdfDocument(str(pdf_path))
        except Exception as exc:
            logger.error("Failed to open %s with pdfium: %s", pdf_path, exc)
            return DocumentExtraction(source_path=pdf_path, blocks=blocks)

        total_pages = len(document)
        if total_pages == 0:
            logger.warning("No pages detected in %s", pdf_path)
            document.close()
            return DocumentExtraction(source_path=pdf_path, blocks=blocks)

        if self.settings.max_pages:
            total_pages = min(total_pages, self.settings.max_pages)

        for index in range(total_pages):
            page_number = index + 1
            try:
                page = document.get_page(index)
            except Exception as exc:
                logger.error("Unable to access page {} of {}: {}", page_number, pdf_path, exc)
                continue

            try:
                bitmap = page.render(scale=self._ocr_scale)
                pil_image = bitmap.to_pil()
            except Exception as exc:
                logger.error("Rendering failed for {} page {}: {}", pdf_path, page_number, exc)
                page.close()
                continue

            text = self._run_ocr(pil_image, pdf_path, page_number)
            bitmap.close()
            language = self._detect_language(text) if text else None
            blocks.append(PageBlock(page_number=page_number, text=text, language=language))
            page.close()

        document.close()
        return DocumentExtraction(source_path=pdf_path, blocks=blocks)

    def _extract_with_pdfplumber(self, pdf_path: Path) -> DocumentExtraction:
        logger.info("Extracting (native text) from %s", pdf_path)
        blocks: List[PageBlock] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                language = self._detect_language(text) if text else None
                blocks.append(PageBlock(page_number=page_idx, text=text, language=language))
                if self.settings.max_pages and page_idx >= self.settings.max_pages:
                    break
        return DocumentExtraction(source_path=pdf_path, blocks=blocks)

    def extract(self, pdf_path: Path) -> DocumentExtraction:
        if self.settings.enable_ocr:
            return self._extract_with_ocr(pdf_path)
        return self._extract_with_pdfplumber(pdf_path)
