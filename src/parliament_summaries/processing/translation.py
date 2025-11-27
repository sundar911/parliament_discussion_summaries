from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..config.models import ProcessingSettings
from ..utils.logging import get_logger
from .text_extraction import PageBlock

logger = get_logger(__name__)

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Mapping from language detection outputs to IndicTrans2 language tags.
INDIC_LANGUAGE_TAGS = {
    "hi": "hin_Deva",
    "hin": "hin_Deva",
    "mr": "mar_Deva",
    "mar": "mar_Deva",
    "bn": "ben_Beng",
    "ben": "ben_Beng",
    "ta": "tam_Taml",
    "tam": "tam_Taml",
    "te": "tel_Telu",
    "tel": "tel_Telu",
    "ml": "mal_Mlym",
    "mal": "mal_Mlym",
    "gu": "guj_Gujr",
    "guj": "guj_Gujr",
    "pa": "pan_Guru",
    "pan": "pan_Guru",
    "kn": "kan_Knda",
    "kan": "kan_Knda",
    "ka": "kan_Knda",
    "or": "ory_Orya",
    "ory": "ory_Orya",
    "od": "ory_Orya",
    "ne": "npi_Deva",
    "npi": "npi_Deva",
    "as": "asm_Beng",
    "asm": "asm_Beng",
    "ur": "urd_Arab",
    "urd": "urd_Arab",
    "sd": "snd_Deva",
    "snd": "snd_Deva",
    "mai": "mai_Deva",
    "maithili": "mai_Deva",
    "mag": "mag_Deva",
    "bho": "bho_Deva",
    "gom": "gom_Deva",
    "doi": "doi_Deva",
    "en": "eng_Latn",
}

DEFAULT_SOURCE_TAG = "hin_Deva"
TARGET_LANGUAGE_TAG = "eng_Latn"


@dataclass
class TranslatedBlock:
    page_number: int
    source_language: Optional[str]
    translated_text: str


class TranslationService:
    """Translate Indic text segments to English while retaining existing English passages."""

    def __init__(self, settings: ProcessingSettings) -> None:
        self.settings = settings
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._device = self._resolve_device(settings.device_preference)
        self._available = False
        self._attempted = False
        self._target_prefix = TARGET_LANGUAGE_TAG

    def _resolve_device(self, preference: str) -> torch.device:
        if preference == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if preference == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self) -> bool:
        if self._available:
            return True
        if self._attempted:
            return False
        model_name = self.settings.translation_model
        if not model_name:
            logger.warning("No translation model configured; skipping translation.")
            return False

        try:
            logger.info("Loading translation model %s on %s", model_name, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(
                self._device
            )
            self._available = True
        except Exception as exc:
            logger.warning("Failed to load translation model %s: %s", model_name, exc)
            self._available = False
        finally:
            self._attempted = True
        return self._available

    def _segment_text(self, text: str) -> List[Tuple[bool, str]]:
        segments: List[Tuple[bool, str]] = []
        buffer: List[str] = []
        current_flag: Optional[bool] = None

        for line in text.splitlines():
            has_devanagari = bool(DEVANAGARI_RE.search(line))
            if current_flag is None:
                current_flag = has_devanagari
            if has_devanagari == current_flag:
                buffer.append(line)
            else:
                segments.append((current_flag, "\n".join(buffer)))
                buffer = [line]
                current_flag = has_devanagari
        if buffer:
            segments.append((current_flag, "\n".join(buffer)))
        return segments

    def _chunk_text(self, text: str, max_chars: int = 512) -> List[str]:
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for line in text.splitlines():
            line_len = len(line)
            if current and current_len + line_len + 1 > max_chars:
                chunks.append("\n".join(current))
                current = [line]
                current_len = line_len
            else:
                current.append(line)
                current_len += line_len + 1

        if current:
            chunks.append("\n".join(current))
        return chunks

    def _batch_chunks(self, chunks: List[str], size: int = 4) -> Iterable[List[str]]:
        for idx in range(0, len(chunks), size):
            yield chunks[idx : idx + size]

    def _lang_tag(self, language: Optional[str]) -> str:
        if not language:
            return DEFAULT_SOURCE_TAG

        normalised = language.lower().split("-", 1)[0]
        candidates = [
            normalised,
            normalised[:3] if len(normalised) >= 3 else normalised,
            normalised[:2] if len(normalised) >= 2 else normalised,
        ]

        for candidate in candidates:
            if candidate and candidate in INDIC_LANGUAGE_TAGS:
                return INDIC_LANGUAGE_TAGS[candidate]
        return DEFAULT_SOURCE_TAG

    def _translate_chunks(self, chunks: List[str], src_tag: str) -> List[str]:
        if not self._load_model():
            return chunks
        assert self._tokenizer is not None and self._model is not None  # for mypy

        prefixed_inputs = [
            f"{src_tag} {self._target_prefix} {chunk.strip()}" for chunk in chunks if chunk.strip()
        ]
        if not prefixed_inputs:
            return [""] * len(chunks)

        translations: List[str] = []
        with torch.inference_mode():
            tokenized = self._tokenizer(
                prefixed_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            tokenized = {key: value.to(self._device) for key, value in tokenized.items()}
            generated = self._model.generate(**tokenized, max_new_tokens=256, use_cache=False)
            decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            translations.extend(decoded)

        iterator = iter(translations)
        results: List[str] = []
        for chunk in chunks:
            if chunk.strip():
                results.append(next(iterator, ""))
            else:
                results.append("")
        return results

    def translate_text(self, text: str, source_language: Optional[str] = None) -> str:
        if not text.strip():
            return text
        if not DEVANAGARI_RE.search(text):
            return text

        translated_segments: List[str] = []
        src_tag = self._lang_tag(source_language)
        for is_devanagari, segment in self._segment_text(text):
            if not segment.strip():
                translated_segments.append(segment)
                continue
            if not is_devanagari:
                translated_segments.append(segment)
                continue

            chunks = self._chunk_text(segment)
            translated_chunks: List[str] = []
            for batch in self._batch_chunks(chunks):
                translated_chunks.extend(self._translate_chunks(batch, src_tag))
            translated_segments.append("\n".join(translated_chunks).strip())
        return "\n".join(translated_segments).strip() or text

    def translate_blocks(self, blocks: Iterable[PageBlock]) -> List[TranslatedBlock]:
        translated: List[TranslatedBlock] = []
        for block in blocks:
            text = block.text or ""
            translated_text = self.translate_text(text, block.language)
            translated.append(
                TranslatedBlock(
                    page_number=block.page_number,
                    source_language=block.language,
                    translated_text=translated_text if translated_text else text,
                )
            )
        return translated
