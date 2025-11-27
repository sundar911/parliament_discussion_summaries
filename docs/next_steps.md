# Next Steps & Validation Checklist

## Immediate Tasks
- Implement resume-friendly scraping that honours the `years_back` window and captures pagination cursors.
- Enrich `DebateRecord` with document identifiers (e.g., item ID) to make deduplication deterministic.
- Harden PDF download logic against transient portal failures (captcha, HTML-only responses).

## Processing Enhancements
- Integrate IndicTrans2 or NLLB translation models with batching and Apple MPS acceleration.
- Add OCR fallback (Tesseract + layout detection) for pages with negligible extracted text.
- Introduce sentence-level segmentation before summarisation to avoid 1024-token truncation.
- Cache intermediate artefacts (per-page JSON) for debugging and incremental reruns.

## Summaries & Topics
- Experiment with multiple summarisation models (Pegasus, BART, LongT5) and measure quality vs runtime.
- Layer extractive summaries or rhetorical role tagging (speaker, question, answer) ahead of abstractive stage.
- Automate topic labeling by pairing clustering with a lightweight keyword lexicon.

## Quality Assurance
- Build pytest-based smoke tests for scraper parsing and text extraction on the sample PDF.
- Establish manual review workflow: spot-check translation fidelity via bilingual volunteers/tools.
- Track processing metrics (pages processed, translation coverage, summarisation latency) in SQLite.

## User Experience
- Draft a lightweight Streamlit or static web viewer once summaries stabilise.
- Provide export utilities (CSV/JSONL) for downstream analysis.
- Add CLI progress bars and human-readable logs for long-running jobs.
