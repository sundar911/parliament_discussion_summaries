# Parliament Discussion Summaries

Tooling to download, process, and summarise Lok Sabha debates from the eParliament portal. The project targets local execution on an Apple Silicon Mac (M4, 16 GB) without recurring cloud costs.

## Project Goals
- Scrape the past five years of Lok Sabha debate PDFs from `https://eparlib.sansad.in/handle/123456789/7`.
- Extract bilingual text, translate Hindi passages to English, and generate readable summaries.
- Discover and tag broad debate themes so users can filter by topic (e.g., infrastructure, employment).
- Deliver an easily runnable local pipeline with resumable stages and transparent storage.

## Repository Layout
```
├── data/
│   ├── raw/             # Downloaded PDFs
│   └── processed/
│       ├── original/    # Page-wise original text (Hindi + English)
│       └── english/     # Page-wise translated text (English only)
├── logs/                # Structured run logs
├── notebooks/           # Experiments, EDA, model evaluation
├── src/parliament_summaries/
│   ├── __init__.py
│   ├── config/          # Configuration models & defaults
│   ├── ingest/          # Scrapers, downloaders, metadata storage
│   ├── processing/      # Text extraction, translation, summarisation
│   ├── topics/          # Topic modelling, tagging utilities
│   └── utils/           # Shared helpers (io, batching, logging)
├── tests/               # Unit and integration tests
└── requirements.txt     # Python dependencies
```

## Key Components
- **Scraper**: Navigates the eParliament listing, respects rate limits, and downloads PDFs with metadata into `data/raw`.
- **Document Pipeline**:
  - Extract text using `pdfminer.six` (vector PDFs) with an OCR fallback (`pytesseract`) for scanned pages.
  - Detect language segments (`langdetect` / `fasttext`) and translate Hindi to English via IndicTrans2 or NLLB models running locally.
  - Summarise debates using transformer-based summarisation models with an option to batch on Metal (Apple GPU) or CPU.
- **Topic Discovery**: Generates embeddings (`sentence-transformers`) and clusters debates to propose 10 high-level themes; mapping evolves with more data.
- **Storage**: Persists metadata, transcripts, translations, summaries, and topic assignments in SQLite (via `sqlmodel`) alongside JSON artefacts.
- **CLI**: Provides commands to sync new debates, process pending documents, and export summaries per topic.

## Environment
- Python 3.10+ (recommended via `pyenv` or `conda`) with Apple Silicon wheels.
- Homebrew package: `tesseract` with the Hindi language data (e.g. `brew install tesseract tesseract-lang`).
- Optional GPU acceleration (Metal / CUDA) improves translation speed.
- Optional: `huggingface_hub` cache directory with enough disk (~10 GB) for models.

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Quick Test With Sample PDF
```bash
python -m parliament_summaries.cli process --pdf lsd_18_II_24-07-2024-2.pdf
```
This invokes the extraction + translation stubs + summariser pipeline on the provided sample file and stores results under `data/processed/`.

### Configuration
Edit `src/parliament_summaries/config/defaults.yaml` to adjust scrape ranges, rate limits, and processing toggles. Per-run overrides can be supplied via CLI flags or environment variables.

## Running the Pipeline (outline)
1. **Scrape metadata**  
   ```bash
   python -m parliament_summaries.cli scrape --limit 0 --years-back 10
   ```
2. **Process documents**  
   ```bash
   python -m parliament_summaries.cli process
   ```
   This writes page-wise originals to `data/processed/original/` and English translations to `data/processed/english/`.
3. **Generate topic summaries**  
   ```bash
   python -m parliament_summaries.cli topics
   ```

## Next Steps
- Implement and test scraping + download flow.
- Integrate text extraction with PDF structure heuristics from sample file.
- Benchmark translation/summarisation models on Apple Silicon (MPS) vs CPU.
- Define validation checks and publish sample outputs.
