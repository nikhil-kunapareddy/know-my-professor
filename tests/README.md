# tests/

Offline unit tests for the pure logic in `core/`, `preprocessing/`, and `shared/`.

```bash
.venv/bin/python -m pytest tests/ -q       # all
.venv/bin/python -m pytest tests/test_weblinks.py -q
```

No network, GCS, Gemini, or Pinecone access — everything runs offline.

## How it works (see `conftest.py`)

The source is a normal package now. `pyproject.toml` sets `pythonpath = ["."]`,
so tests import it directly (`from preprocessing.ingest.chunking import Chunker`).
The old flat-import shim is gone.

`conftest.py` only installs lightweight stubs for cloud libs that may be missing
in a bare environment (`pinecone`, `mistralai`, `trafilatura`,
`google.generativeai`, `google.cloud.storage`). When the real libraries are
present (e.g. in `.venv`) the stubs are skipped. Tests never call the stubbed
APIs — they exercise parsing, chunking, hashing, crawling, and the pipeline
orchestration, not live I/O.

## Coverage

- `test_scraper.py` — `ProfileParser.parse` (header/aside/accordion),
  `DirectoryFetcher.extract_total_pages` / `extract_profile_urls`.
- `test_ingest.py` — `Chunker` content-hash + `enrichment_to_chunks` (source-url,
  disjoint ids, empty-drop), `PineconeStore.fetch_existing_hashes` (batching,
  dedup), and the "same hash skips / changed hash re-embeds" rule.
- `test_weblinks.py` — `SiteCrawler` one-hop link selection + fetch encoding,
  `Extractor` success guard / `page_hash` (determinism + `SCHEMA_VERSION`
  sensitivity) / `clean_pages`, `WeblinksCrawlJob.website_url` / `build_record`.
- `test_core.py` — `RAGPipeline` orchestration (ordered sources, no-match
  short-circuit, empty-answer fallback) and `PromptBuilder` numbering, with fakes.
- `test_e2e_live.py` — opt-in (`KMP_LIVE_E2E=1`); hits the real APIs + index.
