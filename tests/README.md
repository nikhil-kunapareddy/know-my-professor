# tests/

Unit tests for the pure logic in `scraper/`, `ingest/`, and `weblinks/`.

```bash
python3 -m pytest tests/ -q          # all
python3 -m pytest tests/test_weblinks.py -q
```

No network, GCS, Gemini, or Pinecone access — everything runs offline.

## How it works (see `conftest.py`)

The three source folders are **not packages**: each runs with its own folder on
`sys.path` (`python scraper.py`), and several reuse module names (`config.py`,
`gcs.py`). So tests can't plain-`import` them. The `load(folder, module)` fixture
puts one folder on `sys.path`, purges the shared flat names, and imports the
module fresh — so `from config import ...` resolves to the folder under test.

Some source modules import cloud libs that aren't installed (or are the wrong
version) in a bare environment: `trafilatura`, `google.generativeai`,
`google.cloud.storage`, and an incompatible `pinecone`. `conftest.py` installs
lightweight stubs for those so the pure logic stays importable. Tests never call
the stubbed APIs — they exercise crawling, parsing, chunking, hashing, and
record-building, not live I/O.

## Coverage

- `test_scraper.py` — `profile_parser.parse_profile` (header/aside/accordion),
  `fetcher.extract_total_pages` / `extract_profile_urls`.
- `test_ingest.py` — `chunking` content-hash + `enrichment_to_chunks` (source-url,
  disjoint ids, empty-drop), `pinecone_store.fetch_existing_hashes` (batching,
  dedup), and the "same hash skips / changed hash re-embeds" rule.
- `test_weblinks.py` — one-hop link selection, success guard, `page_hash`
  (determinism + `SCHEMA_VERSION` sensitivity), `clean_pages` sort/label/truncate,
  `website_url`, `build_record`.
