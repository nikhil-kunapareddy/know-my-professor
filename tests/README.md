# tests/

Offline unit tests for `core/`, `preprocessing/`, `shared/`, `serving/`, and
`evaluation/`.

```bash
.venv/bin/python -m pytest tests/ -q       # all
.venv/bin/python -m pytest tests/test_api.py -q
.venv/bin/ruff check .                     # lint + import-layering rules
```

No network, GCS, Gemini, Pinecone, or model access — everything runs offline.
These run on every PR into `dev`/`main` and again in Cloud Build before either
Service image is built.

## How it works (see `conftest.py`)

`pyproject.toml` sets `pythonpath = ["."]`, so tests import the package directly
(`from preprocessing.sources.registry import SOURCES`).

`conftest.py` installs lightweight stubs for cloud libs that may be missing in a
bare environment (`pinecone`, `mistralai`, `trafilatura`, `google.generativeai`,
`google.cloud.storage`). Where the real libraries are present they are used.
Tests never call a live API — providers are injected as fakes.

`deepeval` is deliberately NOT stubbed. It is an opt-in extra (`pip install -e
".[eval]"`) that CI does not install, and a stub of a judging library would
assert nothing; the tests that need it call `pytest.importorskip("deepeval")`
and skip instead. 52 of the 68 tests in `test_deepeval_eval.py` run without it.

## Coverage

- `test_scraper.py` — `ProfileParser.parse` (header/aside/accordion),
  `DirectoryFetcher.extract_total_pages` / `extract_profile_urls`.
- `test_colleges.py` — multi-college scraping: the college registry, vector-id
  namespacing (`{college}-{slug}`, Khoury stays bare), that a slug shared by two
  colleges cannot collide, that `college` metadata never moves `content_hash`,
  runner wiring (parser selection, URL-cache keys, `_Pacer`), and the LLM profile
  parser against a fake client — including the thin-page, truncation, and
  unparseable-output paths, which are the ones that would otherwise store a
  hollow profile. Also pins that `ProfileSource` and `WeblinksSource` mint
  entity ids through the same `sources/entities.py` helper: if they disagreed,
  a College of Science professor's website would enrich the Khoury professor
  with the same slug.
- `test_courses.py` — the course corpus (catalog + term schedule): catalog parsing (title/credits/requisites,
  non-breaking spaces, unreadable blocks), entity ids that cannot collide with
  professor slugs, and the **namespace** rules — that both course sources share
  one namespace, that people stay in the default one, that the registry rejects a
  dependent source alone in a namespace, that `_collect_chunks` groups by
  namespace, and that `RAGPipeline` passes it through. Namespace bugs never
  raise; they silently return nothing or write to the wrong partition.
- `test_ingest.py` — per-source chunk rendering, registry invariants (duplicate
  section keys / prefixes rejected), `_collect_chunks` over the registry
  (entity scoping, `--limit` semantics, stale-enrichment skip), and
  `PineconeStore.fetch_existing_hashes` batching + the re-embed rule.
- `test_chunk_golden.py` — **the re-ingest guard.** Pins chunk text and metadata
  byte-for-byte against `fixtures/chunk_golden.json`, walking the registry so a
  new source needs fixtures, not edits here. Chunk text is hashed into
  `content_hash`, which decides whether ingest re-embeds, so any rendering change
  silently invalidates the index. If this fails, either the change was
  unintended, or it was intended and the index owes you a full re-ingest.
- `test_providers.py` — embedder/generator registries, the dimension guard, the
  shared backoff policy, and that query embedding reuses the document path.
- `test_core.py` — `RAGPipeline` orchestration: ordered sources, score floor,
  filter pass-through, stage timings, no-match and empty-answer fallbacks.
- `test_api.py` — routing (`/v1/chat` + legacy `/chat`), health/readiness,
  citation filtering, error-status mapping, and that upstream error text never
  reaches the client.
- `test_frontend.py` — `ChatClient` transport (timeout, retry policy, error
  translation, schema validation) and citation formatting.
- `test_settings.py` — env parsing/validation and the Pinecone dimension guard.
- `test_weblinks.py` — `SiteCrawler` one-hop selection + fetch encoding,
  `Extractor` guard/`page_hash`/`clean_pages`, and that the Gemini extraction
  schema matches the source's declared section keys.
- `test_eval.py` — evaluation scoring (recall@k, MRR, citation precision),
  golden-file parsing and its invariants (every case names slugs or expects a
  refusal; every registered section type has a case), no-answer scoring
  including refusals that add a caveat, and reproducible `--sample` selection.
- `test_deepeval_eval.py` — the judged framework's pure half (sample building,
  the on-disk dump format including dumps written before the DeepEval port, the
  stage/metric table, the scoring loop's skip-vs-error rules, report aggregation
  and gates) always; plus, only when the `eval` extra is installed, the drift
  guards — that every metric still takes the inputs the table claims, that
  G-Eval's real parameters are read rather than its bare annotation, that no
  sampling parameter can reach the Anthropic API, that a judged metric is never
  silently defaulted to OpenAI, and that a judge cache hit is really a hit.
- `test_e2e_live.py` — opt-in (`KMP_LIVE_E2E=1`); drives the real pipeline
  against the live index.

## Regenerating the chunk golden file

Only when a rendering change is intentional — and it means a full re-ingest:

```bash
.venv/bin/python - <<'PY'
import json, pathlib
from preprocessing.sources.registry import SOURCES, chunks_for
fx = json.loads(pathlib.Path("tests/fixtures/source_records.json").read_text())
dump = lambda cs: [{"vector_id": c.vector_id, "text": c.text, "metadata": c.metadata} for c in cs]
golden = {}
for source in SOURCES:
    entries = []
    for r in fx[source.name]:
        entry = {"slug": r.get("slug"), "chunks": dump(chunks_for(source.name, r))}
        if source.name == "profiles":
            entry["is_substantive"] = source.is_ingestable(r)
        entries.append(entry)
    golden[source.name] = entries
pathlib.Path("tests/fixtures/chunk_golden.json").write_text(
    json.dumps(golden, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
```
