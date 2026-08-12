# Know My Professor

A RAG chatbot over [Northeastern Khoury](https://www.khoury.northeastern.edu/people/)
faculty profiles. Ask *"who at Khoury works on programming languages?"* and get
cited answers.

- **Website:** https://kmp-frontend-309233821309.us-central1.run.app
- **API:** https://kmp-api-309233821309.us-central1.run.app

## Architecture

```
scrape Khoury directory ──► gs://know-my-professor-raw/profiles/{slug}.json
crawl faculty sites + Gemini extract ──► .../weblinks/{slug}.json
ingest: chunk + Mistral embed (1024d) ──► Pinecone (know-my-professor-m1024)

User ─► Streamlit ─► /chat API:  Mistral query embed ─► Pinecone top-K
                                 ─► Llama-4-Maverick ─► answer + citations
```

Three Cloud Run **Jobs** (scrape, weblinks, ingest) run on monthly crons; two
Cloud Run **Services** (api, frontend) auto-deploy from `main` via Cloud Build.

## Repo layout

```
core/           RAG brain — pipeline · retrieval (Retriever ABC) · llm (Generator ABC + registry)
preprocessing/  sources/ (Source ABC + registry: profiles, weblinks) · ingest/
shared/         config · settings (env) · schemas (wire contract) · embeddings (Embedder ABC
                + registry) · retry · gcs
serving/        api/ (FastAPI, versioned /v1/chat) · frontend/ (Streamlit + api_client)
evaluation/     golden question set + recall@k / MRR / citation scoring
deploy/         one Dockerfile (--build-arg COMPONENT) + Cloud Build configs
tests/          offline pytest + opt-in live e2e
```

One installable package; five deployables built from the single `deploy/Dockerfile`.

### Extension points

Both axes are registry-driven, so adding one is a new module plus one line:

| To add | Write | Register in |
| --- | --- | --- |
| An embedding provider | `Embedder` subclass in `shared/embeddings/` | `_EMBEDDERS` in `shared/embeddings/__init__.py` |
| A chat provider | `Generator` subclass in `core/llm/` | `_GENERATORS` in `core/llm/__init__.py` |
| A data source (e.g. courses) | `Source` subclass in `preprocessing/sources/<name>/` | `SOURCES` in `preprocessing/sources/registry.py` |

Ingest names no source and the API names no provider, so neither has to change.
Registries validate their invariants at import: section keys must be unique
across sources (a collision would silently overwrite vectors in Pinecone), and
an embedder's dimension must match the index it will write to.

Note that a new **data source** is additive, but a new **embedding provider**
with a different vector width needs a new Pinecone index and a full re-ingest —
`build_embedder` refuses to start rather than fail part-way through an upsert.

## Local setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # installs all component extras (editable)
python -m pytest tests/ -q             # offline tests
ruff check .                           # lint + import-layering rules
python -m preprocessing.sources.profiles.runner --limit 5   # run a stage locally
python -m evaluation.run_eval          # retrieval quality vs. the golden set (live)
```

Lint and tests run on every PR into `dev`/`main` (`.github/workflows/test.yml`)
and again in Cloud Build before either Service image is built, so a red commit
cannot deploy.

Secrets live in repo-root `.env` (gitignored): `MISTRAL_API_KEY`,
`PINECONE_API_KEY`, `LLAMA_API_KEY`, `GEMINI_API_KEY`. In Cloud Run these are env
vars on the service/job, never committed.

## Stack & constraints

- **Embeddings:** Mistral `mistral-embed-2312` (1024-dim, batched). **Generation:**
  Llama-4-Maverick (Meta Llama API). **Vectors:** Pinecone serverless, cosine,
  vector ID `{slug}#{section_type}`.
- **GCP only**, **zero cost** (everything inside free tiers), **production-level**
  (least-privilege service accounts, idempotent scrape, monthly refresh crons).
