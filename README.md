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
core/           RAG brain — pipeline, query (embed), retrieval (Pinecone), llm (Llama)
preprocessing/  sources/profiles (scrape) · sources/weblinks (crawl+extract) · ingest
shared/         config.py (single source of truth) · gcs.py
serving/        api/app.py (FastAPI) · frontend/app.py (Streamlit)
deploy/         one Dockerfile (--build-arg COMPONENT) + Cloud Build configs
tests/          offline pytest + opt-in live e2e
```

One installable package; five deployables built from the single `deploy/Dockerfile`.

## Local setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # installs all component extras (editable)
python -m pytest tests/ -q             # offline tests
python -m preprocessing.sources.profiles.runner --limit 5   # run a stage locally
```

Secrets live in repo-root `.env` (gitignored): `MISTRAL_API_KEY`,
`PINECONE_API_KEY`, `LLAMA_API_KEY`, `GEMINI_API_KEY`. In Cloud Run these are env
vars on the service/job, never committed.

## Stack & constraints

- **Embeddings:** Mistral `mistral-embed-2312` (1024-dim, batched). **Generation:**
  Llama-4-Maverick (Meta Llama API). **Vectors:** Pinecone serverless, cosine,
  vector ID `{slug}#{section_type}`.
- **GCP only**, **zero cost** (everything inside free tiers), **production-level**
  (least-privilege service accounts, idempotent scrape, monthly refresh crons).
