# Know My Professor

A RAG chatbot over [Northeastern Khoury](https://www.khoury.northeastern.edu/people/)
faculty profiles. Ask questions like *"who at Khoury works on programming languages?"*
and get cited answers.

- **Live website:** https://kmp-frontend-309233821309.us-central1.run.app
- **API:** https://kmp-api-309233821309.us-central1.run.app

## Architecture

```
[Cloud Scheduler: monthly-scrape]
        |
        v  (OAuth as kmp-scheduler SA)
[Cloud Run Job: scrape-khoury] --runs as kmp-scraper SA--> [GCS: gs://know-my-professor-raw]
                                                              |
                                                              v
                                                   chunk + embed -> [Pinecone index]

[User] -> [Streamlit on Cloud Run] -> [/chat API on Cloud Run]
                                          | embed query (Gemini gemini-embedding-001)
                                          v
                                       [Pinecone similarity search]
                                          | top-K chunks + prompt
                                          v
                                       [Gemini 2.5 Flash] -> answer + citations
```

## Repo layout

| Directory   | Phase | Purpose                                                        | Deploys to                       |
|-------------|-------|---------------------------------------------------------------|----------------------------------|
| `scraper/`  | 1–5   | Discovers `/people/` URLs, parses profiles into JSON          | Cloud Run **Job** `scrape-khoury`   |
| `ingest/`   | 6     | Chunk + embed + upsert into Pinecone (resumable)              | Cloud Run **Job** `ingest-pinecone` |
| `api/`      | 7     | FastAPI `POST /chat`, `GET /health`                           | Cloud Run **Service** `kmp-api`     |
| `frontend/` | 8     | Streamlit chat UI; POSTs to `{KMP_API_URL}/chat`             | Cloud Run **Service** `kmp-frontend`|

Each component has its own pinned `requirements.txt` and `Dockerfile`. The root
`requirements.txt` is an aggregate for local development only.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Secrets live in `ingest/.env` (gitignored): `GEMINI_API_KEY`, `PINECONE_API_KEY`.
In Cloud Run these are set as environment variables on the service/job, not committed.

## Constraints

- **GCP only** for storage and compute.
- **Zero cost**: everything stays inside free tiers. Gemini via Google AI Studio
  (not Vertex AI), GCS in `us-central1`, Pinecone serverless free tier.
- **Production-level**: dedicated least-privilege service accounts, idempotent
  scrape, automated monthly refresh crons, structured data.

## Pinecone

- Index `know-my-professor`, serverless on **aws/us-east-1**, dim **3072**
  (`gemini-embedding-001`), metric cosine.
- Vector ID convention: `{slug}#{section_type}`.

See `CLAUDE.md` for detailed phase status and operational commands.
