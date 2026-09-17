# Know My Professor

Ask a question about Northeastern's Khoury College faculty and get an answer
with citations, drawn from ~870 real faculty profiles.

**[Try it →](https://kmp-frontend-309233821309.us-central1.run.app)**

---

> **Who at Khoury works on cryptography?**
>
> Several Khoury faculty work on cryptography:
>
> - **Daniel Wichs** — all aspects of modern cryptography, including homomorphic
>   encryption and outsourcing computation to the cloud **[2]**
> - **Zhengzhong Jin** — proof systems for delegating computation, with
>   blockchain applications **[3]**
> - **Cody Freitag** — cryptographic proof systems, blockchains, and quantum
>   threats to cryptographic protocols **[1]**
> - **Joshua Gancher** — cryptographic software and formal verification **[4]**
>
> *Sources: [1] Cody Freitag — biography · [2] Daniel Wichs — biography · …*

Every claim is numbered, and every number resolves to the profile section it
came from. If the corpus can't answer, it says so instead of guessing.

---

## How it works

```
Khoury directory ──scrape──►  profile JSON  ──┐
                                              ├──► chunk ──► embed ──► Pinecone
faculty websites ──crawl──►  extracted JSON ──┘                         (1024-d)

                                    ┌──────────────────────────────┐
your question ──► embed ──► search ─┤ top 8 matching profile chunks├──► Claude
                                    └──────────────────────────────┘      │
                                                     cited answer ◄────────┘
```

One chunk per section — six from the directory profile (biography, research
interests, education, areas of interest, labs, projects) and five more extracted
from the professor's own website (summary, current projects, recent
publications, lab members, news). A citation therefore points at a specific
claim, not a whole page.

## Stack

| | |
|---|---|
| **Embeddings** | Mistral `mistral-embed-2312` (1024-d) |
| **Vector search** | Pinecone serverless, cosine |
| **Generation** | Claude Opus 5 |
| **Extraction** | Gemini 3.1 Flash Lite, for faculty websites |
| **Serving** | FastAPI + Streamlit on Cloud Run |
| **Refresh** | three monthly jobs: scrape → enrich → ingest |

Everything runs inside free tiers.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run serving/frontend/app.py    # talks to the live API — no keys needed
pytest tests/ -q                         # offline test suite — no keys needed
```

To run the backend yourself, put `MISTRAL_API_KEY`, `PINECONE_API_KEY` and
`ANTHROPIC_API_KEY` in a `.env` at the repo root.

## A few decisions worth explaining

**Queries and documents can't drift apart.** `embed_query()` calls the same
`embed_texts()` that ingest uses — one code path, so the two vector spaces are
the same by construction rather than by convention.

**Adding a provider or a data source is one line.** Embedders, chat models, and
data sources are registries; ingest names no source and the API names no
provider, so neither changes when you add one.

**Mistakes fail loudly and early.** A chunk's text is hashed, so ingest
re-embeds only what changed; an embedder whose dimension doesn't match the index
refuses to start rather than failing halfway through an upsert.

**Answers are measured, not trusted.** A golden question set scores retrieval
and citations on every change, so "does this feel better?" becomes a number.

## Layout

```
core/           the RAG pipeline: retrieve → score → generate
preprocessing/  scrapers, website extraction, chunking, ingest
shared/         config, settings, embeddings, the API contract
serving/        FastAPI backend · Streamlit frontend
evaluation/     golden questions + scoring
tests/          offline test suite
deploy/         one Dockerfile → three jobs + two services
```
