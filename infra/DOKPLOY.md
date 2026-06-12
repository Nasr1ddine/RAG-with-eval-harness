# Dokploy deployment guide

This project is a **multi-service Docker Compose stack**. Do not deploy it as a Dokploy **Application** (Nixpacks/Railpack) — that mode auto-starts `python -m rag-system`, which does not exist and will not expose an HTTP server.

## Migrate from Application mode

1. Stop or delete the existing **Application** deployment in Dokploy.
2. Create a new **Compose** application and connect this repository.
3. Set the compose file path to `infra/docker-compose.yml`.
4. Add environment variables (see `infra/.env.example`). Minimum:
   - `OPENAI_API_KEY`
   - `QDRANT_URL=http://qdrant:6333`
   - `REDIS_URL=redis://redis:6379`
   - `INGESTION_URL=http://ingestion:8002`
   - `RERANKER_URL=http://reranker:8001`
5. Attach your domain **only to the `api` service** on port **8000**.
6. Leave `ingestion`, `reranker`, `qdrant`, and `redis` on the internal Docker network (no public domains).

> **Do not set `PORT` as a global/project environment variable.** Every service reads
> `PORT` and uses it as its own listen port. A global `PORT=8000` forces `ingestion`
> (and `reranker`) to bind to 8000 instead of their real ports, breaking the API's
> internal calls. `PORT` is already scoped to the `api` service in
> `infra/docker-compose.yml`; leave it there and out of the shared environment.

## Verify

- Web UI: `https://your-domain/`
- Health: `https://your-domain/_stcore/health`

## Server sizing

- 2+ CPU cores, 8 GB RAM recommended (reranker model + Qdrant + PDF ingestion).
- Persistent volumes: `qdrant_data`, `redis_data` (defined in compose).
- Outbound access: OpenAI API and Hugging Face Hub (reranker model download on first start).

## Root Dockerfile (optional)

A root [`Dockerfile`](../Dockerfile) mirrors the API image for API-only experiments. It does **not** run Qdrant, Redis, ingestion, or the reranker — use Compose for production.
