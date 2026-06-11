# RAG System

Production RAG monorepo with an evaluation harness.

## Structure

```
services/
  api/          Streamlit web UI (document upload + chat)
  ingestion/    Document ingestion pipeline (FastAPI)
  reranker/     Cross-encoder reranker sidecar (FastAPI)
eval/           DeepEval + RAGAS evaluation harness
infra/          Docker Compose (Dokploy) and environment config
scripts/        Development utilities
tests/          Unit and integration tests
```

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dev dependencies
uv sync --group dev

# Install service-specific dependency groups as needed
uv sync --group api
uv sync --group ingestion
uv sync --group reranker
uv sync --group eval
```

## Development

```bash
# Run linting
uv run ruff check .
uv run ruff format --check .

# Run type checking
uv run mypy .

# Run tests
uv run pytest
```

### Run the Streamlit UI locally

```bash
uv sync --group api
uv run streamlit run services/api/main.py --server.port=8000
```

Open [http://localhost:8000](http://localhost:8000). The sidebar lets you set a **Tenant ID**. Upload a document in the top section, then ask questions in the chat below.

> The ingestion service must be running separately for uploads to work:
> ```bash
> uv sync --group ingestion
> uv run python -m services.ingestion.main
> ```

## Services

Each service lives under `services/<name>/` with a `main.py` entrypoint.

| Service     | Port | Role                                              |
| ----------- | ---- | ------------------------------------------------- |
| `api`       | 8000 | Streamlit web UI — document upload and chat       |
| `ingestion` | 8002 | Document parsing, chunking, and vector indexing   |
| `reranker`  | 8001 | Cross-encoder reranking sidecar                   |
| `qdrant`    | 6333 | Vector store                                      |
| `redis`     | 6379 | Semantic query cache                              |

The `api` service no longer exposes a JSON REST API. The RAG pipeline runs in-process inside the Streamlit app and communicates with `ingestion` and `reranker` over HTTP on the internal Docker network.

## Evaluation

Golden datasets live in `eval/datasets/` (JSONL). Metrics and CI runners live under `eval/metrics/` and `eval/runners/`.

Run eval directly against the in-process pipeline (no HTTP API required):

```bash
uv run rag-eval run \
  --dataset eval/datasets/golden.jsonl \
  --direct \
  --output-dir eval/results
```

Alternatively, point it at a running HTTP ingestion+query stack with `--api-url`:

```bash
uv run rag-eval run \
  --dataset eval/datasets/golden.jsonl \
  --api-url http://localhost:8000 \
  --output-dir eval/results
```

Compare two reports:

```bash
uv run rag-eval compare \
  --baseline eval/datasets/baseline_report.json \
  --current  eval/results/eval_<timestamp>.json
```

## Deployment

### Local (Docker Compose)

```bash
cp infra/.env.example infra/.env
# Set OPENAI_API_KEY in infra/.env

cd infra
docker compose up -d --build
```

Verify:

```bash
curl http://localhost:8000/_stcore/health
open http://localhost:8000/
```

### Dokploy

**Do not use Application/Nixpacks mode.** Dokploy will guess `python -m rag-system`, which is not a valid entrypoint. Use **Compose** only.

See [`infra/DOKPLOY.md`](infra/DOKPLOY.md) for the full guide.

1. Create a **Compose** application in Dokploy and connect this repository.
2. Set the compose file path to `infra/docker-compose.yml`.
3. Set environment variables in Dokploy (or copy `infra/.env.example` to `infra/.env` for local compose). Required:
   - `OPENAI_API_KEY`
   - `QDRANT_URL=http://qdrant:6333`
   - `REDIS_URL=redis://redis:6379`
4. Attach your domain **only to the `api` service** on port `8000`.
5. Keep `ingestion`, `reranker`, `qdrant`, and `redis` on the internal Docker network.

After deploy, the public site is `https://your-domain/` (Streamlit UI).

**Server sizing:** 2+ CPU cores and 8 GB RAM recommended (reranker model + Qdrant + PDF ingestion).

**Persistent volumes:** `qdrant_data` (vectors) and `redis_data` (cache) are defined in compose.

**Outbound access required:** OpenAI API and Hugging Face Hub (reranker model download on first start).

A root [`Dockerfile`](Dockerfile) is provided as a safety net for API-only builds but does not replace the Compose stack for production.

### Environment

See `infra/.env.example` for all variables. Internal service URLs use Docker DNS names:

```env
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379
INGESTION_URL=http://ingestion:8002
RERANKER_URL=http://reranker:8001
```
