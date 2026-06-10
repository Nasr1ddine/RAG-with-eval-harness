# RAG System

Production RAG monorepo with an evaluation harness.

## Structure

```
services/
  api/          FastAPI query entrypoint
  ingestion/    Document ingestion pipeline
  reranker/     Cross-encoder reranker sidecar
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

## Services

Each service lives under `services/<name>/` with a `main.py` entrypoint. Application logic is not yet implemented.

## Evaluation

Golden datasets live in `eval/datasets/` (JSONL). Metrics and CI runners live under `eval/metrics/` and `eval/runners/`.

## Deployment

See `infra/docker-compose.yml` and `infra/.env.example` for Dokploy deployment configuration.
