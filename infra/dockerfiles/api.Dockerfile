FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --group api

COPY services ./services

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8003

CMD ["sh", "-c", "streamlit run services/api/main.py --server.port=${PORT:-8000} --server.address=${API_HOST:-0.0.0.0} --browser.gatherUsageStats=false --server.headless=true"]
