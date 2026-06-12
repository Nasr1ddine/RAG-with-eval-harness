from __future__ import annotations

import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator

from services.ingestion.config import settings
from services.ingestion.observability import (
    RequestLoggingMiddleware,
    bind_tenant_context,
    configure_logging,
)
from services.ingestion.pipeline import IngestionPipeline, IngestionResult

configure_logging(service_name="ingestion", log_level=settings.LOG_LEVEL, environment=settings.ENV)

app = FastAPI(title="Document Ingestion Service")
Instrumentator().instrument(app).expose(app)
app.add_middleware(
    RequestLoggingMiddleware,
    service_name="ingestion",
    default_tenant_id=settings.DEFAULT_TENANT_ID,
)


@app.post("/ingest", response_model=IngestionResult)
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
) -> IngestionResult:
    bind_tenant_context(tenant_id)
    request.state.tenant_id = tenant_id
    request.state.rag_metrics = {
        "tenant_id": tenant_id,
        "cache_hit": False,
        "token_count": 0,
    }
    suffix = Path(file.filename or "").suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await file.read())

    try:
        pipeline = IngestionPipeline()
        return await pipeline.run(
            file_path=str(temp_path),
            metadata={
                "source": file.filename or temp_path.name,
                "content_type": file.content_type,
            },
            tenant_id=tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)


if __name__ == "__main__":
    main()
