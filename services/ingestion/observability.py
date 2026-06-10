from __future__ import annotations

import logging
import sys
import time
import uuid
from collections.abc import Callable
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

LogProcessor = Callable[[Any, str, dict[str, Any]], dict[str, Any]]


def configure_logging(*, service_name: str, log_level: str, environment: str) -> None:
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_observability_fields(service_name),
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any
    if environment.lower() == "development":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
        stream=sys.stdout,
        force=True,
    )
    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def bind_tenant_context(tenant_id: str) -> None:
    structlog.contextvars.bind_contextvars(tenant_id=tenant_id)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, *, service_name: str, default_tenant_id: str) -> None:
        super().__init__(app)
        self.service_name = service_name
        self.default_tenant_id = default_tenant_id
        self.logger = structlog.get_logger(__name__)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        tenant_id = request.headers.get("x-tenant-id") or self.default_tenant_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, tenant_id=tenant_id)
        request.state.request_id = request_id
        request.state.tenant_id = tenant_id

        start_time = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        except Exception:
            self.logger.exception("request failed")
            raise
        finally:
            latency_ms = round((time.perf_counter() - start_time) * 1000)
            metrics = getattr(request.state, "rag_metrics", {})
            tenant_id = str(
                metrics.get("tenant_id") or getattr(request.state, "tenant_id", tenant_id)
            )
            structlog.contextvars.bind_contextvars(tenant_id=tenant_id)

            self.logger.info(
                "request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code if response is not None else 500,
                latency_ms=latency_ms,
                tenant_id=tenant_id,
                cache_hit=metrics.get("cache_hit", False),
                token_count=metrics.get("token_count", 0),
                request_id=request_id,
            )
            structlog.contextvars.clear_contextvars()


def _add_observability_fields(service_name: str) -> LogProcessor:
    def add_observability_fields(
        _logger: Any,
        _method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        event_dict.setdefault("service_name", service_name)
        event_dict.setdefault("tenant_id", "unknown")
        event_dict.setdefault("request_id", "unknown")
        return event_dict

    return add_observability_fields
