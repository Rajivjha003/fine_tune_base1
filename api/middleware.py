"""
FastAPI middleware for request logging and rate limiting.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with:
    - Trace ID (for correlating with Langfuse traces)
    - Method + path
    - Status code
    - Latency in milliseconds
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = request.headers.get("x-trace-id", uuid.uuid4().hex[:12])
        start_time = time.time()

        # Inject trace ID into request state for downstream use
        request.state.trace_id = trace_id

        try:
            response = await call_next(request)
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "request_error | trace=%s method=%s path=%s latency=%.1fms error=%s",
                trace_id,
                request.method,
                request.url.path,
                latency_ms,
                str(e),
            )
            raise

        latency_ms = (time.time() - start_time) * 1000

        # Add trace ID to response headers
        response.headers["x-trace-id"] = trace_id
        response.headers["x-latency-ms"] = f"{latency_ms:.1f}"

        log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            log_level,
            "request | trace=%s method=%s path=%s status=%d latency=%.1fms",
            trace_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )

        return response
