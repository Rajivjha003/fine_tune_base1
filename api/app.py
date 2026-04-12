"""
FastAPI application with lifespan management, middleware, and route mounting.

The entire system bootstraps on startup and cleans up on shutdown.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core import __app_name__, __version__
from core.exceptions import (
    FallbackExhaustedError,
    GuardrailBlockError,
    InferenceError,
    MerchFineError,
    QualityGateFailedError,
)

logger = logging.getLogger(__name__)

# ── Globals set during lifespan ───────────────────────────────────────────
_system_report = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: bootstrap on startup, cleanup on shutdown."""
    global _system_report

    logger.info("Starting %s v%s...", __app_name__, __version__)

    # System bootstrap
    from core.system_init import SystemInitializer

    init = SystemInitializer()
    _system_report = await init.full_bootstrap()
    logger.info("\n%s", _system_report.summary())

    yield

    # Cleanup
    logger.info("Shutting down %s...", __app_name__)


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    app = FastAPI(
        title="MerchFine API",
        description="MerchMix LLMOps — Retail demand forecasting and inventory planning AI",
        version=__version__,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging middleware ────────────────────────────────────
    from api.middleware import RequestLoggingMiddleware

    app.add_middleware(RequestLoggingMiddleware)

    # ── Exception handlers ───────────────────────────────────────────

    @app.exception_handler(GuardrailBlockError)
    async def guardrail_block_handler(request: Request, exc: GuardrailBlockError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "guardrail_block",
                "message": str(exc),
                "details": exc.details,
            },
        )

    @app.exception_handler(FallbackExhaustedError)
    async def fallback_handler(request: Request, exc: FallbackExhaustedError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": "service_unavailable",
                "message": "All inference backends are unavailable. Please try again later.",
                "details": exc.details,
            },
        )

    @app.exception_handler(InferenceError)
    async def inference_handler(request: Request, exc: InferenceError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "inference_error",
                "message": str(exc),
            },
        )

    @app.exception_handler(MerchFineError)
    async def merchfine_handler(request: Request, exc: MerchFineError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "message": str(exc),
                "details": exc.details,
            },
        )

    # ── Mount routes ─────────────────────────────────────────────────
    from api.routes.forecast import router as forecast_router
    from api.routes.admin import router as admin_router

    app.include_router(forecast_router, prefix="/api", tags=["Forecasting"])
    app.include_router(admin_router, prefix="/admin", tags=["Admin"])

    # ── Root health check ────────────────────────────────────────────

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "service": __app_name__,
            "version": __version__,
            "status": "healthy" if _system_report and _system_report.is_healthy else "degraded",
        }

    return app


# ── Module-level app instance for uvicorn ─────────────────────────────────
app = create_app()
