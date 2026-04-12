"""
Admin API routes.

POST /admin/swap-model — hot-swap the active model
POST /admin/eval/run — trigger evaluation suite
GET  /admin/health — system health check
GET  /admin/metrics — system metrics
GET  /admin/registry/models — list registered models
GET  /admin/cache/stats — cache statistics
POST /admin/cache/clear — clear the inference cache
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Request / Response Models ─────────────────────────────────────────────


class SwapModelRequest(BaseModel):
    model_key: str = Field(..., description="Target model key from config (e.g., 'qwen2.5-3b')")
    skip_eval: bool = Field(default=False, description="Skip evaluation suite (not recommended)")
    force: bool = Field(default=False, description="Force swap even if already active")


class SwapModelResponse(BaseModel):
    success: bool
    model_key: str
    previous_model: str | None
    steps_completed: list[str]
    eval_passed: bool | None = None
    rolled_back: bool = False
    duration_seconds: float = 0.0
    error: str | None = None


class HealthResponse(BaseModel):
    gpu_available: bool
    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    ollama_reachable: bool
    ollama_models: list[str]
    mlflow_reachable: bool
    redis_reachable: bool
    configs_valid: bool
    is_healthy: bool


# ── Routes ────────────────────────────────────────────────────────────────


@router.post("/swap-model", response_model=SwapModelResponse)
async def swap_model(body: SwapModelRequest) -> SwapModelResponse:
    """
    Hot-swap the active inference model.

    Executes the full swap protocol:
    1. Verify GGUF exists
    2. Register with Ollama
    3. Update LiteLLM routing
    4. Update MLflow registry
    5. Run evaluation suite (unless skipped)
    6. Auto-rollback on eval failure
    """
    from core.model_switcher import ModelSwitcher

    switcher = ModelSwitcher()

    try:
        result = await switcher.switch(
            body.model_key,
            skip_eval=body.skip_eval,
            force=body.force,
        )

        return SwapModelResponse(
            success=result.success,
            model_key=result.model_key,
            previous_model=result.previous_model,
            steps_completed=result.steps_completed,
            eval_passed=result.eval_passed,
            rolled_back=result.rolled_back,
            duration_seconds=result.duration_seconds,
            error=result.error,
        )

    except Exception as e:
        logger.error("Model swap API error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Run a full system health check."""
    from core.system_init import SystemInitializer

    init = SystemInitializer()
    report = await init.run_health_check()

    return HealthResponse(
        gpu_available=report.gpu_available,
        gpu_name=report.gpu_name,
        vram_total_gb=report.vram_total_gb,
        vram_free_gb=report.vram_free_gb,
        ollama_reachable=report.ollama_reachable,
        ollama_models=report.ollama_models,
        mlflow_reachable=report.mlflow_reachable,
        redis_reachable=report.redis_reachable,
        configs_valid=report.configs_valid,
        is_healthy=report.is_healthy,
    )


@router.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """Return system metrics summary."""
    from core.config import get_settings

    settings = get_settings()
    metrics: dict[str, Any] = {
        "config": {
            "primary_model": None,
            "fallback_models": [],
            "inference_backend": settings.models.defaults.get("inference_backend", "ollama"),
        },
    }

    try:
        pk, ps = settings.models.get_primary_model()
        metrics["config"]["primary_model"] = pk
    except Exception:
        pass

    try:
        metrics["config"]["fallback_models"] = [k for k, _ in settings.models.get_fallback_models()]
    except Exception:
        pass

    # GPU metrics
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            metrics["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "vram_total_gb": round(total / (1024**3), 2),
                "vram_free_gb": round(free / (1024**3), 2),
                "vram_used_gb": round((total - free) / (1024**3), 2),
            }
    except Exception:
        pass

    # Cache metrics
    try:
        from inference.cache import SemanticCache

        cache = SemanticCache()
        metrics["cache"] = cache.stats()
    except Exception:
        pass

    # Fallback chain status
    try:
        from inference.fallback import FallbackChain

        chain = FallbackChain()
        metrics["fallback_chain"] = chain.get_status()
    except Exception:
        pass

    return metrics


@router.get("/registry/models")
async def list_models() -> dict[str, Any]:
    """List all models from config and their registry status."""
    from core.config import get_settings

    settings = get_settings()
    models = {}

    for key, spec in settings.models.models.items():
        models[key] = {
            "hf_id": spec.hf_id,
            "family": spec.family,
            "tier": spec.tier,
            "vram_required_gb": spec.vram_required_gb,
            "ollama_name": spec.ollama_name,
            "description": spec.description,
            "champion": False,
        }

        # Check MLflow registry for champion status
        try:
            from registry.model_manager import ModelManager

            manager = ModelManager()
            champion = manager.get_champion(key)
            if champion:
                models[key]["champion"] = True
                models[key]["champion_version"] = champion.version
        except Exception:
            pass

    return {"models": models}


@router.get("/cache/stats")
async def cache_stats() -> dict[str, Any]:
    """Get inference cache statistics."""
    from inference.cache import SemanticCache

    cache = SemanticCache()
    return cache.stats()


@router.post("/cache/clear")
async def clear_cache() -> dict[str, str]:
    """Clear the inference cache."""
    from inference.cache import SemanticCache

    cache = SemanticCache()
    cache.clear()
    return {"status": "cache_cleared"}


@router.get("/upgrade/plan")
async def upgrade_plan() -> dict[str, Any]:
    """Get hardware upgrade recommendations."""
    from core.upgrade_planner import UpgradePlanner

    planner = UpgradePlanner()
    rec = planner.recommend()

    return {
        "current_tier": rec.current_tier,
        "current_tier_label": rec.current_tier_label,
        "eligible_models": rec.eligible_models,
        "recommended_primary": rec.recommended_primary,
        "recommended_fallback": rec.recommended_fallback,
        "next_tier": rec.next_tier,
        "next_tier_label": rec.next_tier_label,
        "next_tier_unlocks": rec.next_tier_unlocks,
        "notes": rec.notes,
    }
