"""
System initialization and health checks.

Validates all configs, checks GPU/VRAM, verifies local services,
loads the champion model, and emits SYSTEM_READY event.

Usage:
    python -m core.system_init              # Full bootstrap
    python -m core.system_init --check-only # Health check without loading model
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from core.config import get_settings
from core.events import SYSTEM_ERROR, SYSTEM_READY, event_bus
from core.exceptions import SystemInitError

logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    """Result of a full system health check."""

    gpu_available: bool = False
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    cuda_version: str = ""
    ollama_reachable: bool = False
    ollama_models: list[str] = field(default_factory=list)
    mlflow_reachable: bool = False
    redis_reachable: bool = False
    configs_valid: bool = False
    config_errors: list[str] = field(default_factory=list)
    champion_loaded: bool = False

    @property
    def is_healthy(self) -> bool:
        return self.configs_valid and self.gpu_available

    def summary(self) -> str:
        ok = "[OK]"
        fail = "[FAIL]"
        warn = "[WARN]"
        lines = [
            "=== MerchFine System Health ===",
            f"  GPU:        {ok if self.gpu_available else fail} {self.gpu_name} ({self.vram_total_gb:.1f}GB total, {self.vram_free_gb:.1f}GB free)",
            f"  CUDA:       {self.cuda_version or 'N/A'}",
            f"  Configs:    {ok if self.configs_valid else fail} {', '.join(self.config_errors) if self.config_errors else 'All valid'}",
            f"  Ollama:     {ok if self.ollama_reachable else warn + ' Not running'} {f'({len(self.ollama_models)} models)' if self.ollama_reachable else ''}",
            f"  MLflow:     {ok if self.mlflow_reachable else warn + ' Not running'}",
            f"  Redis:      {ok if self.redis_reachable else warn + ' Not running (using in-memory cache)'}",
            f"  Champion:   {ok + ' Loaded' if self.champion_loaded else warn + ' Not loaded'}",
            f"  Status:     {'HEALTHY' if self.is_healthy else 'ISSUES DETECTED'}",
            "===============================",
        ]
        return "\n".join(lines)


class SystemInitializer:
    """Boots up the MerchFine system."""

    def __init__(self):
        self.settings = get_settings()
        self.report = HealthReport()

    def check_gpu(self) -> None:
        """Detect GPU, VRAM, and CUDA version."""
        try:
            import torch

            if torch.cuda.is_available():
                self.report.gpu_available = True
                self.report.gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                self.report.vram_total_gb = props.total_memory / (1024**3)
                free_mem, _ = torch.cuda.mem_get_info(0)
                self.report.vram_free_gb = free_mem / (1024**3)
                self.report.cuda_version = torch.version.cuda or ""
            else:
                logger.warning("No CUDA GPU detected. Training will not be available.")
        except ImportError:
            logger.warning("PyTorch not installed — GPU detection skipped.")
        except Exception as e:
            logger.error("GPU detection failed: %s", e)

    def check_configs(self) -> None:
        """Validate all configuration files load correctly."""
        try:
            settings = get_settings()
            # Verify critical configs exist
            if not settings.models.models:
                self.report.config_errors.append("No models defined in models.yaml")
            if not settings.training.profiles:
                self.report.config_errors.append("No training profiles in training.yaml")
            self.report.configs_valid = len(self.report.config_errors) == 0
        except Exception as e:
            self.report.config_errors.append(str(e))
            self.report.configs_valid = False

    async def check_ollama(self) -> None:
        """Check if Ollama is running and list loaded models."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.settings.ollama_host}/api/tags")
                if resp.status_code == 200:
                    self.report.ollama_reachable = True
                    data = resp.json()
                    self.report.ollama_models = [m["name"] for m in data.get("models", [])]
        except Exception:
            self.report.ollama_reachable = False

    async def check_mlflow(self) -> None:
        """Check if MLflow tracking server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.settings.mlflow_tracking_uri}/health")
                self.report.mlflow_reachable = resp.status_code == 200
        except Exception:
            # MLflow might be in file-based mode (no server needed)
            mlruns_path = self.settings.project_root / "mlruns"
            if mlruns_path.exists():
                self.report.mlflow_reachable = True
                self.settings.mlflow_tracking_uri = str(mlruns_path.resolve().as_uri())
                logger.info("MLflow in file-based mode (no server). Using local mlruns/.")
            else:
                self.report.mlflow_reachable = False

    async def check_redis(self) -> None:
        """Check if Redis is reachable."""
        try:
            import redis as redis_lib

            r = redis_lib.from_url(self.settings.redis_url, socket_timeout=3)
            r.ping()
            self.report.redis_reachable = True
            r.close()
        except Exception:
            self.report.redis_reachable = False
            logger.info("Redis not available — semantic cache will use in-memory fallback.")

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        dirs = [
            self.settings.data_dir / "raw",
            self.settings.data_dir / "processed",
            self.settings.data_dir / "synthetic",
            self.settings.data_dir / "knowledge_base",
            self.settings.data_dir / "vector_store",
            self.settings.outputs_dir,
            self.settings.project_root / "mlruns",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    async def run_health_check(self) -> HealthReport:
        """Run all health checks and return the report."""
        self.check_gpu()
        self.check_configs()
        self.ensure_directories()

        # Run async checks concurrently
        await asyncio.gather(
            self.check_ollama(),
            self.check_mlflow(),
            self.check_redis(),
        )

        return self.report

    async def full_bootstrap(self) -> HealthReport:
        """
        Full system bootstrap:
        1. Run all health checks
        2. Initialize MLflow experiment
        3. Emit SYSTEM_READY or SYSTEM_ERROR event
        """
        report = await self.run_health_check()

        # Initialize MLflow experiment
        if report.mlflow_reachable:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
                mlflow.set_experiment(self.settings.mlflow_experiment_name)
                logger.info("MLflow experiment '%s' initialized.", self.settings.mlflow_experiment_name)
            except Exception as e:
                logger.warning("MLflow experiment init failed: %s", e)

        # Emit event
        if report.is_healthy:
            await event_bus.emit(
                SYSTEM_READY,
                data={"gpu": report.gpu_name, "vram_gb": report.vram_total_gb},
                source="system_init",
            )
            logger.info("System bootstrap complete.")
        else:
            await event_bus.emit(
                SYSTEM_ERROR,
                data={"errors": report.config_errors},
                source="system_init",
            )
            logger.error("System bootstrap completed with errors.")

        return report


async def _async_main(check_only: bool = False) -> None:
    """Async entry point."""
    init = SystemInitializer()

    if check_only:
        report = await init.run_health_check()
    else:
        report = await init.full_bootstrap()

    print(report.summary())

    if not report.is_healthy:
        sys.exit(1)


def cli_entrypoint() -> None:
    """CLI entry point for `python -m core.system_init`."""
    check_only = "--check-only" in sys.argv
    asyncio.run(_async_main(check_only=check_only))


if __name__ == "__main__":
    cli_entrypoint()
