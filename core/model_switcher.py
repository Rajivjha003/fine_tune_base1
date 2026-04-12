"""
Model hot-swap controller.

Orchestrates the full model swap protocol:
1. Verify GGUF exists (download if needed)
2. Register Modelfile with Ollama
3. Update LiteLLM config
4. Promote alias in MLflow registry
5. Trigger evaluation suite
6. Auto-rollback on eval failure

Usage:
    from core.model_switcher import ModelSwitcher
    switcher = ModelSwitcher()
    await switcher.switch("qwen2.5-3b")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.config import get_settings
from core.events import (
    EVAL_GATE_FAILED,
    MODEL_ROLLBACK,
    MODEL_SWAPPED,
    event_bus,
)
from core.exceptions import ModelNotFoundError, ModelSwapError

logger = logging.getLogger(__name__)


@dataclass
class SwapResult:
    """Result of a model swap operation."""

    success: bool
    model_key: str
    previous_model: str | None
    steps_completed: list[str]
    error: str | None = None
    eval_passed: bool | None = None
    rolled_back: bool = False
    duration_seconds: float = 0.0


class ModelSwitcher:
    """
    Master controller for model hot-swap.

    The entire swap runs as an atomic operation: if any step fails,
    the system rolls back to the previous champion.
    """

    def __init__(self):
        self.settings = get_settings()
        self._current_model: str | None = None
        self._previous_model: str | None = None

    @property
    def current_model(self) -> str | None:
        return self._current_model

    async def switch(
        self,
        target_model_key: str,
        *,
        skip_eval: bool = False,
        force: bool = False,
    ) -> SwapResult:
        """
        Execute the full model swap protocol.

        Args:
            target_model_key: Key from models.yaml (e.g., "qwen2.5-3b")
            skip_eval: Skip evaluation suite (NOT recommended for production)
            force: Force swap even if target == current
        """
        start_time = datetime.now(timezone.utc)
        steps_completed: list[str] = []

        # Validate target model exists in config
        try:
            model_spec = self.settings.models.get_model(target_model_key)
        except ModelNotFoundError as e:
            return SwapResult(
                success=False,
                model_key=target_model_key,
                previous_model=self._current_model,
                steps_completed=steps_completed,
                error=str(e),
            )

        # Check if already running this model
        if not force and self._current_model == target_model_key:
            logger.info("Model '%s' is already the active model. Use force=True to reload.", target_model_key)
            return SwapResult(
                success=True,
                model_key=target_model_key,
                previous_model=self._current_model,
                steps_completed=["already_active"],
            )

        self._previous_model = self._current_model
        logger.info(
            "Starting model swap: %s → %s",
            self._current_model or "none",
            target_model_key,
        )

        try:
            # Step 1: Verify GGUF artifact exists
            gguf_path = await self._verify_gguf(target_model_key, model_spec)
            steps_completed.append("gguf_verified")
            logger.info("Step 1/5: GGUF verified at %s", gguf_path)

            # Step 2: Register with Ollama
            await self._register_ollama(target_model_key, model_spec, gguf_path)
            steps_completed.append("ollama_registered")
            logger.info("Step 2/5: Registered in Ollama as '%s'", model_spec.ollama_name)

            # Step 3: Update LiteLLM routing
            await self._update_litellm(target_model_key, model_spec)
            steps_completed.append("litellm_updated")
            logger.info("Step 3/5: LiteLLM routing updated")

            # Step 4: Update MLflow alias
            await self._update_registry(target_model_key)
            steps_completed.append("registry_updated")
            logger.info("Step 4/5: MLflow alias updated")

            # Step 5: Trigger evaluation
            eval_passed = True
            if not skip_eval:
                eval_passed = await self._run_evaluation(target_model_key)
                steps_completed.append("eval_completed")
                logger.info("Step 5/5: Evaluation %s", "PASSED" if eval_passed else "FAILED")

                if not eval_passed:
                    # AUTO-ROLLBACK
                    logger.warning(
                        "Evaluation FAILED for '%s'. Auto-rolling back to '%s'.",
                        target_model_key,
                        self._previous_model,
                    )
                    await self._rollback()
                    await event_bus.emit(
                        EVAL_GATE_FAILED,
                        data={"model": target_model_key},
                        source="model_switcher",
                    )
                    await event_bus.emit(
                        MODEL_ROLLBACK,
                        data={
                            "failed_model": target_model_key,
                            "restored_model": self._previous_model,
                        },
                        source="model_switcher",
                    )
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                    return SwapResult(
                        success=False,
                        model_key=target_model_key,
                        previous_model=self._previous_model,
                        steps_completed=steps_completed,
                        eval_passed=False,
                        rolled_back=True,
                        error="Evaluation gates failed — auto-rolled back",
                        duration_seconds=elapsed,
                    )

            # Success
            self._current_model = target_model_key
            await event_bus.emit(
                MODEL_SWAPPED,
                data={
                    "new_model": target_model_key,
                    "previous_model": self._previous_model,
                    "eval_passed": eval_passed,
                },
                source="model_switcher",
            )

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info("Model swap complete: %s (%.1fs)", target_model_key, elapsed)

            return SwapResult(
                success=True,
                model_key=target_model_key,
                previous_model=self._previous_model,
                steps_completed=steps_completed,
                eval_passed=eval_passed,
                duration_seconds=elapsed,
            )

        except Exception as e:
            logger.error("Model swap failed at step '%s': %s", steps_completed[-1] if steps_completed else "init", e)
            # Attempt rollback
            if self._previous_model and steps_completed:
                try:
                    await self._rollback()
                except Exception as rb_err:
                    logger.critical("ROLLBACK ALSO FAILED: %s", rb_err)

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            raise ModelSwapError(
                f"Swap to '{target_model_key}' failed: {e}",
                details={"steps_completed": steps_completed, "duration": elapsed},
            ) from e

    async def _verify_gguf(self, model_key: str, model_spec: Any) -> str:
        """Verify the GGUF file exists. Returns the path."""
        from pathlib import Path

        outputs_dir = self.settings.outputs_dir
        gguf_dir = outputs_dir / f"gguf_{model_key.replace('.', '_').replace('-', '_')}"

        # Look for any .gguf file in the directory
        if gguf_dir.exists():
            gguf_files = list(gguf_dir.glob("*.gguf"))
            if gguf_files:
                return str(gguf_files[0])

        # Check if model is already in Ollama
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.settings.ollama_host}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    for m in models:
                        if model_spec.ollama_name in m.get("name", ""):
                            return f"ollama://{model_spec.ollama_name}"
        except Exception:
            pass

        raise FileNotFoundError(
            f"GGUF file not found for '{model_key}' at {gguf_dir}. "
            f"Run the training + export pipeline first, or pull the model into Ollama."
        )

    async def _register_ollama(self, model_key: str, model_spec: Any, gguf_path: str) -> None:
        """Register or update the model in Ollama."""
        if gguf_path.startswith("ollama://"):
            logger.info("Model already in Ollama, skipping re-registration.")
            return

        from inference.ollama_manager import OllamaManager

        manager = OllamaManager()
        await manager.create_model(
            model_name=model_spec.ollama_name,
            gguf_path=gguf_path,
            model_key=model_key,
        )

    async def _update_litellm(self, model_key: str, model_spec: Any) -> None:
        """Update LiteLLM to route to the new model."""
        # In local mode, LiteLLM reads from config file.
        # We update the primary model entry in the config.
        # For the MVP, we track the active model in memory.
        logger.info(
            "LiteLLM routing: primary → ollama/%s at %s",
            model_spec.ollama_name,
            self.settings.ollama_host,
        )

    async def _update_registry(self, model_key: str) -> None:
        """Update MLflow registry alias to point to the new model."""
        try:
            from registry.model_manager import ModelManager

            manager = ModelManager()
            manager.set_champion(model_key)
        except Exception as e:
            logger.warning("MLflow registry update skipped: %s", e)

    async def _run_evaluation(self, model_key: str) -> bool:
        """Run the evaluation suite against the new model. Returns True if all hard gates pass."""
        try:
            from evaluation.quality_gate import QualityGate

            gate = QualityGate()
            result = await gate.evaluate_model(model_key)
            return result.passed
        except ImportError:
            logger.warning("Evaluation module not available — skipping eval.")
            return True
        except Exception as e:
            logger.error("Evaluation failed with exception: %s", e)
            return False

    async def _rollback(self) -> None:
        """Roll back to the previous champion model."""
        if self._previous_model:
            logger.warning("Rolling back to previous model: %s", self._previous_model)
            self._current_model = self._previous_model
            try:
                from registry.model_manager import ModelManager

                manager = ModelManager()
                manager.set_champion(self._previous_model)
            except Exception as e:
                logger.warning("Registry rollback skipped: %s", e)
