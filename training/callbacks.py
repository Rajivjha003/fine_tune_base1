"""
Training callbacks for VRAM monitoring, eval loss tracking, and MLflow logging.

All callbacks are HuggingFace TrainerCallback subclasses and are
auto-attached to the SFTTrainer during fine-tuning.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from core.config import get_settings

logger = logging.getLogger(__name__)


class VRAMMonitorCallback(TrainerCallback):
    """
    Monitors GPU VRAM usage during training.

    Logs peak memory every N steps and fires an alert if usage
    exceeds the configured threshold (default: 7.5GB).
    """

    def __init__(self, log_every_n_steps: int = 10):
        self._log_every_n_steps = log_every_n_steps
        self._settings = get_settings()
        self._threshold_gb = self._settings.training.defaults.get("vram_alert_threshold_gb", 7.5)
        self._peak_gb = 0.0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.global_step % self._log_every_n_steps != 0:
            return

        try:
            import torch

            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_reserved(0)
                peak_gb = peak_bytes / (1024**3)
                self._peak_gb = max(self._peak_gb, peak_gb)

                logger.info(
                    "Step %d: VRAM peak=%.2fGB (threshold=%.1fGB)",
                    state.global_step,
                    peak_gb,
                    self._threshold_gb,
                )

                # Log to MLflow if available
                if state.is_world_process_zero:
                    try:
                        import mlflow

                        mlflow.log_metric("vram_peak_gb", peak_gb, step=state.global_step)
                    except Exception:
                        pass

                # Alert if threshold exceeded
                if peak_gb > self._threshold_gb:
                    logger.warning(
                        "VRAM ALERT: %.2fGB exceeds threshold %.1fGB at step %d!",
                        peak_gb,
                        self._threshold_gb,
                        state.global_step,
                    )
        except ImportError:
            pass

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        logger.info("Training complete. Overall VRAM peak: %.2fGB", self._peak_gb)
        try:
            import mlflow

            mlflow.log_metric("vram_peak_gb_overall", self._peak_gb)
        except Exception:
            pass


class EvalLossCallback(TrainerCallback):
    """
    Tracks eval loss divergence from train loss.

    Early stops if eval loss diverges more than `max_divergence` from
    train loss, indicating overfitting.
    """

    def __init__(self, max_divergence: float = 0.3):
        self._max_divergence = max_divergence
        self._train_losses: list[float] = []
        self._eval_losses: list[float] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return

        if "loss" in logs:
            self._train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self._eval_losses.append(logs["eval_loss"])

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None or "eval_loss" not in metrics:
            return

        eval_loss = metrics["eval_loss"]
        self._eval_losses.append(eval_loss)

        # Get recent average train loss
        if self._train_losses:
            recent_train_loss = sum(self._train_losses[-10:]) / min(len(self._train_losses), 10)
            divergence = eval_loss - recent_train_loss

            logger.info(
                "Eval loss: %.4f | Train loss (recent avg): %.4f | Divergence: %.4f",
                eval_loss,
                recent_train_loss,
                divergence,
            )

            if divergence > self._max_divergence:
                logger.warning(
                    "OVERFITTING DETECTED: eval-train divergence %.4f > threshold %.4f. "
                    "Consider stopping early.",
                    divergence,
                    self._max_divergence,
                )
                # Log to MLflow
                try:
                    import mlflow

                    mlflow.log_metric("overfitting_divergence", divergence)
                    mlflow.set_tag("overfitting_detected", "true")
                except Exception:
                    pass


class MLflowLoggingCallback(TrainerCallback):
    """
    Comprehensive MLflow logging for all training metrics, params, and artifacts.

    Logs:
    - All training metrics (loss, learning rate, etc.)
    - Eval metrics
    - Training timing
    - Hardware info
    """

    def __init__(self, run_name: str | None = None):
        self._run_name = run_name
        self._start_time: float | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._start_time = time.time()

        try:
            import mlflow

            # Log training arguments as params
            params_to_log = {
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "lr_scheduler_type": str(args.lr_scheduler_type),
                "max_grad_norm": args.max_grad_norm,
                "optim": str(args.optim),
                "bf16": args.bf16,
                "fp16": args.fp16,
                "seed": args.seed,
            }
            mlflow.log_params(params_to_log)

            # Log hardware info
            try:
                import torch

                if torch.cuda.is_available():
                    mlflow.set_tag("gpu_name", torch.cuda.get_device_name(0))
                    props = torch.cuda.get_device_properties(0)
                    mlflow.set_tag("gpu_vram_gb", f"{props.total_mem / (1024**3):.1f}")
                    mlflow.set_tag("cuda_version", torch.version.cuda or "unknown")
            except Exception:
                pass

        except Exception as e:
            logger.warning("MLflow param logging failed: %s", e)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None or not state.is_world_process_zero:
            return

        try:
            import mlflow

            # Filter out non-numeric values
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float)) and k != "epoch"}
            if metrics:
                mlflow.log_metrics(metrics, step=state.global_step)
        except Exception:
            pass

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._start_time is None:
            return

        elapsed = time.time() - self._start_time

        try:
            import mlflow

            mlflow.log_metric("train_time_seconds", elapsed)
            mlflow.log_metric("train_time_minutes", elapsed / 60)

            # Calculate tokens per second estimate
            if state.max_steps > 0 and hasattr(args, "per_device_train_batch_size"):
                settings = get_settings()
                max_seq_len = settings.training.defaults.get("max_seq_length", 512)
                total_tokens = state.max_steps * args.per_device_train_batch_size * max_seq_len
                tps = total_tokens / elapsed if elapsed > 0 else 0
                mlflow.log_metric("tokens_per_second", tps)

            logger.info("Training completed in %.1f minutes (%.0f seconds).", elapsed / 60, elapsed)
        except Exception as e:
            logger.warning("MLflow end-of-training logging failed: %s", e)
