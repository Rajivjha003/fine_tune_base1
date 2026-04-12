"""
Optuna hyperparameter sweep for LoRA fine-tuning.

Automates exploration of LoRA rank, alpha, learning rate, and batch size.
Each trial runs a full train+eval cycle and logs results to MLflow.

Search spaces are defined in config/training.yaml under the `sweep` key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.exceptions import TrainingError

logger = logging.getLogger(__name__)


class HyperparameterSweep:
    """
    Automated hyperparameter search using Optuna.

    Each trial:
    1. Samples from the configured search space
    2. Overrides the training profile
    3. Runs a full fine-tune
    4. Returns the objective metric (eval_loss by default)
    """

    def __init__(self, model_key: str, dataset_path: str | Path):
        self.settings = get_settings()
        self.model_key = model_key
        self.dataset_path = str(dataset_path)
        self.sweep_config = self.settings.training.sweep

    def run(self, n_trials: int | None = None, timeout: int | None = None) -> dict[str, Any]:
        """
        Execute the hyperparameter sweep.

        Args:
            n_trials: Override number of trials (default from config).
            timeout: Override timeout in seconds (default from config).

        Returns:
            Dict with best_params, best_value, and all trial summaries.
        """
        import optuna

        n_trials = n_trials or self.sweep_config.n_trials
        timeout = timeout or self.sweep_config.timeout_seconds

        study_name = f"merchfine-sweep-{self.model_key}"
        study = optuna.create_study(
            study_name=study_name,
            direction=self.sweep_config.direction,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        )

        logger.info(
            "Starting sweep: %d trials, timeout=%ds, objective=%s",
            n_trials,
            timeout,
            self.sweep_config.objective_metric,
        )

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        best = study.best_trial
        logger.info(
            "Sweep complete. Best trial #%d: %s=%.4f, params=%s",
            best.number,
            self.sweep_config.objective_metric,
            best.value,
            best.params,
        )

        # Log best params to MLflow
        try:
            import mlflow

            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.settings.mlflow_experiment_name)
            with mlflow.start_run(run_name=f"sweep-best-{self.model_key}"):
                mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
                mlflow.log_metric(f"best_{self.sweep_config.objective_metric}", best.value)
                mlflow.set_tag("sweep_study", study_name)
                mlflow.set_tag("sweep_n_trials", str(n_trials))
        except Exception as e:
            logger.warning("MLflow logging for sweep results failed: %s", e)

        return {
            "best_params": best.params,
            "best_value": best.value,
            "n_trials_completed": len(study.trials),
            "trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
        }

    def _objective(self, trial) -> float:
        """
        Optuna objective function.

        Samples hyperparameters from the configured search space,
        runs a fine-tune, and returns the objective metric.
        """
        # Sample from search space
        search_space = self.sweep_config.search_space
        sampled_params: dict[str, Any] = {}

        if "lora_r" in search_space:
            sampled_params["lora_r"] = trial.suggest_categorical("lora_r", search_space["lora_r"])
        if "lora_alpha" in search_space:
            sampled_params["lora_alpha"] = trial.suggest_categorical("lora_alpha", search_space["lora_alpha"])
        if "learning_rate" in search_space:
            sampled_params["learning_rate"] = trial.suggest_categorical("learning_rate", search_space["learning_rate"])
        if "per_device_train_batch_size" in search_space:
            sampled_params["per_device_train_batch_size"] = trial.suggest_categorical(
                "per_device_train_batch_size", search_space["per_device_train_batch_size"]
            )
        if "gradient_accumulation_steps" in search_space:
            sampled_params["gradient_accumulation_steps"] = trial.suggest_categorical(
                "gradient_accumulation_steps", search_space["gradient_accumulation_steps"]
            )

        logger.info("Trial %d: params=%s", trial.number, sampled_params)

        # Override training profile with sampled params
        from training.finetune import QLoRATrainer

        trainer = QLoRATrainer(model_key=self.model_key)

        # Override profile fields
        for key, value in sampled_params.items():
            if hasattr(trainer.profile, key):
                object.__setattr__(trainer.profile, key, value)

        # Run training with trial-specific output dir
        output_dir = str(
            self.settings.outputs_dir / f"sweep_{self.model_key}" / f"trial_{trial.number}"
        )

        try:
            result = trainer.train(
                dataset_path=self.dataset_path,
                output_dir=output_dir,
            )

            # Return the objective metric
            objective_metric = self.sweep_config.objective_metric
            if objective_metric in result.metrics:
                return result.metrics[objective_metric]
            elif objective_metric == "eval_loss":
                return result.eval_loss
            elif objective_metric == "train_loss":
                return result.train_loss
            else:
                logger.warning("Metric '%s' not found, using eval_loss.", objective_metric)
                return result.eval_loss

        except TrainingError as e:
            logger.error("Trial %d failed: %s", trial.number, e)
            raise
        except Exception as e:
            logger.error("Trial %d unexpected error: %s", trial.number, e)
            return float("inf") if self.sweep_config.direction == "minimize" else float("-inf")
