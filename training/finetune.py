"""
QLoRA fine-tuning engine using Unsloth + SFTTrainer.

All hyperparameters are read from config/training.yaml.
No values are hardcoded — the trainer adapts to any model
in the registry by reading its profile.

Usage:
    from training.finetune import QLoRATrainer
    trainer = QLoRATrainer(model_key="gemma-3-4b")
    result = trainer.train(dataset_path="data/processed/train_v1.0_*.jsonl")
"""

from __future__ import annotations

import unsloth
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.events import TRAINING_COMPLETED, TRAINING_FAILED, TRAINING_STARTED, event_bus
from core.exceptions import TrainingError

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result of a fine-tuning run."""

    success: bool
    model_key: str
    output_dir: str = ""
    train_loss: float = 0.0
    eval_loss: float = 0.0
    train_time_seconds: float = 0.0
    vram_peak_gb: float = 0.0
    num_samples: int = 0
    mlflow_run_id: str = ""
    error: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


class QLoRATrainer:
    """
    Config-driven QLoRA fine-tuning engine.

    Loads model via Unsloth → applies LoRA → trains with SFTTrainer →
    logs everything to MLflow. All parameters from training.yaml.
    """

    def __init__(self, model_key: str):
        self.settings = get_settings()
        self.model_key = model_key
        self.model_spec = self.settings.models.get_model(model_key)
        self.train_config = self.settings.training.get_merged_profile(model_key)
        self.profile = self.settings.training.get_profile(model_key)

    def train(
        self,
        dataset_path: str | Path,
        *,
        output_dir: str | None = None,
        resume_from: str | None = None,
    ) -> TrainResult:
        """
        Execute a full QLoRA fine-tuning run.

        Args:
            dataset_path: Path to the training JSONL file.
            output_dir: Where to save the LoRA adapter. Auto-generated if None.
            resume_from: Checkpoint path to resume training from.

        Returns:
            TrainResult with metrics and artifact paths.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(event_bus.emit(
                TRAINING_STARTED,
                data={"model_key": self.model_key, "dataset": str(dataset_path)},
                source="finetune",
            ))
        except RuntimeError:
            asyncio.run(event_bus.emit(
                TRAINING_STARTED,
                data={"model_key": self.model_key, "dataset": str(dataset_path)},
                source="finetune",
            ))

        start_time = time.time()

        if output_dir is None:
            safe_name = self.model_key.replace(".", "_").replace("-", "_")
            output_dir = str(self.settings.outputs_dir / f"lora_{safe_name}")

        try:
            # Step 1: Load model with Unsloth
            model, tokenizer = self._load_model()

            # Step 2: Apply LoRA
            model = self._apply_lora(model)

            # Step 3: Load and prepare dataset
            dataset = self._load_dataset(dataset_path, tokenizer)

            # Step 4: Configure trainer
            trainer = self._create_trainer(model, tokenizer, dataset, output_dir, resume_from)

            # Step 5: Run MLflow tracked training
            metrics = self._run_training(trainer, output_dir)

            elapsed = time.time() - start_time

            # Get VRAM peak
            vram_peak = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    vram_peak = torch.cuda.max_memory_reserved(0) / (1024**3)
            except Exception:
                pass

            result = TrainResult(
                success=True,
                model_key=self.model_key,
                output_dir=output_dir,
                train_loss=metrics.get("train_loss", 0.0),
                eval_loss=metrics.get("eval_loss", 0.0),
                train_time_seconds=elapsed,
                vram_peak_gb=vram_peak,
                num_samples=len(dataset),
                metrics=metrics,
            )

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(event_bus.emit(
                    TRAINING_COMPLETED,
                    data={
                        "model_key": self.model_key,
                        "output_dir": output_dir,
                        "train_loss": result.train_loss,
                        "eval_loss": result.eval_loss,
                        "duration_s": elapsed,
                    },
                    source="finetune",
                ))
            except RuntimeError:
                asyncio.run(event_bus.emit(
                    TRAINING_COMPLETED,
                    data={
                        "model_key": self.model_key,
                        "output_dir": output_dir,
                        "train_loss": result.train_loss,
                        "eval_loss": result.eval_loss,
                        "duration_s": elapsed,
                    },
                    source="finetune",
                ))

            logger.info(
                "Training complete: loss=%.4f, eval_loss=%.4f, time=%.1fmin, vram=%.2fGB",
                result.train_loss,
                result.eval_loss,
                elapsed / 60,
                vram_peak,
            )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Training failed after %.1fs: %s", elapsed, e, exc_info=True)

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(event_bus.emit(
                    TRAINING_FAILED,
                    data={"model_key": self.model_key, "error": str(e)},
                    source="finetune",
                ))
            except RuntimeError:
                asyncio.run(event_bus.emit(
                    TRAINING_FAILED,
                    data={"model_key": self.model_key, "error": str(e)},
                    source="finetune",
                ))

            raise TrainingError(f"Fine-tuning '{self.model_key}' failed: {e}") from e

    def _load_model(self):
        """Load the base model using Unsloth for memory optimization."""
        from unsloth import FastLanguageModel

        logger.info(
            "Loading model '%s' (%s) with 4-bit quantization...",
            self.model_key,
            self.model_spec.hf_id,
        )

        max_seq_length = self.profile.max_seq_length

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_spec.hf_id,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=self.train_config.get("load_in_4bit", True),
        )

        logger.info("Model loaded. Tokenizer/Processor type: %s", type(tokenizer).__name__)
        return model, tokenizer

    def _apply_lora(self, model):
        """Apply LoRA adapter configuration from the training profile."""
        from unsloth import FastLanguageModel

        logger.info(
            "Applying LoRA: r=%d, alpha=%d, dropout=%.2f",
            self.profile.lora_r,
            self.profile.lora_alpha,
            self.profile.lora_dropout,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.profile.lora_r,
            lora_alpha=self.profile.lora_alpha,
            lora_dropout=self.profile.lora_dropout,
            target_modules=self.profile.target_modules,
            bias="none",
            use_gradient_checkpointing=self.train_config.get("use_gradient_checkpointing", "unsloth"),
            random_state=self.train_config.get("seed", 42),
        )

        # Log trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = trainable / total * 100
        logger.info("LoRA applied: %d/%d trainable params (%.2f%%)", trainable, total, pct)

        return model

    def _load_dataset(self, path: str | Path, tokenizer):
        """Load JSONL dataset and prepare for training."""
        from datasets import Dataset, load_dataset

        path = Path(path).resolve()

        if path.suffix == ".jsonl":
            dataset = load_dataset("json", data_files=str(path), split="train")
        elif path.suffix == ".json":
            dataset = load_dataset("json", data_files=str(path), split="train")
        elif path.is_dir():
            # Load all JSONL files in directory
            files = sorted(path.glob("train_*.jsonl"))
            if not files:
                raise TrainingError(f"No training files found in {path}")
            dataset = load_dataset("json", data_files=[str(f) for f in files], split="train")
        else:
            raise TrainingError(f"Unsupported dataset format: {path}")

        logger.info("Dataset loaded: %d samples", len(dataset))

        # Apply prompt formatting
        from training.prompt_templates import PromptFormatter

        formatter = PromptFormatter.for_model(self.model_key, eos_token=tokenizer.eos_token)
        formatting_func = formatter.create_hf_formatting_func()

        dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

        return dataset

    def _create_trainer(self, model, tokenizer, dataset, output_dir: str, resume_from: str | None):
        """Create the SFTTrainer with all callbacks."""
        from trl import SFTConfig, SFTTrainer

        from training.callbacks import EvalLossCallback, MLflowLoggingCallback, VRAMMonitorCallback

        # Split dataset for eval
        eval_ratio = self.train_config.get("eval_split_ratio", 0.1)
        split = dataset.train_test_split(test_size=eval_ratio, seed=self.train_config.get("seed", 42))

        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=self.profile.per_device_train_batch_size,
            gradient_accumulation_steps=self.profile.gradient_accumulation_steps,
            num_train_epochs=self.profile.num_train_epochs,
            learning_rate=self.profile.learning_rate,
            lr_scheduler_type=self.train_config.get("lr_scheduler_type", "cosine"),
            warmup_steps=self.train_config.get("warmup_steps", 10),
            weight_decay=self.train_config.get("weight_decay", 0.01),
            max_grad_norm=self.profile.max_grad_norm,
            optim=self.train_config.get("optim", "adamw_8bit"),
            fp16=self.train_config.get("fp16", False),
            bf16=self.train_config.get("bf16", True),
            logging_steps=self.train_config.get("logging_steps", 5),
            save_strategy=self.train_config.get("save_strategy", "epoch"),
            eval_strategy=self.train_config.get("eval_strategy", "no"),
            seed=self.train_config.get("seed", 42),
            report_to="none",  # We handle MLflow via custom callback
            load_best_model_at_end=self.train_config.get("load_best_model_at_end", False),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataset_text_field="text",
            max_seq_length=self.profile.max_seq_length,
            packing=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            args=training_args,
            callbacks=[
                VRAMMonitorCallback(),
                EvalLossCallback(),
                MLflowLoggingCallback(),
            ],
        )

        logger.info(
            "Trainer created: %d train / %d eval samples, %d epochs",
            len(split["train"]),
            len(split["test"]),
            self.profile.num_train_epochs,
        )

        return trainer

    def _run_training(self, trainer, output_dir: str) -> dict[str, float]:
        """Execute training within an MLflow run context."""
        metrics = {}

        try:
            import mlflow

            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.settings.mlflow_experiment_name)

            with mlflow.start_run(run_name=f"finetune-{self.model_key}") as run:
                mlflow.set_tag("model_key", self.model_key)
                mlflow.set_tag("model_hf_id", self.model_spec.hf_id)
                mlflow.set_tag("model_tier", self.model_spec.tier)

                # Log LoRA config
                mlflow.log_params({
                    "lora_r": self.profile.lora_r,
                    "lora_alpha": self.profile.lora_alpha,
                    "lora_dropout": self.profile.lora_dropout,
                    "max_seq_length": self.profile.max_seq_length,
                })

                # Train
                train_result = trainer.train()
                metrics = {k: v for k, v in train_result.metrics.items() if isinstance(v, (int, float))}

                # Log final metrics
                mlflow.log_metrics(metrics)

                # Save LoRA adapter
                trainer.save_model(output_dir)
                mlflow.log_artifacts(output_dir, artifact_path="lora_adapter")

                logger.info("MLflow run ID: %s", run.info.run_id)

        except ImportError:
            logger.warning("MLflow not available — training without experiment tracking.")
            train_result = trainer.train()
            metrics = {k: v for k, v in train_result.metrics.items() if isinstance(v, (int, float))}
            trainer.save_model(output_dir)

        return metrics
