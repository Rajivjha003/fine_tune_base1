"""
Model export pipeline: LoRA merge → GGUF quantization → Ollama Modelfile.

Takes a trained LoRA adapter, merges it with the base model,
exports as quantized GGUF, generates an Ollama Modelfile, and
registers artifacts in MLflow.

Usage:
    from training.export import ModelExporter
    exporter = ModelExporter(model_key="gemma-3-4b")
    result = exporter.export(lora_path="outputs/lora_gemma_3_4b")
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.events import MODEL_EXPORTED, event_bus
from core.exceptions import ExportError

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Result of a model export operation."""

    success: bool
    model_key: str
    gguf_path: str = ""
    modelfile_path: str = ""
    gguf_size_mb: float = 0.0
    sha256: str = ""
    quant_method: str = ""
    error: str = ""


class ModelExporter:
    """
    Exports fine-tuned LoRA adapters to serving-ready formats.

    Pipeline:
    1. Load base model + LoRA adapter via Unsloth
    2. Merge LoRA weights into base model
    3. Export as GGUF with specified quantization
    4. Generate Ollama Modelfile
    5. Compute checksum and register in MLflow
    """

    def __init__(self, model_key: str):
        self.settings = get_settings()
        self.model_key = model_key
        self.model_spec = self.settings.models.get_model(model_key)

    def export(
        self,
        lora_path: str | Path,
        *,
        quant_method: str = "q4_k_m",
        output_dir: str | None = None,
    ) -> ExportResult:
        """
        Export LoRA adapter to GGUF + Ollama Modelfile.

        Args:
            lora_path: Path to the saved LoRA adapter directory.
            quant_method: GGUF quantization method (q4_k_m, q5_k_m, q8_0).
            output_dir: Override output directory.

        Returns:
            ExportResult with paths and metadata.
        """
        lora_path = Path(lora_path).resolve()
        if not lora_path.exists():
            raise ExportError(f"LoRA adapter not found: {lora_path}")

        if quant_method not in self.model_spec.gguf_quant_methods:
            logger.warning(
                "Quantization '%s' not in model's supported methods %s. Proceeding anyway.",
                quant_method,
                self.model_spec.gguf_quant_methods,
            )

        if output_dir is None:
            safe_name = self.model_key.replace(".", "_").replace("-", "_")
            output_dir = str(self.settings.outputs_dir / f"gguf_{safe_name}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Merge LoRA and export GGUF
            gguf_path = self._merge_and_export(lora_path, output_path, quant_method)

            # Step 2: Generate Ollama Modelfile
            modelfile_path = self._generate_modelfile(gguf_path, output_path)

            # Step 3: Compute checksum
            sha256 = self._compute_checksum(gguf_path)
            gguf_size = Path(gguf_path).stat().st_size / (1024 * 1024)

            # Step 4: Log to MLflow
            self._log_to_mlflow(gguf_path, modelfile_path, sha256, quant_method, gguf_size)

            # Step 5: Emit event
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                event_bus.emit(
                    MODEL_EXPORTED,
                    data={
                        "model_key": self.model_key,
                        "gguf_path": gguf_path,
                        "quant_method": quant_method,
                        "size_mb": gguf_size,
                    },
                    source="export",
                )
            )

            result = ExportResult(
                success=True,
                model_key=self.model_key,
                gguf_path=gguf_path,
                modelfile_path=modelfile_path,
                gguf_size_mb=gguf_size,
                sha256=sha256,
                quant_method=quant_method,
            )

            logger.info(
                "Export complete: %s (%.1fMB, %s, sha256=%s...)",
                gguf_path,
                gguf_size,
                quant_method,
                sha256[:16],
            )

            return result

        except Exception as e:
            logger.error("Export failed: %s", e, exc_info=True)
            raise ExportError(f"Export failed for '{self.model_key}': {e}") from e

    def _merge_and_export(self, lora_path: Path, output_path: Path, quant_method: str) -> str:
        """Merge LoRA into base model and export as GGUF."""
        from unsloth import FastLanguageModel

        profile = self.settings.training.get_profile(self.model_key)

        logger.info("Loading base model + LoRA adapter for merging...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(lora_path),
            max_seq_length=profile.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        logger.info("Exporting to GGUF with quantization '%s'...", quant_method)
        model.save_pretrained_gguf(
            str(output_path),
            tokenizer,
            quantization_method=quant_method,
        )

        # Find the generated GGUF file
        gguf_files = list(output_path.glob("*.gguf"))
        if not gguf_files:
            raise ExportError(f"No GGUF file generated in {output_path}")

        return str(gguf_files[0])

    def _generate_modelfile(self, gguf_path: str, output_path: Path) -> str:
        """Generate an Ollama Modelfile for the exported model."""
        from jinja2 import Template

        template_path = self.settings.project_root / "inference" / "Modelfile.template"

        if template_path.exists():
            template_str = template_path.read_text(encoding="utf-8")
        else:
            # Default template
            template_str = self._default_modelfile_template()

        template = Template(template_str)
        content = template.render(
            gguf_path=gguf_path,
            model_name=self.model_spec.ollama_name,
            system_prompt=self._get_system_prompt(),
            temperature=0.05,
            top_p=0.9,
            repeat_penalty=1.15,
            num_ctx=self.model_spec.context_window,
        )

        modelfile_path = output_path / "Modelfile"
        modelfile_path.write_text(content, encoding="utf-8")
        logger.info("Modelfile generated: %s", modelfile_path)

        return str(modelfile_path)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Modelfile."""
        from training.prompt_templates import SYSTEM_PROMPT

        return SYSTEM_PROMPT

    @staticmethod
    def _default_modelfile_template() -> str:
        """Fallback Modelfile template if inference/Modelfile.template doesn't exist."""
        return """FROM {{ gguf_path }}

SYSTEM \"\"\"{{ system_prompt }}\"\"\"

PARAMETER temperature {{ temperature }}
PARAMETER top_p {{ top_p }}
PARAMETER repeat_penalty {{ repeat_penalty }}
PARAMETER num_ctx {{ num_ctx }}
PARAMETER num_gpu 99
"""

    @staticmethod
    def _compute_checksum(file_path: str) -> str:
        """Compute SHA256 of a file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _log_to_mlflow(
        self,
        gguf_path: str,
        modelfile_path: str,
        sha256: str,
        quant_method: str,
        size_mb: float,
    ) -> None:
        """Log export artifacts to MLflow."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.settings.mlflow_experiment_name)

            with mlflow.start_run(run_name=f"export-{self.model_key}-{quant_method}"):
                mlflow.log_params({
                    "model_key": self.model_key,
                    "quant_method": quant_method,
                })
                mlflow.log_metrics({
                    "gguf_size_mb": size_mb,
                })
                mlflow.set_tag("gguf_path", gguf_path)
                mlflow.set_tag("gguf_sha256", sha256)
                mlflow.log_artifact(modelfile_path)
                logger.info("Export artifacts logged to MLflow.")
        except Exception as e:
            logger.warning("MLflow export logging failed: %s", e)
