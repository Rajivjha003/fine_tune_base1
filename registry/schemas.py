"""
Pydantic models for the model registry layer.

Defines structured types for model versions, run metadata,
promotion records, and artifact manifests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelAlias(str, Enum):
    """Standard model aliases used in the registry."""

    CHAMPION = "champion"
    CHALLENGER = "challenger"
    ARCHIVED = "archived"


class ModelVersionStatus(str, Enum):
    """Status of a model version."""

    REGISTERED = "registered"
    EVALUATING = "evaluating"
    PROMOTED = "promoted"
    EVAL_FAILED = "eval_failed"
    ARCHIVED = "archived"


class ModelVersion(BaseModel):
    """A specific version of a registered model."""

    model_key: str
    version: int
    hf_id: str
    alias: str | None = None
    status: ModelVersionStatus = ModelVersionStatus.REGISTERED
    gguf_path: str | None = None
    gguf_sha256: str | None = None
    lora_path: str | None = None
    quant_method: str = "q4_k_m"
    train_loss: float | None = None
    eval_loss: float | None = None
    eval_scores: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    promoted_at: datetime | None = None
    mlflow_run_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class RunMetadata(BaseModel):
    """Metadata for an MLflow training run."""

    run_id: str
    experiment_name: str
    model_key: str
    status: str = "RUNNING"
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    artifact_paths: list[str] = Field(default_factory=list)


class PromotionRecord(BaseModel):
    """Record of a model promotion event."""

    from_version: int | None = None
    to_version: int
    from_alias: str | None = None
    to_alias: str = ModelAlias.CHAMPION.value
    promoted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    eval_scores: dict[str, float] = Field(default_factory=dict)
    promoted_by: str = "model_switcher"
    reason: str = ""


class ArtifactManifest(BaseModel):
    """Manifest tracking all artifacts for a model version."""

    model_key: str
    version: int
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Map of artifact_type -> path (e.g., 'gguf' -> '/path/to/model.gguf')",
    )
    checksums: dict[str, str] = Field(
        default_factory=dict,
        description="Map of artifact_type -> SHA256 checksum",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
