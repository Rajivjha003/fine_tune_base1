"""
Pydantic v2 schemas for all training data types.

Every training sample passes through these validators before entering
the pipeline. Invalid samples are rejected with detailed error messages.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class SampleCategory(str, Enum):
    """Categories for training samples — used for distribution tracking."""

    DEMAND_FORECAST = "demand_forecast"
    REORDER_INVENTORY = "reorder_inventory"
    MIO_PLAN = "mio_plan"
    SELL_THROUGH = "sell_through"
    REFUSAL = "refusal"
    GENERAL = "general"


class TrainingSample(BaseModel):
    """
    A single training sample in Alpaca format.

    Every sample is checksummed for deduplication and lineage tracking.
    The `category` field enables refusal ratio enforcement.
    """

    instruction: str = Field(..., min_length=10, description="The task instruction for the model")
    input: str = Field(default="", description="Optional context/input data")
    output: str = Field(..., min_length=5, description="The expected model response")
    category: SampleCategory = Field(default=SampleCategory.GENERAL, description="Sample category for distribution tracking")
    source: str = Field(default="manual", description="Origin of this sample (manual, synthetic, augmented)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @computed_field
    @property
    def checksum(self) -> str:
        """MD5 hash of instruction+input+output for deduplication."""
        content = f"{self.instruction}|{self.input}|{self.output}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    @field_validator("instruction")
    @classmethod
    def instruction_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Instruction cannot be empty or whitespace-only.")
        return v

    @field_validator("output")
    @classmethod
    def output_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Output cannot be empty or whitespace-only.")
        return v

    def to_alpaca_dict(self) -> dict[str, str]:
        """Convert to standard Alpaca format dict."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSONL line with full metadata."""
        data = {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "category": self.category.value,
            "source": self.source,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }
        return json.dumps(data, ensure_ascii=False)


# ── Specialized Sample Types ──────────────────────────────────────────────


class ForecastSample(TrainingSample):
    """Training sample for demand forecasting tasks."""

    category: SampleCategory = SampleCategory.DEMAND_FORECAST

    @field_validator("instruction")
    @classmethod
    def must_mention_forecast(cls, v: str) -> str:
        v = v.strip()
        forecast_keywords = ["forecast", "predict", "demand", "sales", "project", "estimate"]
        if not any(kw in v.lower() for kw in forecast_keywords):
            raise ValueError(
                f"Forecast sample instruction must contain a forecasting keyword. "
                f"Got: '{v[:80]}...'"
            )
        return v


class MIOSample(TrainingSample):
    """Training sample for MIO (Months of Inventory Outstanding) planning."""

    category: SampleCategory = SampleCategory.MIO_PLAN

    @field_validator("instruction")
    @classmethod
    def must_mention_mio(cls, v: str) -> str:
        v = v.strip()
        mio_keywords = ["mio", "inventory", "reorder", "allocation", "stock", "replenish"]
        if not any(kw in v.lower() for kw in mio_keywords):
            raise ValueError(
                f"MIO sample instruction must contain an inventory-related keyword. "
                f"Got: '{v[:80]}...'"
            )
        return v


class RefusalSample(TrainingSample):
    """Training sample where the model should refuse or express uncertainty."""

    category: SampleCategory = SampleCategory.REFUSAL

    @field_validator("output")
    @classmethod
    def must_express_uncertainty(cls, v: str) -> str:
        v = v.strip()
        refusal_indicators = [
            "cannot", "don't have", "insufficient", "unable",
            "not enough", "need more", "unclear", "I'm not sure",
            "cannot determine", "no data", "not available",
        ]
        if not any(ind in v.lower() for ind in refusal_indicators):
            raise ValueError(
                "Refusal sample output must express uncertainty or inability to answer. "
                f"Got: '{v[:80]}...'"
            )
        return v


# ── Dataset Manifest ──────────────────────────────────────────────────────


class DatasetManifest(BaseModel):
    """
    Metadata about a complete dataset.

    Tracks distribution, checksums, and enforces the refusal_ratio constraint.
    """

    version: str = Field(default="1.0", description="Dataset version string")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_samples: int = 0
    category_distribution: dict[str, int] = Field(default_factory=dict)
    unique_checksums: int = 0
    duplicate_count: int = 0
    file_path: str = ""
    sha256: str = ""

    @computed_field
    @property
    def refusal_ratio(self) -> float:
        """Fraction of samples that are refusal/uncertainty cases."""
        refusal_count = self.category_distribution.get(SampleCategory.REFUSAL.value, 0)
        if self.total_samples == 0:
            return 0.0
        return refusal_count / self.total_samples

    def validate_refusal_ratio(self, min_ratio: float = 0.20) -> bool:
        """Check if refusal ratio meets the minimum threshold."""
        return self.refusal_ratio >= min_ratio

    def summary(self) -> str:
        """Human-readable summary of the dataset."""
        lines = [
            f"Dataset v{self.version} — {self.total_samples} samples",
            f"  Created: {self.created_at.isoformat()}",
            f"  Unique: {self.unique_checksums} | Duplicates removed: {self.duplicate_count}",
            f"  Refusal ratio: {self.refusal_ratio:.1%} (min: 20%)",
            "  Distribution:",
        ]
        for cat, count in sorted(self.category_distribution.items()):
            pct = count / self.total_samples * 100 if self.total_samples else 0
            lines.append(f"    {cat}: {count} ({pct:.0f}%)")
        return "\n".join(lines)


# ── API Response Schemas (used by guardrails/format_validator) ─────────────


class ForecastResponse(BaseModel):
    """Structured response for demand forecasting queries."""

    sku_id: str
    forecast_horizon_days: int
    predicted_demand: list[float]
    confidence_interval: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Upper and lower bounds: {'lower': [...], 'upper': [...]}"
    )
    reasoning: str = ""
    data_sources: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class MIOPlanResponse(BaseModel):
    """Structured response for MIO planning queries."""

    sku_id: str
    current_stock: float
    reorder_point: float
    recommended_order_qty: float
    months_of_inventory: float
    allocation_plan: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    risk_factors: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """General chat response from the agent."""

    message: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RefusalResponse(BaseModel):
    """Response when the model cannot or should not answer."""

    message: str
    reason: str
    missing_data: list[str] = Field(default_factory=list)
    suggestion: str = ""
