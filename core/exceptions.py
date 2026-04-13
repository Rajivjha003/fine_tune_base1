"""
Custom exception hierarchy for MerchFine.

Every layer raises domain-specific exceptions that inherit from MerchFineError.
This allows callers to catch at any granularity level.
"""

from __future__ import annotations


class MerchFineError(Exception):
    """Base exception for all MerchFine errors."""

    def __init__(self, message: str, *, details: dict | None = None):
        self.details = details or {}
        super().__init__(message)


# ── Layer 1: Data ────────────────────────────────────────────────────────


class DataValidationError(MerchFineError):
    """A training sample or dataset failed Pydantic validation."""


class DataPipelineError(MerchFineError):
    """General data pipeline failure (I/O, checksums, splits)."""


class RefusalRatioError(DataValidationError):
    """Dataset does not meet the minimum refusal sample ratio."""


# ── Layer 2: Training ────────────────────────────────────────────────────


class TrainingError(MerchFineError):
    """Fine-tuning run failed."""


class VRAMExceededError(TrainingError):
    """GPU VRAM usage exceeded the safety threshold during training."""


class ExportError(TrainingError):
    """GGUF export or Modelfile generation failed."""


# ── Layer 3: Registry ────────────────────────────────────────────────────


class RegistryError(MerchFineError):
    """MLflow registry operation failed."""


class ModelNotFoundError(RegistryError):
    """Requested model version or alias does not exist in the registry."""


class PromotionBlockedError(RegistryError):
    """Model promotion blocked by quality gate failure."""


# ── Layer 4: Inference ───────────────────────────────────────────────────


class InferenceError(MerchFineError):
    """Inference request failed."""


class OllamaConnectionError(InferenceError):
    """Cannot connect to Ollama server."""


class CircuitBreakerOpenError(InferenceError):
    """Circuit breaker is open — backend temporarily unavailable."""


class FallbackExhaustedError(InferenceError):
    """All models in the fallback chain have failed."""


# ── Layer 5: RAG ─────────────────────────────────────────────────────────


class RAGError(MerchFineError):
    """RAG pipeline failure (indexing, retrieval, or reranking)."""


class IndexBuildError(RAGError):
    """Failed to build or load the vector store index."""


class RetrievalError(RAGError):
    """Document retrieval failed."""


# Aliases used by rag.retriever and rag.indexer modules
RAGRetrievalError = RetrievalError
RAGIndexError = IndexBuildError


# ── Layer 6: Agents ──────────────────────────────────────────────────────


class AgentError(MerchFineError):
    """Agentic workflow failure."""


class ToolExecutionError(AgentError):
    """A tool call within the agent graph failed."""


# ── Layer 7: Guardrails ──────────────────────────────────────────────────


class GuardrailError(MerchFineError):
    """Base class for guardrail-related issues."""


class GuardrailBlockError(GuardrailError):
    """Response blocked by a guardrail in strict mode."""


class InputRejectedError(GuardrailError):
    """Input failed sanitization checks."""


class HallucinationDetectedError(GuardrailError):
    """Provenance or consistency check flagged hallucination."""


# ── Layer 8: Evaluation ──────────────────────────────────────────────────


class EvaluationError(MerchFineError):
    """Evaluation run failed to complete."""


class QualityGateFailedError(EvaluationError):
    """One or more hard quality gates failed — promotion blocked."""

    def __init__(self, message: str, *, failed_gates: list[str], details: dict | None = None):
        self.failed_gates = failed_gates
        super().__init__(message, details=details)


# ── Layer 10: System ─────────────────────────────────────────────────────


class ModelSwapError(MerchFineError):
    """Model hot-swap failed at some step in the protocol."""


class SystemInitError(MerchFineError):
    """System bootstrap / health check failed."""
