"""
Protocol definitions for all layer interfaces.

Using Python Protocols (structural subtyping) so that implementations
don't need to inherit from a base class — they just need to implement
the required methods. This keeps layers fully decoupled.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


# ── Inference ─────────────────────────────────────────────────────────────


@runtime_checkable
class InferenceBackend(Protocol):
    """Any system that can produce text completions (Ollama, vLLM, remote API)."""

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.05,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a completion dict with at least 'text' and 'usage' keys."""
        ...

    async def health_check(self) -> bool:
        """Return True if backend is reachable and healthy."""
        ...


@runtime_checkable
class ChatBackend(Protocol):
    """Chat-oriented inference with message history."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.05,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Return a response dict with 'message', 'usage' keys."""
        ...


# ── Embeddings ────────────────────────────────────────────────────────────


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into embedding vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        ...

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


# ── Vector Store ──────────────────────────────────────────────────────────


@runtime_checkable
class VectorStore(Protocol):
    """Persistent vector store for document embeddings."""

    def add(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        """Add documents to the store."""
        ...

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query by vector similarity. Returns list of {id, score, metadata}."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Remove documents by ID."""
        ...

    def count(self) -> int:
        """Return the total number of stored documents."""
        ...


# ── Retriever ─────────────────────────────────────────────────────────────


@runtime_checkable
class Retriever(Protocol):
    """Retrieves relevant context for a query."""

    async def retrieve(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top-k relevant documents. Returns list of {text, score, metadata}."""
        ...


# ── Guardrail ─────────────────────────────────────────────────────────────


@runtime_checkable
class GuardrailLayer(Protocol):
    """A single guardrail check in the chain of responsibility."""

    @property
    def name(self) -> str:
        """Human-readable name of this guardrail."""
        ...

    async def check(
        self,
        *,
        input_text: str | None = None,
        output_text: str | None = None,
        context: list[str] | None = None,
    ) -> "GuardrailResult":
        """Run the check. Returns a GuardrailResult."""
        ...


# ── Evaluator ─────────────────────────────────────────────────────────────


@runtime_checkable
class Evaluator(Protocol):
    """Evaluates model output quality."""

    async def evaluate(
        self,
        predictions: list[dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Return a dict of metric_name → score."""
        ...


# ── Metrics Collector ─────────────────────────────────────────────────────


@runtime_checkable
class MetricsCollector(Protocol):
    """Collects and exports metrics (Prometheus, file, etc.)."""

    def increment(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment a counter."""
        ...

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Observe a value for a histogram."""
        ...

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge to a specific value."""
        ...


# ── Artifact Store ────────────────────────────────────────────────────────


@runtime_checkable
class ArtifactStorage(Protocol):
    """Manages model artifacts (LoRA weights, GGUF files)."""

    def store(self, artifact_path: str, metadata: dict[str, Any]) -> str:
        """Store an artifact. Returns a unique artifact ID / URI."""
        ...

    def retrieve(self, artifact_id: str) -> str:
        """Return local filesystem path to the artifact."""
        ...

    def verify_checksum(self, artifact_id: str) -> bool:
        """Verify integrity via stored checksum."""
        ...


# ── Supporting data classes ───────────────────────────────────────────────

from dataclasses import dataclass, field
from enum import Enum


class GuardrailVerdict(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    BLOCK = "block"


@dataclass
class GuardrailResult:
    """Result of a single guardrail check."""

    layer_name: str
    verdict: GuardrailVerdict
    score: float | None = None
    reason: str = ""
    flagged_spans: list[str] = field(default_factory=list)
