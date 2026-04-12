"""
Circuit breaker and fallback chain for inference resilience.

Tracks consecutive failures per backend and temporarily disables
failing backends to prevent cascade failures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.config import get_settings
from core.exceptions import CircuitBreakerOpenError, FallbackExhaustedError

logger = logging.getLogger(__name__)


class CircuitState:
    """State enum for circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Backend disabled
    HALF_OPEN = "half_open"  # Testing if backend recovered


@dataclass
class BackendState:
    """Tracks the health state of a single backend."""

    name: str
    state: str = CircuitState.CLOSED
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_requests: int = 0
    total_failures: int = 0

    def record_success(self) -> None:
        """Record a successful request."""
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED
        self.last_success_time = time.time()
        self.total_requests += 1

    def record_failure(self, threshold: int) -> None:
        """Record a failed request. Opens circuit if threshold exceeded."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self.total_requests += 1
        self.total_failures += 1

        if self.consecutive_failures >= threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "Circuit OPEN for '%s': %d consecutive failures (threshold=%d)",
                self.name,
                self.consecutive_failures,
                threshold,
            )

    def is_available(self, cooldown: int) -> bool:
        """Check if this backend is available for requests."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if cooldown period has elapsed
            elapsed = time.time() - self.last_failure_time
            if elapsed >= cooldown:
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit HALF-OPEN for '%s' after %.0fs cooldown.",
                    self.name,
                    elapsed,
                )
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return True

        return False


class FallbackChain:
    """
    Manages fallback routing with circuit breakers.

    When the primary backend fails repeatedly, routes to the next
    backend in the chain. Auto-recovers when the cooldown expires.
    """

    def __init__(self):
        self.settings = get_settings()
        self._backends: dict[str, BackendState] = {}
        self._chain: list[str] = []
        self._threshold = self.settings.inference.circuit_breaker_threshold
        self._cooldown = self.settings.inference.circuit_breaker_cooldown
        self._init_chain()

    def _init_chain(self) -> None:
        """Initialize the fallback chain from model config."""
        # Primary model first
        try:
            primary_key, primary_spec = self.settings.models.get_primary_model()
            self._chain.append(primary_spec.ollama_name)
            self._backends[primary_spec.ollama_name] = BackendState(name=primary_spec.ollama_name)
        except Exception:
            pass

        # Fallback models
        for key, spec in self.settings.models.get_fallback_models():
            if spec.ollama_name not in self._chain:
                self._chain.append(spec.ollama_name)
                self._backends[spec.ollama_name] = BackendState(name=spec.ollama_name)

        logger.info("Fallback chain initialized: %s", " → ".join(self._chain))

    def get_next_available(self) -> str:
        """
        Get the next available backend in the chain.

        Raises FallbackExhaustedError if all backends are unavailable.
        """
        for backend_name in self._chain:
            state = self._backends.get(backend_name)
            if state and state.is_available(self._cooldown):
                return backend_name

        raise FallbackExhaustedError(
            f"All backends in fallback chain are unavailable. "
            f"Chain: {self._chain}. "
            f"States: {[(b.name, b.state, b.consecutive_failures) for b in self._backends.values()]}"
        )

    def record_success(self, backend_name: str) -> None:
        """Record a successful request for a backend."""
        state = self._backends.get(backend_name)
        if state:
            state.record_success()

    def record_failure(self, backend_name: str) -> None:
        """Record a failed request for a backend."""
        state = self._backends.get(backend_name)
        if state:
            state.record_failure(self._threshold)

    def get_status(self) -> list[dict[str, Any]]:
        """Get the current status of all backends."""
        return [
            {
                "name": state.name,
                "state": state.state,
                "consecutive_failures": state.consecutive_failures,
                "total_requests": state.total_requests,
                "total_failures": state.total_failures,
                "available": state.is_available(self._cooldown),
            }
            for state in self._backends.values()
        ]

    def reset(self, backend_name: str | None = None) -> None:
        """Reset circuit breaker state for one or all backends."""
        if backend_name:
            state = self._backends.get(backend_name)
            if state:
                state.state = CircuitState.CLOSED
                state.consecutive_failures = 0
                logger.info("Circuit breaker reset for '%s'", backend_name)
        else:
            for state in self._backends.values():
                state.state = CircuitState.CLOSED
                state.consecutive_failures = 0
            logger.info("All circuit breakers reset.")
