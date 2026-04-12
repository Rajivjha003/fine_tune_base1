"""
Async event bus for cross-layer communication.

Allows any layer to emit events without coupling to the listeners.
Observability, evaluation triggers, and model swap orchestration
all hook into this bus.

Usage:
    from core.events import event_bus, Event

    # Register a listener
    @event_bus.on("model_swapped")
    async def handle_swap(event: Event):
        print(f"Model swapped to {event.data['new_model']}")

    # Emit an event
    await event_bus.emit("model_swapped", data={"new_model": "qwen2.5-3b"})
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine
from uuid import uuid4

logger = logging.getLogger(__name__)

# Type alias for async event handlers
EventHandler = Callable[["Event"], Coroutine[Any, Any, None]]


@dataclass(frozen=True)
class Event:
    """Immutable event payload passed to all listeners."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""  # Which module emitted this event

    def __str__(self) -> str:
        return f"Event({self.name}, id={self.event_id}, source={self.source})"


class EventBus:
    """
    Simple async event bus with wildcard support.

    Features:
    - Named event subscriptions
    - Wildcard '*' listener receives ALL events
    - Errors in one handler don't block others
    - Event history for debugging (ring buffer)
    """

    def __init__(self, *, history_size: int = 100):
        self._handlers: dict[str, list[EventHandler]] = {}
        self._history: list[Event] = []
        self._history_size = history_size

    def on(self, event_name: str) -> Callable[[EventHandler], EventHandler]:
        """
        Decorator to register an async handler for a named event.

        @event_bus.on("model_swapped")
        async def handle(event: Event):
            ...
        """

        def decorator(func: EventHandler) -> EventHandler:
            self.subscribe(event_name, func)
            return func

        return decorator

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Register a handler for a named event."""
        if event_name not in self._handlers:
            self._handlers[event_name] = []
        if handler not in self._handlers[event_name]:
            self._handlers[event_name].append(handler)
            logger.debug("Subscribed %s to event '%s'", handler.__qualname__, event_name)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """Remove a handler from a named event."""
        if event_name in self._handlers:
            self._handlers[event_name] = [h for h in self._handlers[event_name] if h is not handler]

    async def emit(self, event_name: str, *, data: dict[str, Any] | None = None, source: str = "") -> Event:
        """
        Emit an event to all registered handlers (including wildcard '*').

        Handlers run concurrently via asyncio.gather. Errors are logged
        but do not propagate — one failing handler never blocks another.
        """
        event = Event(name=event_name, data=data or {}, source=source)

        # Record in history
        self._history.append(event)
        if len(self._history) > self._history_size:
            self._history = self._history[-self._history_size :]

        # Collect all handlers: specific + wildcard
        handlers = list(self._handlers.get(event_name, []))
        handlers.extend(self._handlers.get("*", []))

        if not handlers:
            logger.debug("Event '%s' emitted with no handlers", event_name)
            return event

        logger.info("Emitting event '%s' to %d handlers", event_name, len(handlers))

        # Run handlers concurrently, capturing errors
        results = await asyncio.gather(
            *[self._safe_call(handler, event) for handler in handlers],
            return_exceptions=True,
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Handler %s failed for event '%s': %s",
                    handlers[i].__qualname__,
                    event_name,
                    result,
                    exc_info=result,
                )

        return event

    @staticmethod
    async def _safe_call(handler: EventHandler, event: Event) -> None:
        """Call a handler with error isolation."""
        try:
            await handler(event)
        except Exception:
            raise

    def get_history(self, event_name: str | None = None, limit: int = 20) -> list[Event]:
        """Return recent events, optionally filtered by name."""
        events = self._history if event_name is None else [e for e in self._history if e.name == event_name]
        return events[-limit:]

    def clear(self) -> None:
        """Remove all handlers and history."""
        self._handlers.clear()
        self._history.clear()


# ── Global singleton ──────────────────────────────────────────────────────

event_bus = EventBus()

# ── Standard event names (constants to avoid typos) ───────────────────────

TRAINING_STARTED = "training_started"
TRAINING_COMPLETED = "training_completed"
TRAINING_FAILED = "training_failed"
MODEL_EXPORTED = "model_exported"
MODEL_REGISTERED = "model_registered"
MODEL_PROMOTED = "model_promoted"
MODEL_SWAPPED = "model_swapped"
MODEL_ROLLBACK = "model_rollback"
EVAL_STARTED = "eval_started"
EVAL_COMPLETED = "eval_completed"
EVAL_GATE_PASSED = "eval_gate_passed"
EVAL_GATE_FAILED = "eval_gate_failed"
RAG_INDEX_REBUILT = "rag_index_rebuilt"
GUARDRAIL_TRIGGERED = "guardrail_triggered"
VRAM_ALERT = "vram_alert"
SYSTEM_READY = "system_ready"
SYSTEM_ERROR = "system_error"
