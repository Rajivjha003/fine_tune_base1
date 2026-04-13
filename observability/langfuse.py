"""
Langfuse Tracing Integration.

Hooks into LiteLLM and Langchain to trace all LLM calls, RAG retrievals,
and agent steps. Traces are correlated using trace_ids.

Includes custom score posting for faithfulness, hallucination rate,
and guardrail results to enable quality tracking over time.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)


class LangfuseTracker:
    """Manages Langfuse tracing configuration, initialization, and custom scoring."""

    def __init__(self):
        self.settings = get_settings()
        self._enabled = bool(
            self.settings.langfuse_public_key and self.settings.langfuse_secret_key
        )
        self._callbacks = None
        self._client = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_client(self):
        """Lazy-load the Langfuse client for direct API calls."""
        if self._client is None and self._enabled:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=self.settings.langfuse_public_key,
                    secret_key=self.settings.langfuse_secret_key,
                    host=self.settings.langfuse_host,
                )
            except ImportError:
                logger.warning("langfuse SDK missing — custom scoring unavailable.")
            except Exception as e:
                logger.error("Failed to initialize Langfuse client: %s", e)
        return self._client

    def init_tracing(self) -> None:
        """Initialize globals for Langfuse if enabled."""
        if not self._enabled:
            logger.info("Langfuse tracing disabled (keys not set).")
            return

        try:
            import litellm
            
            # Configure LiteLLM to use Langfuse
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.settings.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.settings.langfuse_secret_key
            os.environ["LANGFUSE_HOST"] = self.settings.langfuse_host
            
            # litellm config
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["langfuse"]
            
            logger.info("Langfuse tracing enabled for LiteLLM.")
        except ImportError:
            logger.warning("litellm missing, tracing cannot be enabled.")

    def get_langchain_callbacks(self) -> list[Any]:
        """Get the Langchain CallbackHandler for graph tracing."""
        if not self._enabled:
            return []

        if self._callbacks is None:
            try:
                from langfuse.callback import CallbackHandler
                self._callbacks = [CallbackHandler(
                    public_key=self.settings.langfuse_public_key,
                    secret_key=self.settings.langfuse_secret_key,
                    host=self.settings.langfuse_host,
                )]
            except ImportError:
                logger.warning("langfuse SDK missing.")
                self._callbacks = []

        return self._callbacks

    def score_response(
        self,
        trace_id: str,
        *,
        faithfulness_score: float | None = None,
        hallucination_rate: float | None = None,
        guardrail_result: str | None = None,
        numeric_grounding: float | None = None,
    ) -> None:
        """
        Post custom evaluation scores to a Langfuse trace.
        
        These scores enable:
        - Quality monitoring dashboards in Langfuse
        - A/B model comparison via score distributions
        - Alerting on faithfulness degradation
        
        Args:
            trace_id: The Langfuse trace ID to score.
            faithfulness_score: 0-1, how grounded the response is in context.
            hallucination_rate: 0-1, fraction of ungrounded claims.
            guardrail_result: "pass", "flag", or "block".
            numeric_grounding: 0-1, fraction of numbers traceable to context.
        """
        client = self._get_client()
        if client is None:
            return

        try:
            if faithfulness_score is not None:
                client.score(
                    trace_id=trace_id,
                    name="faithfulness_score",
                    value=faithfulness_score,
                    comment="Fraction of response grounded in retrieved context",
                )

            if hallucination_rate is not None:
                client.score(
                    trace_id=trace_id,
                    name="hallucination_rate",
                    value=hallucination_rate,
                    comment="Fraction of claims not supported by context",
                )

            if guardrail_result is not None:
                # Map string to numeric for Langfuse score
                verdict_scores = {"pass": 1.0, "flag": 0.5, "block": 0.0}
                client.score(
                    trace_id=trace_id,
                    name="guardrail_result",
                    value=verdict_scores.get(guardrail_result, 0.5),
                    comment=f"Guardrail verdict: {guardrail_result}",
                )

            if numeric_grounding is not None:
                client.score(
                    trace_id=trace_id,
                    name="numeric_grounding",
                    value=numeric_grounding,
                    comment="Fraction of numeric values traceable to input context",
                )

            logger.debug("Scores posted to Langfuse trace %s", trace_id)

        except Exception as e:
            logger.warning("Failed to post scores to Langfuse: %s", e)

    def log_deployment_event(
        self,
        *,
        from_model: str | None = None,
        to_model: str,
        reason: str = "",
        eval_metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Log a model deployment/swap event to Langfuse as a trace.
        
        Creates a dedicated trace for each deployment event, enabling
        tracking of model version history and promotion reasons.
        """
        client = self._get_client()
        if client is None:
            return

        try:
            trace = client.trace(
                name="model_deployment",
                metadata={
                    "from_model": from_model or "none",
                    "to_model": to_model,
                    "reason": reason,
                    "eval_metrics": eval_metrics or {},
                },
                tags=["deployment", "model_swap"],
            )

            if eval_metrics:
                for metric_name, score_value in eval_metrics.items():
                    client.score(
                        trace_id=trace.id,
                        name=f"deploy_{metric_name}",
                        value=score_value,
                    )

            logger.info(
                "Deployment event logged to Langfuse: %s → %s (trace: %s)",
                from_model or "none", to_model, trace.id,
            )

        except Exception as e:
            logger.warning("Failed to log deployment to Langfuse: %s", e)

    def flush(self) -> None:
        """Flush any pending Langfuse events."""
        client = self._get_client()
        if client:
            try:
                client.flush()
            except Exception:
                pass
