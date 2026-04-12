"""
Output guardrails for format validation and provenance checking.
Ensures the LLM output matches expected structure and doesn't hallucinate facts.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.config import get_settings
from core.protocols import GuardrailLayer, GuardrailResult, GuardrailVerdict

logger = logging.getLogger(__name__)


class FormatValidator(GuardrailLayer):
    """
    Validates that output matches expected JSON structure if required.
    """

    def __init__(self):
        self.settings = get_settings()
        self._config = self.settings.guardrails.format_validator

    @property
    def name(self) -> str:
        return "format_validator"

    async def check(
        self,
        *,
        input_text: str | None = None,
        output_text: str | None = None,
        context: list[str] | None = None,
    ) -> GuardrailResult:
        if not self._config.enabled or not output_text:
            return GuardrailResult(layer_name=self.name, verdict=GuardrailVerdict.PASS)

        # Basic check: If expected to be JSON, try parsing
        # Here we just heuristically check if it looks like JSON if strict mode is on
        text = output_text.strip()
        
        if self._config.strict_json_mode:
            if not (text.startswith("{") and text.endswith("}") or text.startswith("[") and text.endswith("]")):
                return GuardrailResult(
                    layer_name=self.name,
                    verdict=GuardrailVerdict.BLOCK,
                    reason="Strict JSON mode enabled, but output does not start with { or [.",
                )
            
            try:
                json.loads(text)
            except json.JSONDecodeError as e:
                return GuardrailResult(
                    layer_name=self.name,
                    verdict=GuardrailVerdict.BLOCK,
                    reason=f"Strict JSON mode enabled, but output is invalid JSON: {e}",
                )

        return GuardrailResult(layer_name=self.name, verdict=GuardrailVerdict.PASS)


class ProvenanceGuard(GuardrailLayer):
    """
    Checks if output facts are grounded in the provided context
    using embedding cosine similarity.
    """

    def __init__(self):
        self.settings = get_settings()
        self._config = self.settings.guardrails.provenance_guard
        self._embed_model = None

    @property
    def name(self) -> str:
        return "provenance_guard"

    def _get_embed_model(self):
        if self._embed_model is None:
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                model_name = self._config.embedding_model or self.settings.rag.embedding.model_name
                self._embed_model = HuggingFaceEmbedding(
                    model_name=model_name,
                    device=self.settings.rag.embedding.device,
                )
            except ImportError:
                logger.warning("ProvenanceGuard requires llama-index embeddings.")
        return self._embed_model

    async def check(
        self,
        *,
        input_text: str | None = None,
        output_text: str | None = None,
        context: list[str] | None = None,
    ) -> GuardrailResult:
        if not self._config.enabled or not output_text or not context:
            return GuardrailResult(layer_name=self.name, verdict=GuardrailVerdict.PASS)

        embed_model = self._get_embed_model()
        if not embed_model:
            return GuardrailResult(
                layer_name=self.name,
                verdict=GuardrailVerdict.FLAG,
                reason="Embedding model unavailable for provenance checking."
            )

        try:
            import numpy as np
            
            # Very basic block-level similarity
            out_emb = np.array(embed_model.get_text_embedding(output_text))
            
            max_sim = 0.0
            for ctx in context:
                ctx_emb = np.array(embed_model.get_text_embedding(ctx))
                sim = np.dot(out_emb, ctx_emb) / (np.linalg.norm(out_emb) * np.linalg.norm(ctx_emb))
                if sim > max_sim:
                    max_sim = sim

            if max_sim < self._config.cosine_threshold:
                action = self._config.flag_action
                if self.settings.guardrails.mode == "strict":
                    action = "block"
                    
                verdict = GuardrailVerdict.BLOCK if action == "block" else GuardrailVerdict.FLAG
                return GuardrailResult(
                    layer_name=self.name,
                    verdict=verdict,
                    score=float(max_sim),
                    reason=f"Output not grounded in context. Max similarity {max_sim:.2f} < threshold {self._config.cosine_threshold}",
                )

            return GuardrailResult(
                layer_name=self.name,
                verdict=GuardrailVerdict.PASS,
                score=float(max_sim),
            )
            
        except Exception as e:
            logger.error("Provenance check failed: %s", e)
            return GuardrailResult(
                layer_name=self.name,
                verdict=GuardrailVerdict.FLAG,
                reason=f"Check failed: {e}"
            )
