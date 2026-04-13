"""
Output guardrails for format validation, provenance checking,
claim-level NLI entailment, and numerical consistency.

Ensures the LLM output matches expected structure, is grounded in
retrieved context, and doesn't hallucinate facts or numbers.
"""

from __future__ import annotations

import json
import logging
import re
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


class ClaimLevelNLIGuard(GuardrailLayer):
    """
    Decomposes the LLM output into atomic claims (sentence-level)
    and checks each claim's entailment against retrieved context chunks
    using a cross-encoder NLI model.
    
    This catches fine-grained hallucinations that block-level cosine
    similarity misses — e.g., a response that is topically relevant
    but contains fabricated specific claims.
    """

    # NLI label mapping for cross-encoder models
    NLI_LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def __init__(self, entailment_threshold: float = 0.5, nli_model_name: str | None = None):
        self.settings = get_settings()
        self._guardrails_config = self.settings.guardrails
        self._entailment_threshold = entailment_threshold
        self._nli_model_name = nli_model_name or "cross-encoder/nli-deberta-v3-small"
        self._nli_model = None

    @property
    def name(self) -> str:
        return "claim_level_nli"

    def _get_nli_model(self):
        """Lazy-load the NLI cross-encoder model."""
        if self._nli_model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._nli_model = CrossEncoder(self._nli_model_name)
                logger.info("NLI model loaded: %s", self._nli_model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed — ClaimLevelNLIGuard disabled. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                logger.error("Failed to load NLI model: %s", e)
        return self._nli_model

    @staticmethod
    def _decompose_claims(text: str) -> list[str]:
        """
        Split text into atomic claims (sentence-level).
        Filters out very short fragments and structural markers.
        """
        # Split on sentence-ending punctuation, keeping the delimiter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        claims = []
        for sent in sentences:
            sent = sent.strip()
            # Skip very short fragments, headers, bullet markers
            if len(sent) < 15:
                continue
            # Skip lines that are just labels/headers
            if sent.endswith(":") and len(sent) < 60:
                continue
            # Skip emoji-only or marker-only lines
            if re.match(r'^[\u2600-\u26FF\u2700-\u27BF⚠️✓✅❌]+\s*$', sent):
                continue
            claims.append(sent)
        
        return claims

    async def check(
        self,
        *,
        input_text: str | None = None,
        output_text: str | None = None,
        context: list[str] | None = None,
    ) -> GuardrailResult:
        if not output_text or not context:
            return GuardrailResult(layer_name=self.name, verdict=GuardrailVerdict.PASS)

        nli_model = self._get_nli_model()
        if nli_model is None:
            return GuardrailResult(
                layer_name=self.name,
                verdict=GuardrailVerdict.FLAG,
                reason="NLI model unavailable — claim-level check skipped.",
            )

        try:
            claims = self._decompose_claims(output_text)
            if not claims:
                return GuardrailResult(layer_name=self.name, verdict=GuardrailVerdict.PASS)

            context_text = " ".join(context)
            ungrounded_claims = []

            # Check each claim against the full context
            pairs = [(context_text, claim) for claim in claims]
            scores = nli_model.predict(pairs)

            for claim, score_row in zip(claims, scores):
                # score_row is [entailment, neutral, contradiction] probabilities
                if hasattr(score_row, '__len__') and len(score_row) >= 3:
                    entailment_score = float(score_row[0])
                else:
                    # Single score model — treat as entailment probability
                    entailment_score = float(score_row)

                if entailment_score < self._entailment_threshold:
                    ungrounded_claims.append(claim)

            if ungrounded_claims:
                ratio = len(ungrounded_claims) / len(claims)
                mode = self._guardrails_config.mode

                if mode == "strict" and ratio > 0.3:
                    verdict = GuardrailVerdict.BLOCK
                elif ratio > 0.5:
                    verdict = GuardrailVerdict.FLAG
                else:
                    verdict = GuardrailVerdict.FLAG

                return GuardrailResult(
                    layer_name=self.name,
                    verdict=verdict,
                    score=1.0 - ratio,
                    reason=f"{len(ungrounded_claims)}/{len(claims)} claims not supported by context.",
                    flagged_spans=ungrounded_claims[:5],  # Limit to 5 for readability
                )

            return GuardrailResult(
                layer_name=self.name,
                verdict=GuardrailVerdict.PASS,
                score=1.0,
            )

        except Exception as e:
            logger.error("Claim-level NLI check failed: %s", e)
            return GuardrailResult(
                layer_name=self.name,
                verdict=GuardrailVerdict.FLAG,
                reason=f"NLI check error: {e}",
            )


class NumericalConsistencyGuard(GuardrailLayer):
    """
    Extracts all numbers from the LLM response and verifies they
    can be traced back to the input context.
    
    Catches hallucinated metrics, invented sales figures, and
    fabricated percentages that look plausible but are unsupported.
    """

    # Numbers that commonly appear in text without needing context support
    WHITELISTED_NUMBERS = {
        # Percentages / common fractions
        "0", "1", "2", "3", "4", "5", "10", "100",
        # Common date/time fragments
        "12", "24", "30", "31", "60", "90", "365",
    }

    def __init__(self, min_grounding_ratio: float = 0.5):
        self.settings = get_settings()
        self._guardrails_config = self.settings.guardrails
        self._min_grounding_ratio = min_grounding_ratio

    @property
    def name(self) -> str:
        return "numerical_consistency"

    async def check(
        self,
        *,
        input_text: str | None = None,
        output_text: str | None = None,
        context: list[str] | None = None,
    ) -> GuardrailResult:
        if not output_text or not context:
            return GuardrailResult(layer_name=self.name, verdict=GuardrailVerdict.PASS)

        try:
            # Extract numbers from response (significant ones: 2+ digits)
            response_numbers = set(re.findall(r'\b(\d{2,}(?:\.\d+)?)\b', output_text))
            
            # Remove whitelisted numbers
            response_numbers -= self.WHITELISTED_NUMBERS
            
            if not response_numbers:
                return GuardrailResult(layer_name=self.name, verdict=GuardrailVerdict.PASS)

            # Extract numbers from input context
            context_text = " ".join(context)
            input_text_combined = (input_text or "") + " " + context_text
            context_numbers = set(re.findall(r'\b(\d{2,}(?:\.\d+)?)\b', input_text_combined))

            # Check grounding
            grounded = response_numbers & context_numbers
            ungrounded = response_numbers - context_numbers

            if not response_numbers:
                ratio = 1.0
            else:
                ratio = len(grounded) / len(response_numbers)

            if ratio < self._min_grounding_ratio:
                mode = self._guardrails_config.mode
                verdict = GuardrailVerdict.BLOCK if mode == "strict" else GuardrailVerdict.FLAG

                return GuardrailResult(
                    layer_name=self.name,
                    verdict=verdict,
                    score=ratio,
                    reason=(
                        f"Only {len(grounded)}/{len(response_numbers)} numbers traceable to context. "
                        f"Potentially fabricated: {sorted(list(ungrounded))[:10]}"
                    ),
                    flagged_spans=sorted(list(ungrounded))[:10],
                )

            return GuardrailResult(
                layer_name=self.name,
                verdict=GuardrailVerdict.PASS,
                score=ratio,
            )

        except Exception as e:
            logger.error("Numerical consistency check failed: %s", e)
            return GuardrailResult(
                layer_name=self.name,
                verdict=GuardrailVerdict.FLAG,
                reason=f"Check failed: {e}",
            )
