"""
Input sanitization layer for protecting the LLM from prompt injection
and stripping PII before it reaches the model.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.config import get_settings
from core.protocols import GuardrailLayer, GuardrailResult, GuardrailVerdict

logger = logging.getLogger(__name__)


class InputSanitizer(GuardrailLayer):
    """
    Checks input constraints (length) and scans for prompt injection
    and PII (Personally Identifiable Information).
    """

    def __init__(self):
        self.settings = get_settings()
        self._config = self.settings.guardrails.input_sanitizer

        # Compile PII regexes once
        self._pii_patterns: dict[str, re.Pattern] = {}
        if self._config.pii_detection.get("enabled", True):
            patterns = self._config.pii_detection.get("patterns", {})
            for name, pattern_str in patterns.items():
                try:
                    self._pii_patterns[name] = re.compile(pattern_str)
                except re.error as e:
                    logger.warning("Invalid PII regex for %s: %s", name, e)

    @property
    def name(self) -> str:
        return "input_sanitizer"

    async def check(
        self,
        *,
        input_text: str | None = None,
        output_text: str | None = None,
        context: list[str] | None = None,
    ) -> GuardrailResult:
        """Run all input checks on input_text."""
        if not self._config.enabled or not input_text:
            return GuardrailResult(layer_name=self.name, verdict=GuardrailVerdict.PASS)

        flagged_spans = []

        # 1. Length check
        if len(input_text) > self._config.max_query_length:
            return GuardrailResult(
                layer_name=self.name,
                verdict=GuardrailVerdict.BLOCK,
                reason=f"Input exceeds maximum length of {self._config.max_query_length} characters.",
            )

        # 2. Injection check
        text_lower = input_text.lower()
        for pattern in self._config.injection_patterns:
            if pattern.lower() in text_lower:
                flagged_spans.append(pattern)
                logger.warning("Prompt injection detected: matched '%s'", pattern)

        if flagged_spans:
            return GuardrailResult(
                layer_name=self.name,
                verdict=GuardrailVerdict.BLOCK,
                reason="Potential prompt injection detected.",
                flagged_spans=flagged_spans,
            )

        # 3. PII Detection
        if self._pii_patterns:
            pii_found = []
            for name, pattern in self._pii_patterns.items():
                if pattern.search(input_text):
                    pii_found.append(name)
                    logger.warning("PII detected matching category: %s", name)
            
            if pii_found:
                action = self._config.pii_detection.get("action", "redact")
                if action == "block":
                    return GuardrailResult(
                        layer_name=self.name,
                        verdict=GuardrailVerdict.BLOCK,
                        reason=f"PII detected ({', '.join(pii_found)}) and blocking is enabled.",
                        flagged_spans=pii_found,
                    )
                else:
                    return GuardrailResult(
                        layer_name=self.name,
                        verdict=GuardrailVerdict.FLAG,
                        reason=f"PII detected ({', '.join(pii_found)}). Needs redaction.",
                        flagged_spans=pii_found,
                    )

        return GuardrailResult(layer_name=self.name, verdict=GuardrailVerdict.PASS)

    def redact_pii(self, input_text: str) -> str:
        """Helper to redact PII from input text."""
        if not self._pii_patterns:
            return input_text

        text = input_text
        for name, pattern in self._pii_patterns.items():
            text = pattern.sub(f"[REDACTED {name.upper()}]", text)
        return text
