"""
Guardrail Pipeline Orchestrator.

Runs a chain of responsibility for input and output guardrails.
Raises GuardrailBlockError if a BLOCK verdict is reached.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import get_settings
from core.exceptions import GuardrailBlockError
from core.protocols import GuardrailLayer, GuardrailVerdict

logger = logging.getLogger(__name__)


class GuardrailPipeline:
    """
    Orchestrates execution of multiple guardrail layers.
    Can be used before (input) and after (output) LLM execution.
    """

    def __init__(self):
        self.settings = get_settings()
        self.mode = self.settings.guardrails.mode  # "strict" or "audit"
        self._input_layers: list[GuardrailLayer] = []
        self._output_layers: list[GuardrailLayer] = []
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize all active guardrail layers."""
        from guardrails.input_guard import InputSanitizer
        from guardrails.output_guard import FormatValidator, ProvenanceGuard

        if self.settings.guardrails.input_sanitizer.enabled:
            self._input_layers.append(InputSanitizer())

        if self.settings.guardrails.format_validator.enabled:
            self._output_layers.append(FormatValidator())

        if self.settings.guardrails.provenance_guard.enabled:
            self._output_layers.append(ProvenanceGuard())

    @property
    def has_input_guards(self) -> bool:
        return len(self._input_layers) > 0

    @property
    def has_output_guards(self) -> bool:
        return len(self._output_layers) > 0

    async def guard_input(self, input_text: str) -> str:
        """
        Run input guardrails.
        Automatically applies any required redactions (e.g., PII).
        Throws GuardrailBlockError if blocked.
        """
        if not self._input_layers:
            return input_text

        sanitized_text = input_text
        for layer in self._input_layers:
            result = await layer.check(input_text=sanitized_text)

            if result.verdict == GuardrailVerdict.BLOCK:
                # In audit mode, we log but don't block
                if self.mode == "audit":
                    logger.warning("[AUDIT MODE] Input guard '%s' blocked: %s", layer.name, result.reason)
                else:
                    raise GuardrailBlockError(
                        f"Blocked by input guard '{layer.name}': {result.reason}",
                        details={"flagged_spans": result.flagged_spans},
                    )

            if result.verdict == GuardrailVerdict.FLAG:
                logger.warning("Input flagged by '%s': %s", layer.name, result.reason)
                
                # Auto-redact PII if flagged
                if layer.name == "input_sanitizer":
                    sanitizer = layer  # Type ignore
                    sanitized_text = sanitizer.redact_pii(sanitized_text)

        return sanitized_text

    async def guard_output(
        self,
        output_text: str,
        input_text: str | None = None,
        context: list[str] | None = None,
    ) -> str:
        """
        Run output guardrails.
        Throws GuardrailBlockError if blocked.
        """
        if not self._output_layers:
            return output_text

        for layer in self._output_layers:
            result = await layer.check(
                output_text=output_text,
                input_text=input_text,
                context=context,
            )

            if result.verdict == GuardrailVerdict.BLOCK:
                if self.mode == "audit":
                    logger.warning("[AUDIT MODE] Output guard '%s' blocked: %s", layer.name, result.reason)
                else:
                    raise GuardrailBlockError(
                        f"Blocked by output guard '{layer.name}': {result.reason}"
                    )

            if result.verdict == GuardrailVerdict.FLAG:
                logger.warning("Output flagged by '%s': %s", layer.name, result.reason)

        return output_text
