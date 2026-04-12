"""
Prompt template registry for multi-model support.

Each model family has a specific chat format. Using the wrong format
degrades performance significantly. This module auto-selects the correct
template based on the model family from config/models.yaml.
"""

from __future__ import annotations

import logging
from typing import Callable

from core.config import get_settings

logger = logging.getLogger(__name__)

FormatFn = Callable[[str, str, str, str], str]

SYSTEM_PROMPT = (
    "You are a retail demand forecasting and inventory planning "
    "assistant for MerchMix. Provide accurate, data-driven analysis "
    "based on the context provided. If you lack sufficient data, "
    "clearly state what additional information is needed."
)

# ── Token constants (avoid inline special chars) ──────────────────────────
_GEMMA_USER_START = "\x3cstart_of_turn\x3euser"
_GEMMA_USER_END = "\x3cend_of_turn\x3e"
_GEMMA_MODEL_START = "\x3cstart_of_turn\x3emodel"
_GEMMA_MODEL_END = "\x3cend_of_turn\x3e"

_CHATML_IM_START = "\x3c|im_start|\x3e"
_CHATML_IM_END = "\x3c|im_end|\x3e"

_PHI_SYSTEM = "\x3c|system|\x3e"
_PHI_USER = "\x3c|user|\x3e"
_PHI_ASSISTANT = "\x3c|assistant|\x3e"
_PHI_END = "\x3c|end|\x3e"

_ALPACA_BELOW = "### Response:"


def _build_user_content(instruction: str, input_text: str) -> str:
    """Combine instruction and optional input into user content."""
    if input_text and input_text.strip():
        return f"{instruction}\n\nContext: {input_text}"
    return instruction


def format_gemma3_chat(instruction: str, input_text: str, output: str, eos_token: str = "") -> str:
    """Gemma 3 chat format — uses start_of_turn/end_of_turn markers."""
    user_content = _build_user_content(instruction, input_text)
    return (
        f"{_GEMMA_USER_START}\n"
        f"{user_content}{_GEMMA_USER_END}\n"
        f"{_GEMMA_MODEL_START}\n"
        f"{output}{_GEMMA_MODEL_END}{eos_token}"
    )


def format_qwen2_chat(instruction: str, input_text: str, output: str, eos_token: str = "") -> str:
    """Qwen2 / Qwen2.5 ChatML format."""
    user_content = _build_user_content(instruction, input_text)
    return (
        f"{_CHATML_IM_START}system\n"
        f"{SYSTEM_PROMPT}{_CHATML_IM_END}\n"
        f"{_CHATML_IM_START}user\n"
        f"{user_content}{_CHATML_IM_END}\n"
        f"{_CHATML_IM_START}assistant\n"
        f"{output}{_CHATML_IM_END}{eos_token}"
    )


def format_phi4_chat(instruction: str, input_text: str, output: str, eos_token: str = "") -> str:
    """Phi-4 / Phi-4-mini chat format."""
    user_content = _build_user_content(instruction, input_text)
    return (
        f"{_PHI_SYSTEM}\n"
        f"{SYSTEM_PROMPT}{_PHI_END}\n"
        f"{_PHI_USER}\n"
        f"{user_content}{_PHI_END}\n"
        f"{_PHI_ASSISTANT}\n"
        f"{output}{_PHI_END}{eos_token}"
    )


def format_alpaca(instruction: str, input_text: str, output: str, eos_token: str = "") -> str:
    """Generic Alpaca instruction format (fallback for unknown models)."""
    parts = [f"### Instruction:\n{instruction}"]
    if input_text and input_text.strip():
        parts.append(f"\n### Input:\n{input_text}")
    parts.append(f"\n{_ALPACA_BELOW}\n{output}{eos_token}")
    return "".join(parts)


# ── Registry ──────────────────────────────────────────────────────────────

_FORMAT_REGISTRY: dict[str, FormatFn] = {
    "gemma3_chat": format_gemma3_chat,
    "gemma3": format_gemma3_chat,
    "qwen2_chat": format_qwen2_chat,
    "qwen2": format_qwen2_chat,
    "phi4_chat": format_phi4_chat,
    "phi4": format_phi4_chat,
    "alpaca": format_alpaca,
}


def register_format(name: str, fn: FormatFn) -> None:
    """Register a custom prompt format function."""
    _FORMAT_REGISTRY[name] = fn
    logger.info("Registered prompt format: %s", name)


def get_format_fn(format_name: str) -> FormatFn:
    """Get the prompt format function by name."""
    if format_name == "auto":
        logger.warning("format_name='auto' — falling back to alpaca.")
        return format_alpaca
    fn = _FORMAT_REGISTRY.get(format_name)
    if fn is None:
        available = list(_FORMAT_REGISTRY.keys())
        raise ValueError(f"Unknown prompt format '{format_name}'. Available: {available}")
    return fn


def get_format_fn_for_model(model_key: str) -> FormatFn:
    """Auto-select the prompt format function based on model config."""
    settings = get_settings()
    model_spec = settings.models.get_model(model_key)
    fmt = model_spec.prompt_format

    if fmt == "auto":
        # Infer from model family
        family = model_spec.family.lower()
        if family in _FORMAT_REGISTRY:
            logger.info("Auto-detected format '%s' for model '%s'", family, model_key)
            return _FORMAT_REGISTRY[family]
        logger.warning("Cannot auto-detect format for family '%s', using alpaca.", family)
        return format_alpaca

    return get_format_fn(fmt)


class PromptFormatter:
    """
    High-level formatter that wraps a format function with dataset mapping.

    Usage:
        formatter = PromptFormatter.for_model("gemma-3-4b")
        formatted_dataset = formatter.format_dataset(dataset)
    """

    def __init__(self, format_fn: FormatFn, eos_token: str = ""):
        self._format_fn = format_fn
        self._eos_token = eos_token

    @classmethod
    def for_model(cls, model_key: str, eos_token: str = "") -> "PromptFormatter":
        """Create a formatter for a specific model."""
        fn = get_format_fn_for_model(model_key)
        return cls(fn, eos_token=eos_token)

    def format_sample(self, instruction: str, input_text: str = "", output: str = "") -> str:
        """Format a single sample."""
        return self._format_fn(instruction, input_text, output, self._eos_token)

    def format_dataset(self, dataset: list[dict[str, str]]) -> list[str]:
        """Format an entire dataset of Alpaca-format dicts."""
        return [
            self._format_fn(
                item.get("instruction", ""),
                item.get("input", ""),
                item.get("output", ""),
                self._eos_token,
            )
            for item in dataset
        ]

    def create_hf_formatting_func(self):
        """
        Create a formatting function compatible with HuggingFace SFTTrainer.

        Returns a callable that takes a batch dict and returns {"text": [...]}.
        """
        format_fn = self._format_fn
        eos_token = self._eos_token

        def formatting_func(examples):
            texts = []
            instructions = examples.get("instruction", [])
            inputs = examples.get("input", [])
            outputs = examples.get("output", [])
            for inst, inp, out in zip(instructions, inputs, outputs):
                texts.append(format_fn(inst, inp or "", out, eos_token))
            return {"text": texts}

        return formatting_func
