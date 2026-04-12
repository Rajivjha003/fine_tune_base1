"""
Unified inference gateway via LiteLLM.

Provides a single interface for all LLM completions, routing through
LiteLLM for fallback handling, caching, and observability. Falls back
to direct Ollama calls if LiteLLM is unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.config import get_settings
from core.exceptions import FallbackExhaustedError, InferenceError

logger = logging.getLogger(__name__)


class InferenceGateway:
    """
    Primary interface for all LLM inference.

    Routing priority:
    1. LiteLLM proxy (if running) → handles caching + fallback
    2. Direct Ollama API → no caching, no fallback
    3. Raise FallbackExhaustedError
    """

    def __init__(self):
        self.settings = get_settings()
        self._litellm_available: bool | None = None

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a text completion.

        Args:
            prompt: The input prompt.
            model: Model identifier. Defaults to primary model.
            temperature: Sampling temperature. Defaults from config.
            max_tokens: Max generation length. Defaults from config.
            stop: Stop sequences.

        Returns:
            Dict with 'text', 'model', 'usage' keys.
        """
        temperature = temperature if temperature is not None else self.settings.inference.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.settings.inference.default_max_tokens

        start_time = time.time()

        # Try LiteLLM first
        if await self._check_litellm():
            try:
                result = await self._complete_via_litellm(
                    prompt, model=model, temperature=temperature,
                    max_tokens=max_tokens, stop=stop,
                )
                result["latency_ms"] = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.warning("LiteLLM completion failed, trying direct Ollama: %s", e)

        # Fallback to direct Ollama
        try:
            result = await self._complete_via_ollama(
                prompt, model=model, temperature=temperature,
                max_tokens=max_tokens, stop=stop,
            )
            result["latency_ms"] = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            raise FallbackExhaustedError(
                f"All inference backends failed. Last error: {e}"
            ) from e

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Chat completion with message history.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Max generation length.

        Returns:
            Dict with 'message', 'model', 'usage' keys.
        """
        temperature = temperature if temperature is not None else self.settings.inference.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.settings.inference.default_max_tokens

        start_time = time.time()

        # Try LiteLLM first
        if await self._check_litellm():
            try:
                result = await self._chat_via_litellm(
                    messages, model=model, temperature=temperature,
                    max_tokens=max_tokens,
                )
                result["latency_ms"] = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.warning("LiteLLM chat failed, trying direct Ollama: %s", e)

        # Fallback to direct Ollama
        try:
            result = await self._chat_via_ollama(
                messages, model=model, temperature=temperature,
                max_tokens=max_tokens,
            )
            result["latency_ms"] = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            raise FallbackExhaustedError(
                f"All inference backends failed for chat. Last error: {e}"
            ) from e

    async def _check_litellm(self) -> bool:
        """Check if LiteLLM proxy is reachable (cached result)."""
        if self._litellm_available is not None:
            return self._litellm_available

        try:
            import httpx

            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.settings.litellm_host}/health")
                self._litellm_available = resp.status_code == 200
        except Exception:
            self._litellm_available = False

        return self._litellm_available

    async def _complete_via_litellm(
        self, prompt: str, *, model: str | None, temperature: float,
        max_tokens: int, stop: list[str] | None,
    ) -> dict[str, Any]:
        """Complete via LiteLLM's OpenAI-compatible API."""
        import litellm

        # Default to primary model
        if model is None:
            primary_key, primary_spec = self.settings.models.get_primary_model()
            model = f"ollama/{primary_spec.ollama_name}"

        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            api_base=self.settings.ollama_host,
        )

        choice = response.choices[0]
        return {
            "text": choice.message.content or "",
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

    async def _chat_via_litellm(
        self, messages: list[dict[str, str]], *, model: str | None,
        temperature: float, max_tokens: int,
    ) -> dict[str, Any]:
        """Chat via LiteLLM."""
        import litellm

        if model is None:
            primary_key, primary_spec = self.settings.models.get_primary_model()
            model = f"ollama/{primary_spec.ollama_name}"

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=self.settings.ollama_host,
        )

        choice = response.choices[0]
        return {
            "message": {
                "role": "assistant",
                "content": choice.message.content or "",
            },
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

    async def _complete_via_ollama(
        self, prompt: str, *, model: str | None, temperature: float,
        max_tokens: int, stop: list[str] | None,
    ) -> dict[str, Any]:
        """Direct Ollama API completion."""
        from inference.ollama_manager import OllamaManager

        manager = OllamaManager()

        if model is None:
            primary_key, primary_spec = self.settings.models.get_primary_model()
            model = primary_spec.ollama_name

        # Strip ollama/ prefix if present
        if model.startswith("ollama/"):
            model = model[7:]

        return await manager.generate(
            model_name=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    async def _chat_via_ollama(
        self, messages: list[dict[str, str]], *, model: str | None,
        temperature: float, max_tokens: int,
    ) -> dict[str, Any]:
        """Direct Ollama chat API."""
        import httpx

        if model is None:
            primary_key, primary_spec = self.settings.models.get_primary_model()
            model = primary_spec.ollama_name

        if model.startswith("ollama/"):
            model = model[7:]

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.settings.ollama_host}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            return {
                "message": data.get("message", {"role": "assistant", "content": ""}),
                "model": data.get("model", model),
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
            }

    def reset_cache(self) -> None:
        """Reset the LiteLLM availability cache."""
        self._litellm_available = None
