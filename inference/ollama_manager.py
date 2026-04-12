"""
Ollama lifecycle manager.

Handles model registration, health checks, model listing,
and VRAM status via Ollama's REST API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from core.config import get_settings
from core.exceptions import OllamaConnectionError

logger = logging.getLogger(__name__)


class OllamaManager:
    """
    Manages the Ollama inference server lifecycle.

    Operations:
    - Create models from GGUF + Modelfile
    - Delete models
    - Health check
    - List loaded models
    - Pull models from registry
    """

    def __init__(self):
        self.settings = get_settings()
        self._base_url = self.settings.ollama_host

    async def health_check(self) -> bool:
        """Check if Ollama is running and responsive."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """List all models registered in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                return resp.json().get("models", [])
        except httpx.ConnectError:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Ensure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            logger.error("Failed to list Ollama models: %s", e)
            return []

    async def model_exists(self, model_name: str) -> bool:
        """Check if a specific model is registered in Ollama."""
        models = await self.list_models()
        return any(model_name in m.get("name", "") for m in models)

    async def create_model(
        self,
        model_name: str,
        gguf_path: str,
        model_key: str,
    ) -> None:
        """
        Create an Ollama model from a GGUF file.

        Generates a Modelfile and sends it to Ollama's create endpoint.
        """
        # Generate Modelfile content
        modelfile_content = self._build_modelfile(gguf_path, model_key)

        logger.info("Creating Ollama model '%s' from GGUF: %s", model_name, gguf_path)

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:  # GGUF loading can be slow
                resp = await client.post(
                    f"{self._base_url}/api/create",
                    json={
                        "name": model_name,
                        "modelfile": modelfile_content,
                    },
                )
                resp.raise_for_status()
                logger.info("Ollama model '%s' created successfully.", model_name)
        except httpx.ConnectError:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Ensure Ollama is running: `ollama serve`"
            )
        except httpx.HTTPStatusError as e:
            logger.error("Ollama model creation failed: %s — %s", e.response.status_code, e.response.text)
            raise

    async def delete_model(self, model_name: str) -> None:
        """Delete a model from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.delete(
                    f"{self._base_url}/api/delete",
                    json={"name": model_name},
                )
                if resp.status_code == 200:
                    logger.info("Deleted Ollama model: %s", model_name)
                else:
                    logger.warning("Delete returned status %d for '%s'", resp.status_code, model_name)
        except Exception as e:
            logger.error("Failed to delete model '%s': %s", model_name, e)

    async def pull_model(self, model_name: str) -> None:
        """Pull a model from the Ollama registry."""
        logger.info("Pulling model '%s' from Ollama registry...", model_name)
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{self._base_url}/api/pull",
                    json={"name": model_name},
                )
                resp.raise_for_status()
                logger.info("Model '%s' pulled successfully.", model_name)
        except Exception as e:
            logger.error("Failed to pull model '%s': %s", model_name, e)
            raise

    async def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get detailed info about a specific model."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self._base_url}/api/show",
                    json={"name": model_name},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("Failed to get info for '%s': %s", model_name, e)
            return {}

    async def generate(
        self,
        model_name: str,
        prompt: str,
        *,
        temperature: float = 0.05,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Direct generation via Ollama API (bypasses LiteLLM).

        Used for testing and fallback scenarios.
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload: dict[str, Any] = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                }
                if stop:
                    payload["options"]["stop"] = stop

                resp = await client.post(f"{self._base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()

                return {
                    "text": data.get("response", ""),
                    "model": data.get("model", model_name),
                    "done": data.get("done", True),
                    "usage": {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_duration_ms": data.get("total_duration", 0) / 1e6,
                    },
                }
        except httpx.ConnectError:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self._base_url}")
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            raise

    def _build_modelfile(self, gguf_path: str, model_key: str) -> str:
        """Generate Ollama Modelfile content."""
        model_spec = self.settings.models.get_model(model_key)

        from training.prompt_templates import SYSTEM_PROMPT

        lines = [
            f"FROM {gguf_path}",
            "",
            f'SYSTEM """{SYSTEM_PROMPT}"""',
            "",
            "PARAMETER temperature 0.05",
            "PARAMETER top_p 0.9",
            "PARAMETER repeat_penalty 1.15",
            f"PARAMETER num_ctx {model_spec.context_window}",
            "PARAMETER num_gpu 99",
        ]

        return "\n".join(lines)
