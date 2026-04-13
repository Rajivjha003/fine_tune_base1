"""
MLflow Model Registry manager.

Handles model registration, alias-based promotion (champion/challenger),
rollback, and version listing using MLflow 3.x alias API.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import get_settings
from core.events import MODEL_PROMOTED, MODEL_REGISTERED, event_bus
from core.exceptions import ModelNotFoundError, PromotionBlockedError, RegistryError
from registry.schemas import ModelAlias, ModelVersion, ModelVersionStatus, PromotionRecord

logger = logging.getLogger(__name__)

# Registry name pattern
_REGISTRY_PREFIX = "MerchFine"


class ModelManager:
    """
    MLflow Model Registry manager using alias-based promotion.

    Model lifecycle:
    1. Register → version created as @challenger
    2. Evaluate → eval suite runs
    3. Promote → @challenger becomes @champion, old champion → @archived
    4. Or reject → @challenger tagged as eval_failed
    """

    def __init__(self):
        self.settings = get_settings()
        self._client = None

    @property
    def client(self):
        """Lazy-initialized MLflow client."""
        if self._client is None:
            import mlflow

            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            self._client = mlflow.tracking.MlflowClient(self.settings.mlflow_tracking_uri)
        return self._client

    def _registry_name(self, model_key: str) -> str:
        """Generate the MLflow registered model name."""
        safe_key = model_key.replace("-", "_").replace(".", "_")
        return f"{_REGISTRY_PREFIX}-{safe_key}"

    def register_model(
        self,
        model_key: str,
        *,
        source: str,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> ModelVersion:
        """
        Register a new model version in MLflow.

        The new version is automatically assigned @challenger alias.

        Args:
            model_key: Model identifier from config (e.g., "gemma-3-4b").
            source: Path to the model artifacts (LoRA dir or GGUF).
            run_id: MLflow run ID that produced this model.
            tags: Additional tags to set on the version.

        Returns:
            ModelVersion with the registered version number.
        """
        import mlflow

        registry_name = self._registry_name(model_key)

        # Ensure registered model exists
        try:
            self.client.get_registered_model(registry_name)
        except Exception:
            self.client.create_registered_model(
                registry_name,
                description=f"MerchFine fine-tuned {model_key} for retail forecasting",
            )
            logger.info("Created registered model: %s", registry_name)

        # Create model version
        try:
            mv = self.client.create_model_version(
                name=registry_name,
                source=source,
                run_id=run_id,
                tags=tags or {},
            )

            version_num = int(mv.version)

            # Set as challenger
            self.client.set_registered_model_alias(
                registry_name,
                ModelAlias.CHALLENGER.value,
                version_num,
            )

            logger.info(
                "Registered %s version %d as @challenger (source=%s)",
                registry_name,
                version_num,
                source,
            )

            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
            loop.run_until_complete(
                event_bus.emit(
                    MODEL_REGISTERED,
                    data={"model_key": model_key, "version": version_num, "alias": "challenger"},
                    source="model_manager",
                )
            )

            model_spec = self.settings.models.get_model(model_key)
            return ModelVersion(
                model_key=model_key,
                version=version_num,
                hf_id=model_spec.hf_id,
                alias=ModelAlias.CHALLENGER.value,
                status=ModelVersionStatus.REGISTERED,
                mlflow_run_id=run_id,
                tags=tags or {},
            )

        except Exception as e:
            raise RegistryError(f"Failed to register model version: {e}") from e

    def promote_challenger(self, model_key: str, *, reason: str = "") -> PromotionRecord:
        """
        Promote @challenger to @champion.

        The current @champion (if any) is moved to @archived.
        """
        registry_name = self._registry_name(model_key)

        try:
            # Get current challenger
            challenger_mv = self.client.get_model_version_by_alias(registry_name, ModelAlias.CHALLENGER.value)
            challenger_version = int(challenger_mv.version)
        except Exception as e:
            raise ModelNotFoundError(f"No @challenger found for '{registry_name}': {e}") from e

        # Archive current champion (if exists)
        old_champion_version = None
        try:
            champion_mv = self.client.get_model_version_by_alias(registry_name, ModelAlias.CHAMPION.value)
            old_champion_version = int(champion_mv.version)
            self.client.set_registered_model_alias(
                registry_name,
                ModelAlias.ARCHIVED.value,
                old_champion_version,
            )
            # Remove champion alias from old version
            self.client.delete_registered_model_alias(registry_name, ModelAlias.CHAMPION.value)
            logger.info("Archived previous champion: version %d", old_champion_version)
        except Exception:
            logger.info("No existing champion to archive.")

        # Promote challenger to champion
        self.client.set_registered_model_alias(
            registry_name,
            ModelAlias.CHAMPION.value,
            challenger_version,
        )

        # Remove challenger alias
        try:
            self.client.delete_registered_model_alias(registry_name, ModelAlias.CHALLENGER.value)
        except Exception:
            pass

        logger.info(
            "Promoted %s version %d: @challenger → @champion",
            registry_name,
            challenger_version,
        )

        record = PromotionRecord(
            from_version=old_champion_version,
            to_version=challenger_version,
            from_alias=ModelAlias.ARCHIVED.value if old_champion_version else None,
            to_alias=ModelAlias.CHAMPION.value,
            reason=reason,
        )

        return record

    def set_champion(self, model_key: str) -> None:
        """
        Set a model as the active champion (used by model_switcher during rollback).

        This is a simplified version of promote_challenger for direct alias management.
        """
        registry_name = self._registry_name(model_key)
        try:
            # Find the latest version for this model
            versions = self.client.search_model_versions(f"name='{registry_name}'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                self.client.set_registered_model_alias(
                    registry_name,
                    ModelAlias.CHAMPION.value,
                    int(latest.version),
                )
                logger.info("Set %s version %s as @champion", registry_name, latest.version)
        except Exception as e:
            logger.warning("Failed to set champion alias: %s", e)

    def get_champion(self, model_key: str) -> ModelVersion | None:
        """Get the current @champion model version, or None."""
        registry_name = self._registry_name(model_key)
        try:
            mv = self.client.get_model_version_by_alias(registry_name, ModelAlias.CHAMPION.value)
            model_spec = self.settings.models.get_model(model_key)
            return ModelVersion(
                model_key=model_key,
                version=int(mv.version),
                hf_id=model_spec.hf_id,
                alias=ModelAlias.CHAMPION.value,
                status=ModelVersionStatus.PROMOTED,
                mlflow_run_id=mv.run_id,
                tags=dict(mv.tags) if mv.tags else {},
            )
        except Exception:
            return None

    def list_versions(self, model_key: str) -> list[ModelVersion]:
        """List all registered versions for a model."""
        registry_name = self._registry_name(model_key)
        try:
            versions = self.client.search_model_versions(f"name='{registry_name}'")
            model_spec = self.settings.models.get_model(model_key)
            return [
                ModelVersion(
                    model_key=model_key,
                    version=int(v.version),
                    hf_id=model_spec.hf_id,
                    status=ModelVersionStatus.REGISTERED,
                    mlflow_run_id=v.run_id,
                    tags=dict(v.tags) if v.tags else {},
                )
                for v in versions
            ]
        except Exception as e:
            logger.warning("Failed to list versions for '%s': %s", registry_name, e)
            return []

    def rollback(self, model_key: str, *, reason: str = "") -> bool:
        """
        Roll back to the previous @archived version.

        Current @champion → @archived
        Previous @archived → @champion
        """
        registry_name = self._registry_name(model_key)

        try:
            # Get archived version
            archived_mv = self.client.get_model_version_by_alias(registry_name, ModelAlias.ARCHIVED.value)
            archived_version = int(archived_mv.version)
        except Exception:
            logger.warning("No @archived version to roll back to.")
            return False

        try:
            # Current champion → archived
            champion_mv = self.client.get_model_version_by_alias(registry_name, ModelAlias.CHAMPION.value)
            # Just overwrite the aliases
            self.client.set_registered_model_alias(registry_name, ModelAlias.CHAMPION.value, archived_version)
            logger.info("Rolled back to version %d as @champion", archived_version)
            return True
        except Exception as e:
            logger.error("Rollback failed: %s", e)
            return False
