"""
Artifact store for GGUF files, LoRA adapters, and checksum verification.

Tracks artifact locations, validates integrity via SHA256,
and manages the local artifact filesystem.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from core.config import get_settings
from registry.schemas import ArtifactManifest

logger = logging.getLogger(__name__)


class ArtifactStore:
    """
    Local artifact store with checksum validation.

    Tracks all model artifacts (GGUF, LoRA adapters, Modelfiles)
    with SHA256 checksums for integrity verification.
    """

    def __init__(self):
        self.settings = get_settings()
        self._manifest_dir = self.settings.outputs_dir / "manifests"
        self._manifest_dir.mkdir(parents=True, exist_ok=True)

    def register_artifact(
        self,
        model_key: str,
        version: int,
        artifact_type: str,
        artifact_path: str,
    ) -> ArtifactManifest:
        """
        Register an artifact with its checksum.

        Args:
            model_key: Model identifier.
            version: Model version number.
            artifact_type: Type (e.g., "gguf", "lora", "modelfile").
            artifact_path: Local filesystem path to the artifact.

        Returns:
            Updated ArtifactManifest.
        """
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        # Compute checksum
        checksum = self._compute_sha256(path)

        # Load or create manifest
        manifest = self._load_manifest(model_key, version)
        manifest.artifacts[artifact_type] = str(path.resolve())
        manifest.checksums[artifact_type] = checksum

        # Save manifest
        self._save_manifest(manifest)

        logger.info(
            "Registered artifact: %s/%s v%d (%s, sha256=%s...)",
            model_key,
            artifact_type,
            version,
            path.name,
            checksum[:16],
        )

        return manifest

    def verify_artifact(self, model_key: str, version: int, artifact_type: str) -> bool:
        """
        Verify an artifact's integrity by comparing stored vs computed checksum.

        Returns True if checksums match, False otherwise.
        """
        manifest = self._load_manifest(model_key, version)

        if artifact_type not in manifest.artifacts:
            logger.warning("Artifact '%s' not registered for %s v%d", artifact_type, model_key, version)
            return False

        artifact_path = Path(manifest.artifacts[artifact_type])
        if not artifact_path.exists():
            logger.error("Artifact file missing: %s", artifact_path)
            return False

        stored_checksum = manifest.checksums.get(artifact_type, "")
        computed_checksum = self._compute_sha256(artifact_path)

        if stored_checksum != computed_checksum:
            logger.error(
                "Checksum mismatch for %s/%s v%d! Stored=%s, Computed=%s",
                model_key,
                artifact_type,
                version,
                stored_checksum[:16],
                computed_checksum[:16],
            )
            return False

        logger.info("Artifact verified: %s/%s v%d ✓", model_key, artifact_type, version)
        return True

    def get_artifact_path(self, model_key: str, version: int, artifact_type: str) -> str | None:
        """Get the local path for a registered artifact."""
        manifest = self._load_manifest(model_key, version)
        return manifest.artifacts.get(artifact_type)

    def get_manifest(self, model_key: str, version: int) -> ArtifactManifest:
        """Get the full artifact manifest for a model version."""
        return self._load_manifest(model_key, version)

    def list_artifacts(self, model_key: str, version: int) -> dict[str, str]:
        """List all artifacts for a model version. Returns {type: path}."""
        manifest = self._load_manifest(model_key, version)
        return dict(manifest.artifacts)

    def _manifest_path(self, model_key: str, version: int) -> Path:
        """Get the manifest file path."""
        safe_key = model_key.replace("-", "_").replace(".", "_")
        return self._manifest_dir / f"{safe_key}_v{version}.json"

    def _load_manifest(self, model_key: str, version: int) -> ArtifactManifest:
        """Load or create a manifest."""
        path = self._manifest_path(model_key, version)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return ArtifactManifest.model_validate(data)
        return ArtifactManifest(model_key=model_key, version=version)

    def _save_manifest(self, manifest: ArtifactManifest) -> None:
        """Save a manifest to disk."""
        path = self._manifest_path(manifest.model_key, manifest.version)
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()
