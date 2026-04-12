"""
Data Pipeline — validate, deduplicate, split, and version training datasets.

Reads raw JSONL/JSON files, validates every sample against Pydantic schemas,
enforces refusal ratio, deduplicates by checksum, splits train/eval,
and writes timestamped, versioned output files.

Usage:
    from data.pipeline import DataPipeline
    pipeline = DataPipeline()
    manifest = pipeline.process("data/raw/samples_v1.jsonl")
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from core.config import get_settings
from core.exceptions import DataPipelineError, DataValidationError, RefusalRatioError
from data.schema import DatasetManifest, SampleCategory, TrainingSample

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end data processing pipeline.

    Stages:
    1. Ingest — read raw JSONL/JSON files
    2. Validate — parse each sample through Pydantic schema
    3. Deduplicate — remove samples with duplicate checksums
    4. Enforce ratios — verify refusal_ratio >= min threshold
    5. Split — stratified train/eval split
    6. Write — timestamped JSONL output with manifest
    """

    def __init__(self):
        self.settings = get_settings()
        self._training_config = self.settings.training.defaults

    def process(
        self,
        input_path: str | Path,
        *,
        version: str = "1.0",
        seed: int | None = None,
    ) -> DatasetManifest:
        """
        Process a raw data file into validated, split training data.

        Args:
            input_path: Path to raw JSONL or JSON file.
            version: Version string for this dataset.
            seed: Random seed for reproducible splits.

        Returns:
            DatasetManifest with statistics and output paths.

        Raises:
            DataPipelineError: If the input file is invalid.
            RefusalRatioError: If refusal ratio is below minimum.
        """
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise DataPipelineError(f"Input file not found: {input_path}")

        if seed is None:
            seed = self.settings.training.defaults.get("seed", 42)
        random.seed(seed)

        logger.info("Processing data file: %s (version=%s)", input_path, version)

        # Stage 1: Ingest
        raw_samples = self._ingest(input_path)
        logger.info("Ingested %d raw samples.", len(raw_samples))

        # Stage 2: Validate
        valid_samples, errors = self._validate(raw_samples)
        if errors:
            logger.warning(
                "Validation errors: %d/%d samples rejected.",
                len(errors),
                len(raw_samples),
            )
            for err in errors[:5]:
                logger.warning("  Sample error: %s", err)

        if not valid_samples:
            raise DataValidationError("No valid samples after validation.")

        logger.info("Validated: %d samples passed.", len(valid_samples))

        # Stage 3: Deduplicate
        unique_samples, dup_count = self._deduplicate(valid_samples)
        logger.info(
            "Deduplicated: %d unique (%d duplicates removed).",
            len(unique_samples),
            dup_count,
        )

        # Stage 4: Enforce ratios
        min_ratio = self._training_config.get("refusal_ratio_min", 0.20)
        category_dist = self._compute_distribution(unique_samples)
        refusal_count = category_dist.get(SampleCategory.REFUSAL.value, 0)
        actual_ratio = refusal_count / len(unique_samples) if unique_samples else 0

        if actual_ratio < min_ratio:
            raise RefusalRatioError(
                f"Refusal ratio {actual_ratio:.1%} is below minimum {min_ratio:.0%}. "
                f"Need at least {int(len(unique_samples) * min_ratio)} refusal samples, "
                f"got {refusal_count}.",
                details={"actual": actual_ratio, "required": min_ratio},
            )

        # Stage 5: Split
        eval_ratio = self._training_config.get("eval_split_ratio", 0.1)
        train_samples, eval_samples = self._stratified_split(unique_samples, eval_ratio=eval_ratio)
        logger.info(
            "Split: %d train / %d eval (%.0f%% eval).",
            len(train_samples),
            len(eval_samples),
            eval_ratio * 100,
        )

        # Stage 6: Write output
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = self.settings.data_dir / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / f"train_v{version}_{timestamp}.jsonl"
        eval_path = output_dir / f"eval_v{version}_{timestamp}.jsonl"

        self._write_jsonl(train_samples, train_path)
        self._write_jsonl(eval_samples, eval_path)

        # Compute file hash
        file_hash = self._compute_file_hash(train_path)

        # Build manifest
        manifest = DatasetManifest(
            version=version,
            total_samples=len(unique_samples),
            category_distribution=category_dist,
            unique_checksums=len(unique_samples),
            duplicate_count=dup_count,
            file_path=str(train_path),
            sha256=file_hash,
        )

        # Write manifest
        manifest_path = output_dir / f"manifest_v{version}_{timestamp}.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        logger.info("Pipeline complete. Manifest: %s", manifest_path)
        logger.info("\n%s", manifest.summary())

        return manifest

    def _ingest(self, path: Path) -> list[dict[str, Any]]:
        """Read samples from JSONL or JSON file."""
        text = path.read_text(encoding="utf-8").strip()

        if path.suffix == ".jsonl":
            samples = []
            for i, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSON at line %d: %s", i, e)
            return samples

        elif path.suffix == ".json":
            data = json.loads(text)
            if isinstance(data, list):
                return data
            raise DataPipelineError(f"JSON file must contain an array. Got: {type(data).__name__}")

        else:
            raise DataPipelineError(f"Unsupported file format: {path.suffix}. Use .jsonl or .json")

    def _validate(self, raw_samples: list[dict[str, Any]]) -> tuple[list[TrainingSample], list[str]]:
        """Validate each sample against the Pydantic schema."""
        valid: list[TrainingSample] = []
        errors: list[str] = []

        for i, raw in enumerate(raw_samples):
            try:
                sample = TrainingSample.model_validate(raw)
                valid.append(sample)
            except ValidationError as e:
                errors.append(f"Sample {i}: {e.error_count()} validation errors — {e.errors()[0]['msg']}")

        return valid, errors

    def _deduplicate(self, samples: list[TrainingSample]) -> tuple[list[TrainingSample], int]:
        """Remove samples with duplicate checksums. First occurrence wins."""
        seen: set[str] = set()
        unique: list[TrainingSample] = []

        for sample in samples:
            if sample.checksum not in seen:
                seen.add(sample.checksum)
                unique.append(sample)

        return unique, len(samples) - len(unique)

    def _compute_distribution(self, samples: list[TrainingSample]) -> dict[str, int]:
        """Count samples per category."""
        dist: dict[str, int] = {}
        for sample in samples:
            cat = sample.category.value
            dist[cat] = dist.get(cat, 0) + 1
        return dist

    def _stratified_split(
        self,
        samples: list[TrainingSample],
        eval_ratio: float = 0.1,
    ) -> tuple[list[TrainingSample], list[TrainingSample]]:
        """
        Stratified train/eval split preserving category distribution.

        Each category is split independently so eval set has representative
        samples from every category.
        """
        # Group by category
        groups: dict[str, list[TrainingSample]] = {}
        for sample in samples:
            cat = sample.category.value
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(sample)

        train: list[TrainingSample] = []
        eval_set: list[TrainingSample] = []

        for cat, group in groups.items():
            random.shuffle(group)
            n_eval = max(1, int(len(group) * eval_ratio))
            eval_set.extend(group[:n_eval])
            train.extend(group[n_eval:])

        random.shuffle(train)
        random.shuffle(eval_set)

        return train, eval_set

    def _write_jsonl(self, samples: list[TrainingSample], path: Path) -> None:
        """Write samples to a JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(sample.to_jsonl_line() + "\n")

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def validate_file(self, path: str | Path) -> dict[str, Any]:
        """
        Validate a data file without processing it.

        Returns a summary dict with sample count, valid count, error count,
        and category distribution.
        """
        path = Path(path).resolve()
        raw = self._ingest(path)
        valid, errors = self._validate(raw)
        dist = self._compute_distribution(valid)

        return {
            "file": str(path),
            "total_raw": len(raw),
            "valid": len(valid),
            "errors": len(errors),
            "category_distribution": dist,
            "error_details": errors[:10],
        }
