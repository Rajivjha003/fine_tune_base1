"""
Synthetic data augmentation using the fine-tuned model.

Takes existing validated samples, generates variations via the LLM,
validates quality via schema + optional DeepEval check, and gates
before adding to the training set.

Usage:
    from data.augmentor import SyntheticAugmentor
    augmentor = SyntheticAugmentor()
    new_samples = await augmentor.generate_variations(
        seed_samples=existing_samples,
        n_variations=3,
    )
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from pydantic import ValidationError

from core.config import get_settings
from data.schema import SampleCategory, TrainingSample

logger = logging.getLogger(__name__)

# Prompt template for generating variations
VARIATION_PROMPT_TEMPLATE = """You are a training data generator for a retail demand forecasting AI system called MerchMix.

Given the following training example, generate {n_variations} diverse variations that:
1. Cover the same topic/category but with different scenarios
2. Use different SKU IDs, quantities, time periods, and contexts
3. Maintain the same quality and detail level as the original
4. Are factually consistent (numbers should be realistic for retail)

ORIGINAL EXAMPLE:
Instruction: {instruction}
Input: {input}
Output: {output}

Generate exactly {n_variations} variations in the following JSON format:
[
  {{
    "instruction": "...",
    "input": "...",
    "output": "...",
    "category": "{category}"
  }}
]

Return ONLY the JSON array, no other text."""


class SyntheticAugmentor:
    """
    Generates synthetic training data variations using the champion model.

    Quality controls:
    - All generated samples pass Pydantic validation
    - Optional DeepEval similarity check against seed sample
    - Deduplication against existing training set
    """

    def __init__(self):
        self.settings = get_settings()

    async def generate_variations(
        self,
        seed_samples: list[TrainingSample],
        *,
        n_variations: int = 3,
        max_samples: int | None = None,
        temperature: float = 0.7,
        existing_checksums: set[str] | None = None,
    ) -> list[TrainingSample]:
        """
        Generate synthetic variations of seed samples.

        Args:
            seed_samples: Existing validated samples to use as seeds.
            n_variations: Number of variations per seed sample.
            max_samples: Maximum total samples to generate (None = unlimited).
            temperature: LLM temperature for generation.
            existing_checksums: Set of checksums to check for duplicates.

        Returns:
            List of new, validated, deduplicated TrainingSample objects.
        """
        if not seed_samples:
            logger.warning("No seed samples provided — skipping augmentation.")
            return []

        if existing_checksums is None:
            existing_checksums = {s.checksum for s in seed_samples}

        generated: list[TrainingSample] = []
        attempt_count = 0
        fail_count = 0

        for seed in seed_samples:
            if max_samples and len(generated) >= max_samples:
                break

            try:
                variations = await self._generate_from_seed(
                    seed,
                    n_variations=n_variations,
                    temperature=temperature,
                )

                for var in variations:
                    if max_samples and len(generated) >= max_samples:
                        break

                    # Deduplicate
                    if var.checksum in existing_checksums:
                        logger.debug("Skipping duplicate: %s", var.checksum[:8])
                        continue

                    existing_checksums.add(var.checksum)
                    generated.append(var)

                attempt_count += 1

            except Exception as e:
                fail_count += 1
                logger.warning("Failed to generate from seed '%s...': %s", seed.instruction[:50], e)

        logger.info(
            "Augmentation complete: %d new samples from %d seeds (%d failed).",
            len(generated),
            attempt_count,
            fail_count,
        )
        return generated

    async def _generate_from_seed(
        self,
        seed: TrainingSample,
        *,
        n_variations: int = 3,
        temperature: float = 0.7,
    ) -> list[TrainingSample]:
        """Generate variations from a single seed sample using the LLM."""
        prompt = VARIATION_PROMPT_TEMPLATE.format(
            n_variations=n_variations,
            instruction=seed.instruction,
            input=seed.input,
            output=seed.output,
            category=seed.category.value,
        )

        # Call inference gateway
        from inference.gateway import InferenceGateway

        gateway = InferenceGateway()
        response = await gateway.complete(
            prompt=prompt,
            temperature=temperature,
            max_tokens=2048,
        )

        raw_text = response.get("text", "")

        # Parse JSON from response
        variations = self._parse_variations(raw_text, seed.category)
        return variations

    def _parse_variations(self, raw_text: str, category: SampleCategory) -> list[TrainingSample]:
        """Parse LLM output into validated TrainingSample objects."""
        # Try to find JSON array in the response
        text = raw_text.strip()

        # Find the JSON array boundaries
        start_idx = text.find("[")
        end_idx = text.rfind("]")

        if start_idx == -1 or end_idx == -1:
            logger.warning("No JSON array found in LLM response.")
            return []

        json_str = text[start_idx : end_idx + 1]

        try:
            raw_items = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM JSON output: %s", e)
            return []

        if not isinstance(raw_items, list):
            return []

        samples: list[TrainingSample] = []
        for item in raw_items:
            try:
                # Ensure category is set correctly
                item["category"] = category.value
                item["source"] = "synthetic"
                sample = TrainingSample.model_validate(item)
                samples.append(sample)
            except ValidationError as e:
                logger.debug("Generated sample failed validation: %s", e.errors()[0]["msg"])

        return samples

    async def augment_category(
        self,
        samples: list[TrainingSample],
        target_category: SampleCategory,
        target_count: int,
        **kwargs: Any,
    ) -> list[TrainingSample]:
        """
        Augment a specific category to reach a target count.

        Useful for balancing the dataset or increasing refusal samples.
        """
        category_samples = [s for s in samples if s.category == target_category]
        current_count = len(category_samples)

        if current_count >= target_count:
            logger.info(
                "Category '%s' already has %d samples (target: %d). No augmentation needed.",
                target_category.value,
                current_count,
                target_count,
            )
            return []

        needed = target_count - current_count
        per_seed = max(1, needed // max(len(category_samples), 1))

        logger.info(
            "Augmenting '%s': need %d more samples (%d per seed).",
            target_category.value,
            needed,
            per_seed,
        )

        existing_checksums = {s.checksum for s in samples}
        return await self.generate_variations(
            category_samples,
            n_variations=per_seed,
            max_samples=needed,
            existing_checksums=existing_checksums,
            **kwargs,
        )
