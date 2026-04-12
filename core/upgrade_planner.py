"""
Hardware-aware model tier recommendations.

Detects the current GPU/RAM profile and recommends which models
are eligible, what upgrade path is available, and generates
config diffs for tier transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""

    gpu_name: str = "Unknown"
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    cuda_available: bool = False
    cuda_version: str = ""


@dataclass
class UpgradeRecommendation:
    """Recommendation output from the upgrade planner."""

    current_tier: str
    current_tier_label: str
    eligible_models: list[str]
    recommended_primary: str
    recommended_fallback: str | None
    next_tier: str | None
    next_tier_label: str | None
    next_tier_unlocks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class UpgradePlanner:
    """
    Detects hardware and recommends model configurations.

    Reads hardware tiers from config/models.yaml and matches
    the current system to the appropriate tier.
    """

    def __init__(self):
        self.settings = get_settings()

    def detect_hardware(self) -> HardwareProfile:
        """Auto-detect GPU, VRAM, and system RAM."""
        profile = HardwareProfile()

        # GPU detection
        try:
            import torch

            if torch.cuda.is_available():
                profile.cuda_available = True
                profile.gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                profile.vram_gb = round(props.total_mem / (1024**3), 1)
                profile.cuda_version = torch.version.cuda or ""
        except ImportError:
            logger.warning("PyTorch not available for GPU detection.")

        # RAM detection
        try:
            import psutil

            profile.ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            # Fallback: read from OS
            try:
                import os

                if hasattr(os, "sysconf"):
                    pages = os.sysconf("SC_PHYS_PAGES")
                    page_size = os.sysconf("SC_PAGE_SIZE")
                    profile.ram_gb = round((pages * page_size) / (1024**3), 1)
            except Exception:
                pass

        return profile

    def recommend(self, profile: HardwareProfile | None = None) -> UpgradeRecommendation:
        """
        Generate model recommendations based on hardware profile.

        If no profile is provided, auto-detects current hardware.
        """
        if profile is None:
            profile = self.detect_hardware()

        # Determine current tier
        tiers = self.settings.models.hardware_tiers
        current_tier_key = None
        current_tier = None

        # Find the best matching tier (highest VRAM that fits)
        sorted_tiers = sorted(tiers.items(), key=lambda x: x[1].vram_gb)
        for tier_key, tier in sorted_tiers:
            if profile.vram_gb >= tier.vram_gb * 0.8:  # Allow 20% tolerance
                current_tier_key = tier_key
                current_tier = tier

        if current_tier is None:
            # Below minimum tier
            return UpgradeRecommendation(
                current_tier="none",
                current_tier_label=f"Detected: {profile.gpu_name} ({profile.vram_gb}GB)",
                eligible_models=[],
                recommended_primary="",
                recommended_fallback=None,
                next_tier=sorted_tiers[0][0] if sorted_tiers else None,
                next_tier_label=sorted_tiers[0][1].label if sorted_tiers else None,
                notes=["Insufficient VRAM for any supported model configuration."],
            )

        # Find eligible models
        eligible = self.settings.models.get_models_for_vram(profile.vram_gb)
        eligible_keys = [k for k, _ in eligible]

        # Determine primary and fallback
        primary_key, _ = self.settings.models.get_primary_model()
        recommended_primary = primary_key if primary_key in eligible_keys else (eligible_keys[0] if eligible_keys else "")
        fallback_models = self.settings.models.get_fallback_models()
        recommended_fallback = None
        for fk, _ in fallback_models:
            if fk in eligible_keys and fk != recommended_primary:
                recommended_fallback = fk
                break

        # Find next tier
        next_tier_key = None
        next_tier_label = None
        next_tier_unlocks: list[str] = []
        found_current = False
        for tier_key, tier in sorted_tiers:
            if found_current:
                next_tier_key = tier_key
                next_tier_label = tier.label
                next_tier_unlocks = [m for m in tier.eligible_models if m not in eligible_keys and m != "*"]
                break
            if tier_key == current_tier_key:
                found_current = True

        # Build notes
        notes: list[str] = []
        if profile.vram_gb < 8:
            notes.append("Consider upgrading to 8GB+ VRAM for full model support.")
        if not profile.cuda_available:
            notes.append("No CUDA GPU detected. Training and inference will be CPU-only (very slow).")
        if profile.ram_gb < 16:
            notes.append(f"System RAM ({profile.ram_gb}GB) is below recommended 16GB.")

        return UpgradeRecommendation(
            current_tier=current_tier_key or "unknown",
            current_tier_label=current_tier.label if current_tier else "Unknown",
            eligible_models=eligible_keys,
            recommended_primary=recommended_primary,
            recommended_fallback=recommended_fallback,
            next_tier=next_tier_key,
            next_tier_label=next_tier_label,
            next_tier_unlocks=next_tier_unlocks,
            notes=notes,
        )

    def print_recommendation(self, rec: UpgradeRecommendation | None = None) -> None:
        """Pretty-print the upgrade recommendation."""
        if rec is None:
            rec = self.recommend()

        print("═══ MerchFine Hardware Analysis ═══")
        print(f"  Current Tier:       {rec.current_tier} — {rec.current_tier_label}")
        print(f"  Eligible Models:    {', '.join(rec.eligible_models) or 'None'}")
        print(f"  Recommended Primary: {rec.recommended_primary or 'N/A'}")
        print(f"  Recommended Fallback: {rec.recommended_fallback or 'N/A'}")
        if rec.next_tier:
            print(f"  Next Upgrade:       {rec.next_tier} — {rec.next_tier_label}")
            if rec.next_tier_unlocks:
                print(f"    Unlocks:          {', '.join(rec.next_tier_unlocks)}")
        for note in rec.notes:
            print(f"  ⚠️  {note}")
        print("═══════════════════════════════════")
