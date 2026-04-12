"""
Centralized configuration management using Pydantic Settings.

All YAML config files are loaded, validated, and merged into typed dataclasses.
Environment variables override YAML values (prefix: MERCHFINE__).
Access via: `from core.config import get_settings`
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


# ── Helpers ───────────────────────────────────────────────────────────────


def _resolve_config_dir() -> Path:
    """Resolve the config directory from env or default to ./config relative to project root."""
    env_dir = os.getenv("MERCHFINE_CONFIG_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    # Walk up from this file to find the project root (where config/ lives)
    current = Path(__file__).resolve().parent.parent
    return current / "config"


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML config file from the config directory."""
    path = _resolve_config_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── Model Definitions ────────────────────────────────────────────────────


class ModelSpec(BaseModel):
    """Specification for a single supported model."""

    hf_id: str
    family: str
    tier: str  # primary, fallback, upgrade
    vram_required_gb: float
    ram_required_gb: float
    supports_tool_calling: bool = False
    prompt_format: str = "auto"
    context_window: int = 2048
    gguf_quant_methods: list[str] = Field(default_factory=lambda: ["q4_k_m"])
    ollama_name: str = ""
    description: str = ""


class HardwareTier(BaseModel):
    """Hardware tier definition."""

    label: str
    vram_gb: int
    eligible_models: list[str]
    inference_backend: str = "ollama"


class ModelsConfig(BaseModel):
    """All model configurations loaded from models.yaml."""

    defaults: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    hardware_tiers: dict[str, HardwareTier] = Field(default_factory=dict)
    aliases: dict[str, str | None] = Field(default_factory=dict)

    def get_model(self, model_key: str) -> ModelSpec:
        """Get a model spec by its key, raising if not found."""
        if model_key not in self.models:
            from core.exceptions import ModelNotFoundError

            raise ModelNotFoundError(
                f"Model '{model_key}' not found in config. Available: {list(self.models.keys())}"
            )
        return self.models[model_key]

    def get_primary_model(self) -> tuple[str, ModelSpec]:
        """Return (key, spec) of the primary-tier model."""
        for key, spec in self.models.items():
            if spec.tier == "primary":
                return key, spec
        raise ValueError("No model with tier='primary' found in config.")

    def get_fallback_models(self) -> list[tuple[str, ModelSpec]]:
        """Return all fallback-tier models."""
        return [(k, v) for k, v in self.models.items() if v.tier == "fallback"]

    def get_models_for_vram(self, available_vram_gb: float) -> list[tuple[str, ModelSpec]]:
        """Return all models that fit within the given VRAM budget."""
        return [(k, v) for k, v in self.models.items() if v.vram_required_gb <= available_vram_gb]


# ── Training Config ──────────────────────────────────────────────────────


class TrainingProfile(BaseModel):
    """Per-model training hyperparameters."""

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 512
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_grad_norm: float = 0.3


class SweepConfig(BaseModel):
    """Optuna hyperparameter sweep settings."""

    n_trials: int = 10
    timeout_seconds: int = 7200
    search_space: dict[str, list[Any]] = Field(default_factory=dict)
    objective_metric: str = "eval_loss"
    direction: str = "minimize"


class TrainingConfig(BaseModel):
    """Full training configuration."""

    defaults: dict[str, Any] = Field(default_factory=dict)
    profiles: dict[str, TrainingProfile] = Field(default_factory=dict)
    sweep: SweepConfig = Field(default_factory=SweepConfig)

    def get_profile(self, model_key: str) -> TrainingProfile:
        """Get training profile for a model, falling back to defaults."""
        if model_key in self.profiles:
            return self.profiles[model_key]
        # Build a profile from defaults
        return TrainingProfile(**{k: v for k, v in self.defaults.items() if k in TrainingProfile.model_fields})

    def get_merged_profile(self, model_key: str) -> dict[str, Any]:
        """Get a dict merging defaults + per-model profile for SFTTrainer config."""
        base = dict(self.defaults)
        if model_key in self.profiles:
            base.update(self.profiles[model_key].model_dump(exclude_unset=False))
        return base


# ── RAG Config ────────────────────────────────────────────────────────────


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    embed_batch_size: int = 64
    cache_dir: str = "./outputs/embeddings_cache"


class VectorStoreConfig(BaseModel):
    provider: str = "chroma"
    persist_dir: str = "./data/vector_store"
    collection_name: str = "merchfine_kb"
    distance_metric: str = "cosine"


class ChunkingConfig(BaseModel):
    parent_chunk_size: int = 2048
    child_chunk_size: int = 512
    leaf_chunk_size: int = 128
    chunk_overlap: int = 20


class RetrievalConfig(BaseModel):
    top_k: int = 12
    final_top_k: int = 5
    auto_merge_threshold: float = 0.6
    hybrid_alpha: float = 0.5
    enable_bm25: bool = True
    enable_reranker: bool = False


class RerankerConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = 16


class KnowledgeBaseConfig(BaseModel):
    source_dir: str = "./data/knowledge_base"
    supported_extensions: list[str] = Field(default_factory=lambda: [".txt", ".md", ".csv", ".pdf"])
    auto_rebuild_on_change: bool = True
    rebuild_eval_required: bool = True


class RAGConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)


# ── Guardrails Config ────────────────────────────────────────────────────


class InputSanitizerConfig(BaseModel):
    enabled: bool = True
    max_query_length: int = 2000
    max_query_tokens: int = 500
    injection_patterns: list[str] = Field(default_factory=list)
    pii_detection: dict[str, Any] = Field(default_factory=dict)


class FormatValidatorConfig(BaseModel):
    enabled: bool = True
    max_retries: int = 1
    retry_with_format_instruction: bool = True
    strict_json_mode: bool = False


class ProvenanceGuardConfig(BaseModel):
    enabled: bool = True
    cosine_threshold: float = 0.70
    flag_action: str = "log"
    embedding_model: str | None = None


class ConsistencyCheckerConfig(BaseModel):
    enabled: bool = False
    num_samples: int = 2
    agreement_threshold: float = 0.80
    temperature_for_samples: float = 0.7
    judge_model: str | None = None


class GuardrailsConfig(BaseModel):
    mode: str = "audit"  # "strict" or "audit"
    input_sanitizer: InputSanitizerConfig = Field(default_factory=InputSanitizerConfig)
    format_validator: FormatValidatorConfig = Field(default_factory=FormatValidatorConfig)
    provenance_guard: ProvenanceGuardConfig = Field(default_factory=ProvenanceGuardConfig)
    consistency_checker: ConsistencyCheckerConfig = Field(default_factory=ConsistencyCheckerConfig)


# ── Eval Config ──────────────────────────────────────────────────────────


class QualityGate(BaseModel):
    metric: str
    min_threshold: float | None = None
    max_threshold: float | None = None
    gate_type: str = "hard"  # hard, soft, info
    description: str = ""


class JudgeRubricItem(BaseModel):
    weight: float
    description: str


class JudgeConfig(BaseModel):
    enabled: bool = True
    rubric: dict[str, JudgeRubricItem] = Field(default_factory=dict)
    score_range: list[int] = Field(default_factory=lambda: [1, 5])
    min_passing_score: float = 3.5


class EvalConfig(BaseModel):
    gates: dict[str, QualityGate] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    triggers: list[dict[str, Any]] = Field(default_factory=list)

    def get_hard_gates(self) -> dict[str, QualityGate]:
        return {k: v for k, v in self.gates.items() if v.gate_type == "hard"}

    def get_soft_gates(self) -> dict[str, QualityGate]:
        return {k: v for k, v in self.gates.items() if v.gate_type == "soft"}


# ── Inference Config ─────────────────────────────────────────────────────


class InferenceConfig(BaseModel):
    """Derived from litellm_config.yaml + environment."""

    ollama_host: str = "http://127.0.0.1:11434"
    litellm_host: str = "http://127.0.0.1:4000"
    litellm_master_key: str = "sk-merchfine-local"
    default_temperature: float = 0.05
    default_max_tokens: int = 1024
    request_timeout: int = 90
    fallback_enabled: bool = True
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown: int = 30
    cache_enabled: bool = True
    cache_ttl: int = 3600


# ── Master Settings ──────────────────────────────────────────────────────


class Settings(BaseSettings):
    """
    Root settings object. Aggregates all layer configs.

    Usage:
        from core.config import get_settings
        settings = get_settings()
        primary_model = settings.models.get_primary_model()
    """

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    config_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "config")
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    outputs_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs")

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"

    # Service URIs (overridable via env)
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_name: str = "merchfine"
    ollama_host: str = "http://127.0.0.1:11434"
    litellm_host: str = "http://127.0.0.1:4000"
    redis_url: str = "redis://127.0.0.1:6379/0"
    langfuse_host: str = "http://127.0.0.1:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Layer configs (populated from YAML in model_post_init)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    model_config = {
        "env_prefix": "MERCHFINE_",
        "env_nested_delimiter": "__",
        "case_sensitive": False,
        "extra": "ignore",
    }

    def model_post_init(self, __context: Any) -> None:
        """Load YAML configs after environment parsing."""
        config_dir = _resolve_config_dir()
        if not config_dir.exists():
            return  # Allow running without config dir for tests

        yaml_map: dict[str, tuple[str, type[BaseModel]]] = {
            "models.yaml": ("models", ModelsConfig),
            "training.yaml": ("training", TrainingConfig),
            "rag.yaml": ("rag", RAGConfig),
            "guardrails.yaml": ("guardrails", GuardrailsConfig),
            "eval_thresholds.yaml": ("evaluation", EvalConfig),
        }

        for filename, (attr, model_cls) in yaml_map.items():
            filepath = config_dir / filename
            if filepath.exists():
                raw = _load_yaml(filename)
                try:
                    parsed = model_cls.model_validate(raw)
                    object.__setattr__(self, attr, parsed)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to parse {filename}: {e}", stacklevel=2)

        # Build InferenceConfig from env + litellm yaml
        litellm_path = config_dir / "litellm_config.yaml"
        if litellm_path.exists():
            litellm_raw = _load_yaml("litellm_config.yaml")
            router = litellm_raw.get("router_settings", {})
            object.__setattr__(
                self,
                "inference",
                InferenceConfig(
                    ollama_host=self.ollama_host,
                    litellm_host=self.litellm_host,
                    request_timeout=router.get("timeout", 90),
                    circuit_breaker_threshold=router.get("allowed_fails", 3),
                    circuit_breaker_cooldown=router.get("cooldown_time", 30),
                    cache_enabled=litellm_raw.get("litellm_settings", {}).get("cache", True),
                    cache_ttl=litellm_raw.get("litellm_settings", {}).get("cache_params", {}).get("ttl", 3600),
                ),
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton accessor for the global Settings object.

    First call loads all YAML configs + environment. Subsequent calls
    return the cached instance. Call `get_settings.cache_clear()` to reload.
    """
    from dotenv import load_dotenv

    load_dotenv()
    return Settings()


def reload_settings() -> Settings:
    """Force reload all settings (useful after config changes)."""
    get_settings.cache_clear()
    return get_settings()
