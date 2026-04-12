"""Quick validation script for MerchFine config loading."""
import sys
sys.path.insert(0, ".")

from core.config import get_settings

s = get_settings()

print("=== MerchFine Config Validation ===")

# Models
pk, ps = s.models.get_primary_model()
print(f"Primary model: {pk} ({ps.hf_id})")
fallbacks = s.models.get_fallback_models()
print(f"Fallback models: {[k for k, _ in fallbacks]}")
print(f"Hardware tiers: {list(s.models.hardware_tiers.keys())}")

# Training
profile = s.training.get_profile("gemma-3-4b")
print(f"Training (gemma-3-4b): r={profile.lora_r}, alpha={profile.lora_alpha}, lr={profile.learning_rate}")
print(f"Sweep trials: {s.training.sweep.n_trials}")

# RAG
print(f"RAG embedding: {s.rag.embedding.model_name}")
print(f"RAG chunks: parent={s.rag.chunking.parent_chunk_size}, child={s.rag.chunking.child_chunk_size}, leaf={s.rag.chunking.leaf_chunk_size}")

# Guardrails
print(f"Guardrails mode: {s.guardrails.mode}")
print(f"Injection patterns: {len(s.guardrails.input_sanitizer.injection_patterns)}")

# Eval
hard_gates = s.evaluation.get_hard_gates()
print(f"Eval hard gates: {list(hard_gates.keys())}")

# Inference
print(f"Ollama host: {s.ollama_host}")
print(f"MLflow URI: {s.mlflow_tracking_uri}")
print(f"Circuit breaker threshold: {s.inference.circuit_breaker_threshold}")

print("\n=== ALL CONFIGS VALID ===")
