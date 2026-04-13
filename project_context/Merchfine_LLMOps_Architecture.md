# MerchMix LLMOps Architecture: Production Blueprint 2026

> **Scope:** End-to-end, local-first, hardware-aware AI system for MerchMix retail forecasting.
> **Constraint Profile:** RTX 4070 Laptop 8GB VRAM · 16GB RAM · Windows · CUDA 12.5 · Fully offline-capable.
> **Primary Model:** Gemma 3 4B-IT (QLoRA) · **Fallback:** Qwen2.5-3B-Instruct

---

## 1. Architecture Philosophy

This system is built on four principles that separate production-grade LLMOps from amateur setups[cite:62][cite:63]:

1. **Separation of concerns** — every subsystem (data, training, registry, serving, evaluation, observability) is independently replaceable
2. **Contract-driven interfaces** — every layer communicates via OpenAI-compatible HTTP, never through direct Python imports across layers[cite:37][cite:43]
3. **Offline-first, cloud-ready** — all inference, evaluation, and observability run locally with zero external dependencies; cloud burst is a config change, not a rewrite
4. **Continuous quality gates** — the system evaluates itself after every model swap, fine-tune, or RAG update before promotion[cite:65][cite:71]

The dominant failure mode of LLM systems is not model quality — it is **silent degradation**: hallucination drift, retrieval regression, and prompt regression that return HTTP 200 while delivering wrong answers[cite:67][cite:70]. This architecture is designed to catch every failure mode before users see it.

---

## 2. Complete Tech Stack

### Core ML Stack
| Layer | Tool | Version | Role |
|---|---|---|---|
| Fine-tuning framework | **Unsloth** | Latest | 2–5× faster QLoRA, 70% less VRAM[cite:28] |
| PEFT/LoRA | **PEFT + TRL** | Latest | QLoRA adapter management |
| Base model (primary) | **Gemma 3 4B-IT** | unsloth/gemma-3-4b-it | Best perf/VRAM at 8GB[cite:32][cite:35] |
| Base model (fallback) | **Qwen2.5-3B-Instruct** | unsloth/Qwen2.5-3B-Instruct | Hot-swap fallback[cite:18] |
| Quantization | **bitsandbytes** | Latest | 4-bit NF4, 8-bit AdamW |
| Quantized inference | **llama.cpp / GGUF** | Latest | Q4_K_M export for serving |

### Inference & Serving Stack
| Layer | Tool | Role |
|---|---|---|
| Local inference server | **Ollama** | OpenAI-compat REST, VRAM auto-management[cite:43] |
| AI gateway & routing | **LiteLLM** | Unified proxy, semantic cache, fallback routing[cite:63][cite:83] |
| Semantic cache | **Redis + RedisVL** | Vector similarity cache, avoids redundant inference[cite:83] |
| Upgrade path | **vLLM** | Drop-in swap when RTX 4090+ available[cite:37] |

### RAG & Orchestration Stack
| Layer | Tool | Role |
|---|---|---|
| RAG framework | **LlamaIndex** | Hierarchical auto-merging retrieval[cite:51][cite:54] |
| Vector store | **ChromaDB** | Local persistent, no server needed |
| Embeddings | **BAAI/bge-small-en-v1.5** | 130MB, CPU-only, fully offline |
| Agentic orchestration | **LangGraph** | Stateful multi-step agent workflows[cite:51] |
| Prompt management | **Langfuse** (self-hosted) | Prompt versioning, A/B testing[cite:70][cite:73] |

### Experiment Tracking & Registry
| Layer | Tool | Role |
|---|---|---|
| Experiment tracking | **MLflow 3.9+** | Run tracking, artifact lineage[cite:50][cite:56] |
| Model registry | **MLflow Model Registry** | Alias-based promotion (champion/challenger)[cite:56] |
| Hyperparameter search | **Optuna** | Automated LoRA rank/alpha sweeps |
| Dataset versioning | **DVC** | Git-compatible data + model version control |

### Evaluation Stack
| Layer | Tool | Role |
|---|---|---|
| LLM unit testing | **DeepEval** | Pytest-style, CI/CD integration[cite:74][cite:75][cite:81] |
| RAG evaluation | **RAGAS** | Faithfulness, context precision, answer relevancy[cite:75][cite:87] |
| Custom eval judge | **Local Gemma 3 4B** | LLM-as-judge for domain-specific scoring |
| Regression testing | **DeepEval + pytest** | Auto-run on every model swap or RAG rebuild |

### Safety & Guardrails Stack
| Layer | Tool | Role |
|---|---|---|
| Output validation | **Guardrails AI** | Provenance-based hallucination detection[cite:82] |
| Consistency check | **NeMo Guardrails** | Self-check hallucination via multi-sample consistency[cite:79][cite:85] |
| Input sanitization | Custom middleware | Prompt injection screening, PII detection[cite:63] |
| Format enforcement | **Pydantic v2** | Structured output validation |

### Observability Stack
| Layer | Tool | Role |
|---|---|---|
| Trace & observability | **Langfuse** (self-hosted) | Full request traces, cost, latency, token usage[cite:70][cite:73] |
| Structured logging | **structlog + Loguru** | JSON logs, async, context-aware |
| Metrics | **Prometheus + Grafana** | Latency p50/p95/p99, error rates, VRAM |
| Alerting | **Grafana Alerts** | VRAM threshold, latency spike, eval score drop |

---

## 3. Complete Folder Structure

```
merchfine/
│
├── 📁 config/                          # System-wide configuration
│   ├── models.yaml                     # Model registry (all supported models + paths)
│   ├── training.yaml                   # Training hyperparameters per model
│   ├── rag.yaml                        # RAG chunk sizes, retrieval top-k, thresholds
│   ├── guardrails.yaml                 # Guardrail rules, threshold scores
│   ├── eval_thresholds.yaml            # Pass/fail gates per metric
│   └── litellm_config.yaml             # LiteLLM routing, caching, fallback config
│
├── 📁 data/                            # Data pipeline (Layer 1)
│   ├── raw/                            # Source CSVs, manual exports from MerchMix
│   ├── processed/                      # Cleaned, validated JSONL
│   │   ├── train_v1.0_<timestamp>.jsonl
│   │   └── eval_v1.0_<timestamp>.jsonl
│   ├── synthetic/                      # LLM-augmented training data
│   ├── knowledge_base/                 # RAG documents
│   │   ├── sku_catalog.txt
│   │   ├── mio_rules.md
│   │   ├── seasonal_calendar.md
│   │   └── sales_history.csv
│   ├── dvc.yaml                        # DVC pipeline definition
│   ├── pipeline.py                     # DataPipeline class (validate, format, version)
│   ├── augmentor.py                    # Synthetic data generator (LLM-assisted)
│   └── schema.py                       # Pydantic schemas for all sample types
│
├── 📁 training/                        # Fine-tuning engine (Layer 2)
│   ├── finetune.py                     # Main QLoRA trainer (Unsloth + SFTTrainer)
│   ├── prompt_templates.py             # Gemma 3 / Qwen / Phi format templates
│   ├── callbacks.py                    # MLflow logging callback, VRAM monitor
│   ├── sweep.py                        # Optuna hyperparameter sweep
│   └── export.py                       # GGUF export + Ollama Modelfile generator
│
├── 📁 registry/                        # Model registry & lifecycle (Layer 3)
│   ├── model_manager.py                # MLflow registry: promote, rollback, list
│   ├── experiment_tracker.py           # MLflow run creation, metric/param logging
│   ├── artifact_store.py               # GGUF path tracking, checksum validation
│   └── schemas.py                      # ModelVersion, RunMetadata dataclasses
│
├── 📁 inference/                       # Inference gateway (Layer 4)
│   ├── gateway.py                      # LiteLLM proxy wrapper (OpenAI-compat)
│   ├── ollama_manager.py               # Ollama: register, swap, health check
│   ├── cache.py                        # Redis semantic cache config
│   ├── fallback.py                     # Circuit breaker + fallback chain logic[cite:44]
│   └── Modelfile.template              # Ollama Modelfile with system prompt + params
│
├── 📁 rag/                             # RAG pipeline (Layer 5)
│   ├── pipeline.py                     # LlamaIndex hierarchical auto-merging RAG
│   ├── indexer.py                      # Document ingestion + ChromaDB index builder
│   ├── retriever.py                    # Hybrid retriever (dense + BM25 sparse)
│   ├── reranker.py                     # Cross-encoder reranker (optional, CPU)
│   └── store/                          # Persisted ChromaDB vector store
│
├── 📁 agents/                          # Agentic orchestration (Layer 6)
│   ├── merch_agent.py                  # LangGraph ReAct agent with domain tools
│   ├── tools/
│   │   ├── forecast.py                 # Demand forecasting tool
│   │   ├── inventory.py                # Reorder / stock check tool
│   │   ├── mio_planner.py              # MIO plan generation tool
│   │   └── data_lookup.py              # SKU catalog lookup tool
│   ├── memory.py                       # LangGraph persistent state / conversation memory
│   └── graph.py                        # LangGraph state machine definition
│
├── 📁 guardrails/                      # Safety & output validation (Layer 7)
│   ├── output_guard.py                 # Guardrails AI provenance validator[cite:82]
│   ├── hallucination_check.py          # NeMo multi-sample consistency check[cite:79]
│   ├── input_sanitizer.py              # Prompt injection + PII detection
│   └── format_validator.py             # Pydantic output schema enforcement
│
├── 📁 evaluation/                      # Continuous evaluation (Layer 8)
│   ├── eval_runner.py                  # DeepEval test suite runner
│   ├── rag_eval.py                     # RAGAS metrics: faithfulness, precision, recall[cite:75]
│   ├── test_cases/
│   │   ├── forecasting_cases.json      # Ground-truth QA pairs for forecasting domain
│   │   ├── mio_cases.json              # MIO planning test cases
│   │   └── refusal_cases.json          # Edge case / refusal ground truth
│   ├── quality_gate.py                 # Pass/fail enforcer — blocks promotion if eval fails
│   └── regression_suite.py             # Full regression run triggered on any system change
│
├── 📁 observability/                   # Monitoring & tracing (Layer 9)
│   ├── langfuse_client.py              # Langfuse trace instrumentation wrapper[cite:73]
│   ├── metrics_collector.py            # Prometheus metrics: latency, tokens, VRAM
│   ├── log_config.py                   # structlog + Loguru structured JSON logging
│   ├── dashboards/
│   │   ├── grafana_llm_health.json     # Grafana dashboard: latency, errors, cache hits
│   │   └── grafana_eval_trends.json    # Grafana dashboard: eval scores over time
│   └── alerts.yaml                     # Alert rules: VRAM > 7GB, p95 > 3s, faith < 0.8
│
├── 📁 core/                            # Master controller (Layer 10)
│   ├── model_switcher.py               # One-command full-stack model hot-swap
│   ├── system_init.py                  # Full system bootstrap and health check
│   └── upgrade_planner.py             # Hardware-aware model tier recommendations
│
├── 📁 api/                             # Application interface
│   ├── app.py                          # FastAPI server (REST endpoints)
│   ├── routes/
│   │   ├── forecast.py                 # POST /forecast
│   │   ├── mio.py                      # POST /mio-plan
│   │   ├── chat.py                     # POST /chat (agentic)
│   │   └── admin.py                    # POST /swap-model, GET /health, GET /metrics
│   └── middleware.py                   # Request logging, auth, rate limiting
│
├── 📁 outputs/                         # Model artifacts
│   ├── lora_gemma-3-4b/                # LoRA adapter weights
│   ├── lora_qwen2.5-3b/
│   ├── gguf_gemma-3-4b/
│   │   └── model-q4_k_m.gguf          # Quantized inference model
│   └── gguf_qwen2.5-3b/
│
├── 📁 mlruns/                          # MLflow experiment tracking (auto-generated)
├── 📁 tests/                           # Unit + integration tests
│   ├── test_data_pipeline.py
│   ├── test_inference_gateway.py
│   ├── test_rag_pipeline.py
│   └── test_guardrails.py
│
├── .dvcignore
├── .env.example                        # LANGFUSE_HOST, REDIS_URL, MLFLOW_URI
├── docker-compose.yml                  # Langfuse + Prometheus + Grafana + Redis
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 4. The 10-Layer System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 10 │ MASTER CONTROLLER — model_switcher · system_init        │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 9  │ OBSERVABILITY — Langfuse traces · Prometheus · Grafana  │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 8  │ EVALUATION — DeepEval · RAGAS · Quality Gate CI         │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 7  │ GUARDRAILS — Guardrails AI · NeMo · Pydantic guard      │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 6  │ AGENTS — LangGraph · Tool calls · Persistent memory     │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 5  │ RAG — LlamaIndex hierarchical · ChromaDB · BGE embed    │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 4  │ INFERENCE GATEWAY — LiteLLM · Ollama · Redis Sem. Cache │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 3  │ MODEL REGISTRY — MLflow 3.9+ · DVC · Optuna sweeps      │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 2  │ FINE-TUNING ENGINE — Unsloth · QLoRA · GGUF export      │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 1  │ DATA PIPELINE — schema validation · versioning · DVC    │
└──────────────────────────────────────────────────────────────────────┘
                             │ ALL LAYERS │
                    ┌────────┴────────────┴─────────┐
                    │  FastAPI Application Layer     │
                    │  POST /forecast · /mio · /chat │
                    └────────────────────────────────┘
```

---

## 5. Data Pipeline (Layer 1) — Implementation Plan

**Goal:** Every training sample is validated, checksummed, versioned, and guaranteed to contain zero "hallucination seeds" (outputs that reference facts not present in the input).

**Key implementation decisions:**

- Use Pydantic v2 for schema enforcement at ingestion — invalid samples are rejected before they reach the trainer, never silently ignored
- Every sample gets a `checksum` (MD5 of instruction+input+output) for deduplication and data lineage
- `refusal_ratio` enforcement: 20% of samples must be refusal/uncertainty cases — this is the single highest-impact anti-hallucination technique for domain models[cite:23][cite:25]
- DVC tracks the full `raw → processed → train/eval split` pipeline with SHA hashes, so any training run is fully reproducible from a single `dvc repro` command
- Synthetic data augmentation via the fine-tuned model itself (once v1 is trained): generate variations of existing samples, validate against ground truth, gate quality with DeepEval before adding to the training set

**Recommended dataset composition for MerchMix v1:**

| Sample Type | Count | % of Total |
|---|---|---|
| Demand forecasting (with full input context) | 200 | 40% |
| Reorder point / inventory checks | 75 | 15% |
| MIO plan generation | 75 | 15% |
| Sell-through analysis | 50 | 10% |
| Refusal / insufficient data cases | 100 | 20% |
| **Total** | **500** | **100%** |

---

## 6. Fine-Tuning Engine (Layer 2) — Implementation Plan

**Core config: Gemma 3 4B-IT + QLoRA r=16 + Unsloth on RTX 4070 Laptop 8GB**[cite:32]

| Parameter | Value | Rationale |
|---|---|---|
| `load_in_4bit` | `True` | Weights ~2.8GB, leaves ~5GB for grad/optim[cite:35] |
| `lora_r` | `16` | Balance between capacity and VRAM[cite:18] |
| `lora_alpha` | `16` | Keep equal to r for training stability |
| `lora_dropout` | `0.05` | Light regularization, small dataset |
| `use_gradient_checkpointing` | `"unsloth"` | 30% additional memory savings[cite:18] |
| `per_device_train_batch_size` | `1` | Hard limit at 8GB |
| `gradient_accumulation_steps` | `8` | Effective batch size = 8 |
| `max_seq_length` | `512` | Keeps activations low; raise to 1024 post-upgrade |
| `learning_rate` | `2e-4` | Standard QLoRA LR |
| `lr_scheduler_type` | `cosine` | Smooth decay, prevents late-epoch spikes |
| `optim` | `adamw_8bit` | Halves optimizer memory vs float32 |
| `num_train_epochs` | `3` | For ~500 samples; monitor eval loss |
| **Estimated train time** | **20–40 min** | On RTX 4070 Laptop with Unsloth[cite:5] |

**Prompt template — Gemma 3 chat format** (critical: wrong format = degraded performance):

```
<start_of_turn>user
{instruction}

Context: {input}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>{EOS_TOKEN}
```

**Training callbacks to implement:**
1. `VRAMMonitorCallback` — logs `torch.cuda.max_memory_reserved()` every 10 steps to MLflow; alerts if > 7.5GB
2. `EvalLossCallback` — runs on held-out 10% eval split every epoch; early stops if eval loss diverges > 0.3 from train loss (overfitting signal)
3. `MLflowLoggingCallback` — logs all metrics, params, and artifacts automatically per run

**GGUF export pipeline (post-training):**
- Merge LoRA → base model → export as `q4_k_m.gguf` (best quality/size ratio)
- Generate Ollama `Modelfile` with system prompt + inference parameters
- Register GGUF path + checksum in MLflow artifact store
- Trigger evaluation suite before any model is registered as `@champion`

---

## 7. Model Registry & Experiment Tracking (Layer 3) — Implementation Plan

**MLflow 3.9+ alias system** replaces deprecated `Staging/Production` stages[cite:50]:

```
Model: MerchFine-gemma-3-4b
  ├── Version 1  ←── @champion (current production)
  ├── Version 2  ←── @challenger (under evaluation)
  └── Version 3  ←── @archived
```

**Promotion workflow:**
1. New fine-tune completes → registered as `@challenger`
2. Evaluation suite runs against `@challenger` — must pass all quality gates
3. If gates pass → `switcher.promote(challenger → champion)`; previous champion → `@archived`
4. If gates fail → `@challenger` is tagged `eval_failed=true`; human review required

**What every MLflow run tracks:**
- Parameters: model_id, lora_r, lora_alpha, lr, batch_size, dataset_version, num_samples
- Metrics: train_loss, eval_loss, train_time_secs, vram_peak_gb, tokens_per_sec
- Artifacts: LoRA adapter path, GGUF path, training config YAML, eval report JSON
- Tags: model_tier (primary/fallback/upgrade), hardware_profile, git_commit_hash

**DVC handles data and model versioning** — `dvc push/pull` ensures any teammate (or future cloud instance) can reproduce exact training conditions from git commit alone.

---

## 8. Inference Gateway (Layer 4) — Implementation Plan

**Stack: Ollama (local) → LiteLLM proxy → Application**[cite:43][cite:63]

The critical design choice: every application call hits **LiteLLM's OpenAI-compatible endpoint**, never Ollama directly. LiteLLM adds:

- **Semantic cache** (Redis + RedisVL): identical or near-identical queries (~0.85 cosine similarity) return cached responses — eliminates redundant inference for repeated SKU lookups[cite:83]
- **Fallback routing**: if Gemma 3 4B fails/times out, automatically routes to Qwen2.5-3B[cite:44]
- **Circuit breaker**: after 3 consecutive failures, temporarily disables primary model to prevent cascade[cite:44]
- **Request logging**: every call logged with token count, latency, model used — feeds Langfuse automatically

**Inference parameters (Ollama Modelfile):**
```
PARAMETER temperature 0.05       # Near-deterministic for forecasting
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.15    # Discourages looping/repetition
PARAMETER num_ctx 2048
PARAMETER num_gpu 99             # Use all available VRAM layers
```

**Model swap is a single API call to Ollama** — VRAM is automatically evicted and reloaded[cite:43]. The LiteLLM config file (`litellm_config.yaml`) is updated and LiteLLM is restarted. Zero application code changes required.

---

## 9. RAG Pipeline (Layer 5) — Implementation Plan

**Hierarchical Auto-Merging Retrieval** is the 2026 standard for production RAG — retrieves small chunks for precision, merges to parent context when multiple siblings match for coherence[cite:51][cite:54]:

```
Document → 2048-token parent nodes
              └── 512-token child nodes
                    └── 128-token leaf nodes (retrieved)
```

**Retrieval flow:**
1. Query encoded with BAAI/bge-small (CPU, offline)
2. Top-12 leaf nodes retrieved from ChromaDB (dense vector search)
3. AutoMergingRetriever checks: if ≥ 60% of a parent's children are in top-12, replace children with parent
4. BM25 sparse retrieval runs in parallel; results merged with RRF (Reciprocal Rank Fusion)
5. Optional: cross-encoder reranker re-scores top-12 → top-5 (CPU, adds ~200ms, improves precision significantly)
6. Final context passed to inference gateway

**RAG knowledge base maintenance:**
- `sales_history.csv` refreshed weekly via a scheduled Python script
- Any document update triggers automatic index rebuild + RAGAS evaluation
- Embedding model is pinned and versioned — changing embeddings requires full re-indexing

---

## 10. Guardrails (Layer 7) — Implementation Plan

**Three independent guardrail layers** — each catches different failure modes[cite:63][cite:82][cite:85]:

### Layer A: Input Sanitizer (pre-inference)
- Regex + embedding similarity screening for prompt injection patterns
- PII detection (names, phone numbers, emails) with configurable redact/block policy
- Query length and complexity bounds check

### Layer B: Output Format Validator (post-inference, immediate)
- Pydantic v2 schema validation — all responses must conform to `ForecastResponse`, `MIOPlanResponse`, or `RefusalResponse` models
- If parsing fails, request is retried once with explicit format instruction; on second failure, returns structured error

### Layer C: Hallucination Detector (post-inference, async)
- **Guardrails AI provenance validator**: embeds output sentences + source documents, flags sentences with cosine similarity < 0.7 to any retrieved context[cite:82]
- **NeMo self-check**: samples 2 additional responses from the model; uses LLM-as-judge to check consistency across all 3; flags if agreement < 80%[cite:79][cite:85]
- Flagged responses are logged to Langfuse with `hallucination_risk=high` tag for human review
- In `strict` mode: flagged responses are blocked; in `audit` mode: responses are passed with a warning flag

---

## 11. Evaluation (Layer 8) — Implementation Plan

**Evaluation is not optional — it is the automated quality gate between every system state and production**[cite:65][cite:71].

### Evaluation Triggers
Every one of these events triggers a full evaluation run:
- Fine-tune completes (new model version)
- Model hot-swap executed
- RAG index rebuilt (new documents added)
- Prompt template changed in Langfuse
- Weekly scheduled regression run

### Metrics & Pass Thresholds

| Metric | Tool | Min Pass Threshold | Critical Gate? |
|---|---|---|---|
| **Faithfulness** | RAGAS | ≥ 0.85 | ✅ Hard gate |
| **Answer Relevancy** | RAGAS | ≥ 0.80 | ✅ Hard gate |
| **Contextual Precision** | RAGAS | ≥ 0.75 | ✅ Hard gate |
| **Hallucination Rate** | DeepEval | ≤ 0.10 | ✅ Hard gate |
| **Refusal Accuracy** | DeepEval | ≥ 0.90 | ✅ Hard gate |
| **Forecast Numeric Accuracy** | Custom | ≥ 0.80 | ✅ Hard gate |
| **p95 Latency** | Prometheus | ≤ 3s | ⚠️ Soft gate |
| **Cache Hit Rate** | LiteLLM | ≥ 0.30 | ℹ️ Info only |

**Any hard gate failure blocks promotion**. Eval results are logged to MLflow as run artifacts and visualized in Grafana's eval trend dashboard[cite:65][cite:67].

### LLM-as-Judge for Domain Evaluation
Since ground-truth labels are limited, a second local Gemma 3 4B instance acts as judge for open-ended outputs (MIO plans, free-form forecasts). The judge prompt is versioned in Langfuse and uses a structured 1–5 rubric:
- Factual grounding (are all numbers traceable to input context?)
- Completeness (does the response address all aspects of the query?)
- Actionability (can a merchandiser act on this output immediately?)

---

## 12. Observability (Layer 9) — Implementation Plan

**The principle: every request must be fully reconstructable from logs alone**[cite:63][cite:67][cite:70].

### What Langfuse Traces for Every Request
- Trace ID → linked across RAG retrieval → guardrail check → inference → output validation
- Input prompt + rendered template
- Retrieved context (which documents, similarity scores)
- Raw LLM output (before guardrail filtering)
- Final output (after guardrail filtering)
- Latency breakdown: retrieval_ms, inference_ms, guardrail_ms, total_ms
- Token counts (prompt + completion)
- Model version tag (which champion was active)
- Hallucination guard result (pass/flag/block)

### Prometheus Metrics Exported
- `llm_request_duration_seconds{model, endpoint}` — histogram
- `llm_tokens_total{model, type}` — prompt vs completion
- `vram_used_bytes{device}` — GPU VRAM
- `cache_hit_total` / `cache_miss_total`
- `guardrail_flags_total{type}` — input / output / hallucination
- `eval_score{metric, model_version}` — faithfulness, relevancy trends

### Grafana Dashboards
1. **LLM Health Dashboard**: request volume, p50/p95/p99 latency, error rates, cache hit %, VRAM usage
2. **Eval Trends Dashboard**: faithfulness / relevancy / hallucination rate over time per model version
3. **RAG Quality Dashboard**: retrieval precision, merge rate, top retrieved documents per query cluster

### Alerting Rules (`alerts.yaml`)
```yaml
- name: VRAMCritical        condition: vram_used_bytes > 7.5GB        severity: critical
- name: LatencySpike        condition: p95_latency > 3s (5min window)  severity: warning
- name: FaithfulnessDrop    condition: faithfulness_score < 0.75       severity: critical
- name: HallucinationSpike  condition: guardrail_flag_rate > 0.15      severity: critical
- name: EvalGateFailed      condition: any hard_gate == FAIL           severity: critical
```

---

## 13. Master Controller (Layer 10) — Model Hot-Swap Protocol

A complete model swap — from decision to live traffic — takes **< 2 minutes** with this protocol:

```
1. switcher.switch("qwen2.5-3b")
   │
   ├── 1a. Download GGUF if not cached (first time only)
   ├── 1b. Re-register Modelfile with Ollama (new system prompt + params)
   ├── 1c. Update LiteLLM config → restart LiteLLM proxy (< 3 seconds)
   ├── 1d. Promote model alias in MLflow registry
   ├── 1e. TRIGGER: full evaluation suite runs automatically
   │         └── If any hard gate fails → auto-rollback to previous champion
   └── 1f. Log swap event to Langfuse + structured log
```

**The auto-rollback on eval failure is non-negotiable** — no human intervention needed. If the new model fails its quality gates, the system rolls back to the previous `@champion` alias and fires a `critical` alert.

---

## 14. Hardware Upgrade Path

The architecture requires **zero code changes** at any upgrade tier — only config updates[cite:37][cite:43]:

| Tier | Hardware | Unlocked Models | Inference Backend Change |
|---|---|---|---|
| **Now** | RTX 4070 Laptop 8GB | Gemma 3 4B, Qwen2.5-3B, Phi-4-mini | Ollama (current) |
| **Tier 1** | RTX 4090 24GB | Gemma 3 12B, Llama-3.1-8B, Qwen2.5-7B | Ollama (same config) |
| **Tier 2** | 2× RTX 4090 / A100 | Gemma 3 27B, Llama-3.1-70B Q4 | Replace Ollama → vLLM |
| **Tier 3** | Cloud burst (RunPod) | 70B+ full precision | Point LiteLLM → remote vLLM |

At Tier 2, the only change is: update `litellm_config.yaml` to point to the vLLM endpoint instead of Ollama. Every other layer (RAG, agents, guardrails, evaluation, observability) is unaffected.

---

## 15. Docker Services (Local Infrastructure)

```yaml
# docker-compose.yml — spins up supporting infrastructure only
# The LLM itself runs natively (Ollama) for maximum GPU access

services:
  langfuse:           # Observability + prompt management (self-hosted)
    image: langfuse/langfuse:latest
    ports: ["3000:3000"]
    
  redis:              # Semantic cache backend
    image: redis/redis-stack:latest
    ports: ["6379:6379"]
    
  prometheus:         # Metrics collection
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    
  grafana:            # Dashboards + alerting
    image: grafana/grafana:latest
    ports: ["3001:3000"]
    
  mlflow:             # Experiment tracking UI
    image: ghcr.io/mlflow/mlflow:latest
    ports: ["5000:5000"]
    volumes: ["./mlruns:/mlruns"]
```

**Total RAM overhead from Docker services: ~1.5–2GB** — fits within your 16GB constraint when Ollama is not actively training (inference uses ~2.5GB VRAM, ~2GB RAM for Gemma 3 4B Q4_K_M).

---

## 16. Implementation Phases

### Phase 0: Foundation (Week 1)
- [ ] Conda environment setup, all dependencies installed
- [ ] Docker services running (Langfuse, Redis, Prometheus, Grafana, MLflow)
- [ ] `config/models.yaml` populated with Gemma 3 4B + Qwen2.5-3B paths
- [ ] DVC initialized, knowledge base added to tracking

### Phase 1: Data + First Fine-Tune (Week 1–2)
- [ ] 200 initial MerchMix training samples created in canonical Alpaca JSONL format
- [ ] DataPipeline validates + checksums all samples; refusal_ratio ≥ 20% confirmed
- [ ] First Gemma 3 4B QLoRA fine-tune run (targeting 20–40 min on RTX 4070 Laptop)
- [ ] GGUF Q4_K_M exported and registered in Ollama as `merchfine`
- [ ] All training metrics logged to MLflow

### Phase 2: RAG + Inference Stack (Week 2–3)
- [ ] Knowledge base ingested: SKU catalog, MIO rules, seasonal calendar
- [ ] ChromaDB index built with hierarchical auto-merging nodes
- [ ] LiteLLM proxy configured with Redis semantic cache + Qwen2.5-3B fallback
- [ ] API endpoints (`/forecast`, `/mio-plan`, `/chat`) live via FastAPI

### Phase 3: Guardrails + Evaluation (Week 3–4)
- [ ] Guardrails AI provenance validator configured with domain sources
- [ ] NeMo self-check hallucination rail enabled (audit mode initially)
- [ ] DeepEval test suite with 50 ground-truth cases running
- [ ] RAGAS evaluation integrated — first baseline scores recorded
- [ ] Quality gate thresholds configured in `eval_thresholds.yaml`

### Phase 4: Observability + Automation (Week 4–5)
- [ ] Langfuse instrumentation on all RAG + inference calls
- [ ] Prometheus metrics exporting; Grafana dashboards imported
- [ ] Alert rules configured (VRAM, latency, faithfulness, hallucination)
- [ ] Auto-rollback on eval failure wired into model_switcher.py
- [ ] Weekly scheduled regression run enabled

### Phase 5: Iteration + Scale (Ongoing)
- [ ] Expand training dataset to 500–1,000 samples using synthetic augmentation
- [ ] Optuna sweep over lora_r (8, 16, 32) and learning rate (1e-4, 2e-4, 5e-4)
- [ ] A/B test Gemma 3 4B vs Qwen2.5-3B on real MerchMix queries (Langfuse)
- [ ] Promote winner to `@champion` via registry after eval gate confirmation

---

## 17. Key Design Decisions & Why

| Decision | Alternative Considered | Why This Choice |
|---|---|---|
| **Gemma 3 4B as primary** | Qwen2.5-7B | Qwen 7B risks OOM at 8GB during fine-tuning; Gemma 3 4B has better quality/VRAM ratio[cite:32][cite:35] |
| **LiteLLM as gateway** | Direct Ollama calls | LiteLLM adds semantic cache, fallback routing, and circuit breaking without changing app code[cite:63][cite:83] |
| **LlamaIndex over LangChain** | LangChain RAG | LlamaIndex's auto-merging retriever is superior for structured domain data; LangGraph still used for agents[cite:51][cite:54] |
| **Langfuse (self-hosted)** | LangSmith | LangSmith is proprietary, cloud-only; Langfuse is open-source and fully self-hostable[cite:70][cite:73] |
| **DeepEval for evaluation** | RAGAS only | DeepEval provides pytest-style CI/CD integration AND includes RAGAS metrics; single tool for both[cite:74][cite:81] |
| **Three guardrail layers** | Single layer | Each layer catches different failure modes; Guardrails AI catches factual drift, NeMo catches logical inconsistency[cite:79][cite:82] |
| **Ollama for local serving** | vLLM local | vLLM requires Linux for optimal performance; Ollama has native Windows support with GGUF[cite:37][cite:43] |
