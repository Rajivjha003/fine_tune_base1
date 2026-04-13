# MerchFine — Project Intelligence Report
## What We Tried to Build, What We Got, and What's Next

**Project:** MerchFine — Domain-specific LLMOps system for MerchMix retail forecasting  
**Model Target:** `unsloth/gemma-3-4b-it` fine-tuned via QLoRA on MIO/inventory planning data  
**Stack:** Unsloth · FastAPI · LlamaIndex · ChromaDB · Langfuse · MLflow · DeepEval · RAGAS  
**Hardware:** RTX 4070 Laptop (8GB VRAM) · 16GB RAM · Windows · CUDA 12.5  
**Generated:** 2026-04-13

---

## Part 1 — What We Were Trying to Build (Plain Terms)

Think of MerchFine as a **private AI brain for a retail company's inventory team**. The goal was:

> *"Given a SKU, its sales history, seasonal patterns, and MIO (Merchandise Investment Order) rules — let an AI reason about demand, forecast inventory needs, explain decisions in plain English, and never make things up."*

In practice this meant building **five connected layers**:

1. **A fine-tuned small LLM** — teach a 4B model to speak the language of retail (sell-through, OTB, seasonal OTB, min/max logic, weeks of supply) using domain-specific training data
2. **A RAG knowledge layer** — connect that model to live MerchMix documents (MIO calendars, OTB sheets, product hierarchies) so every answer is grounded in real data
3. **A guardrails + evaluation firewall** — automatically block bad outputs and measure quality on every response
4. **An inference gateway** — serve the model locally with zero internet dependency, Redis caching, and circuit-breaker fallback
5. **A full observability stack** — trace every request end-to-end, alert on hallucination spikes, and auto-rollback when quality degrades

---

## Part 2 — What We Actually Got (Honest Audit)

### ✅ What Is Solid and Working

| Module | Status | What Works |
|---|---|---|
| `core/events.py` | ✅ Production-ready | Async EventBus with wildcard listener, ring-buffer history, concurrent handler execution with error isolation |
| `core/exceptions.py` | ✅ Complete | Typed exception hierarchy — `GuardrailBlockError`, `FallbackExhaustedError`, `QualityGateFailedError` |
| `core/config.py` | ✅ Complete | Pydantic Settings with YAML loader, hardware tier detection, model profiles, all params externalized |
| `training/finetune.py` | ✅ Core logic done | `QLoRATrainer` class, config-driven, MLflow integration, VRAM peak tracking, `TrainResult` dataclass |
| `data/augmentor.py` | ✅ Working | Synthetic variation generation, schema validation gate before adding to training set |
| `data/schema.py` | ✅ Complete | `TrainingSample` Pydantic model with `SampleCategory` enum |
| `rag/indexer.py` | ✅ Production-ready | 3-level hierarchy (2048→512→128), SHA256 incremental diffing, `rag.index.rebuilt` event emission |
| `api/app.py` | ✅ Correct | FastAPI lifespan management, exception handlers for all custom errors, CORS, request logging |
| `export_gguf.py` | ✅ Working | Unsloth LoRA merge + GGUF `q4_k_m` export |
| `_validate.py` | ✅ Working | Full config validation in one command |

### ⚠️ What Exists But Needs Work

| Module | Gap | Severity |
|---|---|---|
| `evaluation/quality_gate.py` | Hard gates defined in config but metrics computation is incomplete — ROUGE/BERTScore not actually computed | 🔴 Critical |
| `observability/langfuse.py` | Langfuse trace wrapper exists but `faithfulness_score` and `hallucination_rate` not passed as custom scores | 🟡 High |
| `inference/cache.py` | Semantic cache written but cosine threshold (0.85) is hardcoded, not from config | 🟡 High |
| `guardrails/output_guard.py` | Provenance validator checks structure but does NOT check if output claims appear in retrieved context | 🔴 Critical |
| `training/sweep.py` | Optuna sweep structure exists but `objective()` function doesn't return eval loss — sweep doesn't converge | 🔴 Critical |
| `ui/app.py` | Placeholder Gradio/Streamlit skeleton — no actual evaluation visualization or metric charts | 🟡 High |
| `agents/planner.py` | LangGraph planner shell exists — no actual tool invocation or state machine transitions implemented | 🟠 Medium |
| `registry/model_manager.py` | MLflow alias promotion logic exists but auto-rollback trigger is disconnected from EventBus | 🟠 Medium |

### ❌ What Is Missing Entirely

- **No evaluation dataset** — `evaluation/test_cases/` folder is empty; without test cases, the quality gate evaluates nothing
- **No Streamlit evaluation dashboard** — the project needs a visual interface to run evals and inspect results
- **No `config/` YAML files** — `training.yaml`, `rag.yaml`, `guardrails.yaml` are referenced by config.py but not present in the repo
- **No `data/raw/` seed samples** — the augmentor has no seed data to work from
- **No Modelfile for Ollama** — `export_gguf.py` creates the GGUF but there's no `Modelfile` to `ollama create` from
- **No Prometheus metrics endpoint** — `dashboard.py` references it but `/metrics` route is not mounted in `api/app.py`
- **No DVC tracking** — dataset versioning planned but not initialized

---

## Part 3 — Key Improvements to Make It Automated, Dynamic & Accurate

### 3.1 — Automation Improvements (Self-Running Pipeline)

**Problem:** Every step (data prep → training → export → eval → deployment) requires manual execution.  
**Fix:** Build a single master orchestration script that chains everything:

```
python orchestrate.py --mode full
→ data pipeline → training → GGUF export → quality gate → if PASS: ollama reload → langfuse alert
→ if FAIL: rollback to previous alias in MLflow + Slack/email alert
```

The `orchestrator/pipeline.py` exists but is not wired to `core/model_switcher.py` and the EventBus. Wire them.

**Specific automation gaps to close:**
- `training/callbacks.py` should emit `TRAINING_COMPLETED` event → triggers `quality_gate.run()` automatically
- `quality_gate.run()` result should emit `GATE_PASSED` or `GATE_FAILED` → `model_switcher.py` listens and acts
- `model_switcher.py` on `GATE_PASSED` → runs `ollama create` subprocess, promotes MLflow alias → emits `MODEL_SWAPPED`
- `observability/langfuse.py` listens to `MODEL_SWAPPED` → logs deployment event as Langfuse trace

### 3.2 — Dynamic Improvements (Live, Self-Updating System)

**Problem:** The RAG index is static; the model never improves from production feedback.  
**Fix: Three dynamic loops:**

**Loop 1 — Incremental RAG:**  
`rag/indexer.py` already has `incremental_update()` with SHA256 diffing. Schedule it via APScheduler every 6 hours watching `data/knowledge_base/` for new MIO documents.

**Loop 2 — Feedback-Driven Data Augmentation:**  
Add a feedback endpoint `POST /api/feedback` that accepts `{query, response, rating, corrected_response}`. Store in `data/feedback/`. A weekly batch job filters low-rated responses (rating < 3), routes them through `data/augmentor.py` to generate corrected training pairs, appends to the versioned dataset, and re-triggers training.

**Loop 3 — Model Registry Rotation:**  
`core/upgrade_planner.py` (exists) should scan MLflow for new Gemma/Qwen releases weekly. If a new model passes quality gates with >5% improvement on domain F1, auto-schedule a fine-tuning run on that base.

### 3.3 — Accuracy Improvements (Hallucination Reduction)

**Problem:** `guardrails/output_guard.py` checks format but not factual grounding.  
**Fix — Three-layer faithfulness stack:**

**Layer 1 — Context-Grounded Output Guard:**  
After RAG retrieval, store `retrieved_chunks` in request context. In `output_guard.py`, decompose the LLM response into atomic claims using an NLI model (`cross-encoder/nli-deberta-v3-base`). For each claim, check if at least one retrieved chunk entails it. If any claim has no supporting chunk → inject a disclaimer or block.

**Layer 2 — Calibrated Temperature:**  
In `inference/gateway.py`, set `temperature=0.1` for forecasting queries (deterministic) and `temperature=0.3` for explanation queries. Currently temperature is not dynamically adjusted by query type.

**Layer 3 — Numerical Consistency Check:**  
MerchMix outputs contain numbers (forecast quantities, sell-through %, OTB values). Add a post-processing validator in `guardrails/output_guard.py` that extracts all numbers from the response using regex, cross-checks them against the input context numbers, and flags responses where numbers appear in output but NOT in context (likely hallucinated).

---

## Part 4 — Complete Evaluation Metrics Reference

### Tier 1 — Fine-Tuning Quality Metrics (Pre-RAG, Model-Level)

| Metric | What It Measures | Target | Tool |
|---|---|---|---|
| **Perplexity (domain)** | How well model predicts domain tokens. Lower = better fit | < 12 (vs base ~24) | `evaluate` library |
| **Perplexity (general)** | Catastrophic forgetting check. Should NOT rise >15% | < 10 on MMLU | `evaluate` library |
| **ROUGE-L** | Longest common subsequence overlap with reference answers | > 0.65 | `rouge-score` |
| **BLEU-4** | 4-gram precision vs reference. Good for structured outputs | > 0.35 | `sacrebleu` |
| **BERTScore (F1)** | Semantic similarity using BERT embeddings — better than ROUGE for paraphrase | > 0.82 | `bert-score` |
| **METEOR** | Recall-weighted, handles synonyms better than BLEU | > 0.55 | `evaluate` library |
| **Domain F1** | Exact match on structured fields (SKU, quantities, dates) | > 0.85 | Custom |
| **Training Loss Curve** | Convergence shape — should decrease monotonically, no spikes | < 0.3 at epoch 3 | MLflow |

### Tier 2 — RAG Pipeline Metrics (Retrieval + Generation)

| Metric | What It Measures | Target | Tool |
|---|---|---|---|
| **Faithfulness** | % of response claims supported by retrieved context | > 0.85 | RAGAS / DeepEval |
| **Answer Relevancy** | How well the answer addresses the actual question | > 0.80 | RAGAS |
| **Context Precision** | % of retrieved chunks that are actually relevant | > 0.75 | RAGAS |
| **Context Recall** | % of ground truth info covered by retrieved chunks | > 0.80 | RAGAS |
| **Context Entity Recall** | Named entities (SKUs, dates, categories) preserved in retrieval | > 0.85 | RAGAS |
| **Answer Correctness** | Combined factual + semantic correctness vs ground truth | > 0.75 | RAGAS |
| **Hallucination Score** | LLM-as-judge assessment of fabricated claims | < 0.10 | DeepEval |
| **Groundedness** | RAG Triad: is the answer grounded in context? | > 0.85 | TruLens / DeepEval |

### Tier 3 — Inference & System Metrics (Production)

| Metric | What It Measures | Target | Tool |
|---|---|---|---|
| **P50 / P95 Latency** | Median and tail inference latency | P50 < 800ms, P95 < 2s | Langfuse + Prometheus |
| **Cache Hit Rate** | % requests served from Redis semantic cache | > 40% | Custom counter |
| **Fallback Rate** | % requests that hit fallback model | < 5% | Prometheus |
| **VRAM Peak (GB)** | Maximum GPU memory during inference | < 6.5 GB | `torch.cuda` |
| **Tokens/Second** | Inference throughput | > 15 tok/s | Ollama API |
| **Circuit Breaker Opens** | How often primary model fails | < 1/hour | Prometheus alert |

### Tier 4 — Domain-Specific MerchMix Metrics

| Metric | What It Measures | Target |
|---|---|---|
| **Forecast MAPE** | Mean Absolute Percentage Error on demand forecasts vs actuals | < 15% |
| **OTB Accuracy** | % of OTB recommendations within ±10% of correct value | > 80% |
| **SKU Hallucination Rate** | % responses containing non-existent SKU IDs | < 2% |
| **Numeric Grounding Rate** | % of numbers in response traceable to input context | > 95% |
| **Category Classification F1** | Correct product category assignment | > 0.90 |
| **Refusal Rate (appropriate)** | % of unanswerable queries correctly refused | > 85% |

---

## Part 5 — Known Bugs

### 🔴 Critical Bugs (System Cannot Run Correctly)

**BUG-001: `training/sweep.py` — Optuna objective returns None**  
Location: `sweep.py` → `objective()` function  
Problem: The Optuna trial does not return `eval_loss` — sweep will run but optimize nothing  
Fix: Return `trainer_output.training_loss` from the `SFTTrainer` result inside `objective()`

**BUG-002: `guardrails/output_guard.py` — No claim-context matching**  
Location: `output_guard.py` → validation logic  
Problem: Output guard checks JSON format only; does NOT verify claims against retrieved context  
Fix: Integrate `cross-encoder/nli-deberta-v3-base` for claim-level entailment check

**BUG-003: `evaluation/test_cases/__init__.py` — Empty**  
Location: `evaluation/test_cases/`  
Problem: Quality gate has nothing to evaluate against; always passes vacuously  
Fix: Create minimum 50 ground-truth QA pairs in `evaluation/test_cases/domain_qa.jsonl`

**BUG-004: Missing `config/` YAML files**  
Location: `core/config.py` references `config/training.yaml`, `config/rag.yaml`  
Problem: System fails to start without these files  
Fix: Create all YAML config files with documented defaults

**BUG-005: `inference/cache.py` — Hardcoded cosine threshold**  
Location: `cache.py` line ~45  
Problem: `similarity_threshold = 0.85` is hardcoded, should come from `settings.inference.cache_similarity_threshold`  
Fix: Replace with config reference

### 🟡 High-Severity Bugs

**BUG-006: `api/app.py` — `/metrics` Prometheus endpoint not mounted**  
Problem: `observability/dashboard.py` sets up metrics but no route exposes them  
Fix: Add `from prometheus_client import make_asgi_app` and mount at `/metrics`

**BUG-007: `registry/model_manager.py` — Auto-rollback disconnected from EventBus**  
Problem: `GATE_FAILED` event is emitted but no listener calls `model_manager.rollback()`  
Fix: Wire listener in `core/system_init.py` during bootstrap

**BUG-008: `data/augmentor.py` — No seed data guard**  
Problem: `generate_variations()` called with empty `seed_samples` crashes silently  
Fix: Add early return with warning log when `len(seed_samples) == 0`

**BUG-009: `export_gguf.py` — No Modelfile generation**  
Problem: GGUF is created but no `Modelfile` is written for `ollama create`  
Fix: After GGUF export, write `Modelfile` with system prompt and parameters

---

## Part 6 — Next Phase Roadmap

### Phase 1 — Stabilize (Week 1–2) 🔴 Must Do First

- [ ] Create `config/training.yaml`, `config/rag.yaml`, `config/guardrails.yaml`
- [ ] Fix BUG-001 through BUG-005 (critical bugs)
- [ ] Create 50–100 ground-truth evaluation pairs in `evaluation/test_cases/domain_qa.jsonl`
- [ ] Write `Modelfile` and test `ollama run merchfine-gemma-3-4b`
- [ ] Run `_validate.py` → all green
- [ ] Run one end-to-end training cycle and confirm MLflow logs appear

### Phase 2 — Evaluation Dashboard (Week 2–3) 🟡 Core Deliverable

- [ ] Build Streamlit evaluation dashboard (`ui/app.py` — full rebuild)
- [ ] Implement all Tier 1 metrics (Perplexity, ROUGE-L, BERTScore, BLEU-4)
- [ ] Implement all Tier 2 RAG metrics via RAGAS
- [ ] Connect dashboard to Langfuse API for live production traces
- [ ] Add comparison view: base model vs fine-tuned vs fine-tuned+RAG

### Phase 3 — Automation (Week 3–4) 🟢 High Leverage

- [ ] Wire EventBus: `TRAINING_COMPLETED` → `quality_gate` → `GATE_PASSED/FAILED` → `model_switcher`
- [ ] Build `POST /api/feedback` endpoint with storage in `data/feedback/`
- [ ] Schedule `rag/indexer.py` incremental update with APScheduler
- [ ] Write `orchestrate.py` master script for full pipeline

### Phase 4 — Dynamic Accuracy (Week 4–5) 🔵 Advanced

- [ ] Implement claim-level NLI output guard (BUG-002 fix + full integration)
- [ ] Add numerical consistency validator in `guardrails/output_guard.py`
- [ ] Build feedback → augmentation → retraining weekly batch loop
- [ ] Add domain-specific MerchMix metrics (Forecast MAPE, OTB Accuracy)
- [ ] A/B testing: route 20% traffic to new model, 80% to current via LiteLLM

### Phase 5 — Production Hardening (Week 5–6) ⚫ Final

- [ ] Load test at 100 concurrent requests — identify bottlenecks
- [ ] Add DVC for full dataset version control
- [ ] Prometheus + Grafana dashboard with alerting rules
- [ ] Document all APIs with OpenAPI spec
- [ ] Security audit: prompt injection stress test on 200 adversarial inputs

---

## Summary Table — Current State vs Target

| Dimension | Current State | Target State |
|---|---|---|
| **Training** | Manual execution, one model | Automated, multi-model sweep |
| **Evaluation** | Config gates, no actual metric computation | Full 20+ metric dashboard |
| **RAG** | Index built, retrieval works | Incremental updates, feedback loop |
| **Guardrails** | Format-only validation | Claim-level NLI faithfulness check |
| **Observability** | Langfuse wrapper exists | Live traces + Prometheus + alerting |
| **Automation** | Zero end-to-end automation | Full pipeline from data → prod |
| **Dashboard** | Placeholder | Streamlit eval + Langfuse integration |
| **Data** | No seed data, no test cases | 500+ samples, 100+ eval cases |

---

*Report generated by MerchFine AI Chief Architect — MerchMix LLMOps Intelligence*
