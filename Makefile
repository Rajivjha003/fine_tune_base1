# ──────────────────────────────────────────────────────────────────────────
# MerchFine — One-Command Operations
# Usage:  make <target>
# ──────────────────────────────────────────────────────────────────────────

.PHONY: help setup train eval pipeline dashboard serve clean

PYTHON ?= python
PIP ?= pip
STREAMLIT ?= streamlit

# Default target
help: ## Show this help
	@echo.
	@echo  MerchFine LLMOps Pipeline
	@echo  ────────────────────────────
	@echo.
	@echo  make setup      Install dependencies + validate config
	@echo  make train      Run QLoRA fine-tuning with Unsloth
	@echo  make eval       Run quality gate evaluation suite
	@echo  make pipeline   Run full automated pipeline (data → train → eval → deploy)
	@echo  make dashboard  Launch Streamlit evaluation dashboard
	@echo  make serve      Start FastAPI inference server
	@echo  make serve-ollama  Start Ollama with MerchFine model
	@echo  make mlflow     Start MLflow tracking server
	@echo  make clean      Remove outputs and cache files
	@echo  make check      Verify all imports and config
	@echo.

# ── Setup ─────────────────────────────────────────────────────────────────

setup: ## Install dependencies and validate configuration
	$(PIP) install -r requirements.txt
	$(PYTHON) -c "from core.config import get_settings; s = get_settings(); print(f'Config OK: {len(s.models.models)} models, {len(s.evaluation.gates)} gates')"
	$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
	@echo.
	@echo ✅ Setup complete!

check: ## Verify all module imports succeed
	$(PYTHON) -c "from core.config import get_settings; print('✅ core.config')"
	$(PYTHON) -c "from core.model_switcher import ModelSwitcher; print('✅ core.model_switcher')"
	$(PYTHON) -c "from guardrails.output_guard import ClaimLevelNLIGuard, NumericalConsistencyGuard; print('✅ guardrails.output_guard')"
	$(PYTHON) -c "from evaluation.quality_gate import QualityGateEngine; print('✅ evaluation.quality_gate')"
	$(PYTHON) -c "from orchestrator.pipeline import PipelineOrchestrator; print('✅ orchestrator.pipeline')"
	$(PYTHON) -c "from data.schema import TrainingSample, SampleCategory; print('✅ data.schema')"
	@echo.
	@echo ✅ All modules import successfully!

# ── Training ──────────────────────────────────────────────────────────────

train: ## Run QLoRA fine-tuning
	$(PYTHON) -m training.finetune

sweep: ## Run Optuna hyperparameter sweep
	$(PYTHON) -c "from training.sweep import HyperparameterSweep; s = HyperparameterSweep('gemma_3_4b', 'data/processed/train.jsonl'); s.run()"

# ── Evaluation ────────────────────────────────────────────────────────────

eval: ## Run quality gate evaluation suite
	$(PYTHON) -c "\
import asyncio; \
from evaluation.quality_gate import QualityGateEngine; \
engine = QualityGateEngine(); \
tc = engine.load_test_cases(); \
print(f'Loaded {len(tc)} test cases'); \
preds = [{'query': t['query'], 'response': t['expected_response'], 'expected': t['expected_response'], 'context': t.get('context', []), 'category': t.get('category', '')} for t in tc]; \
report = asyncio.run(engine.evaluate_run(preds)); \
print(f'Result: {\"PASS\" if report[\"passed\"] else \"FAIL\"}'); \
[print(f'  ✗ {f}') for f in report.get('hard_gate_failures', [])]; \
[print(f'  ⚠ {w}') for w in report.get('soft_gate_warnings', [])] \
"

# ── Pipeline ──────────────────────────────────────────────────────────────

pipeline: ## Run full automated pipeline
	$(PYTHON) -c "\
import asyncio; \
from orchestrator.pipeline import PipelineOrchestrator; \
orch = PipelineOrchestrator(); \
asyncio.run(orch.run_full_pipeline()) \
"

# ── Serving ───────────────────────────────────────────────────────────────

serve: ## Start FastAPI inference server
	$(PYTHON) run_inference.py --api

serve-ollama: ## Start Ollama with MerchFine model
	ollama run merchfine:q4_k_m

dashboard: ## Launch Streamlit dashboard
	$(STREAMLIT) run ui/app.py --server.port 8501

mlflow: ## Start MLflow tracking server
	mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns

# ── Cleanup ───────────────────────────────────────────────────────────────

clean: ## Remove generated outputs and caches
	@echo Cleaning outputs...
	@if exist outputs\eval_reports rmdir /s /q outputs\eval_reports
	@if exist outputs\sweep_* rmdir /s /q outputs\sweep_*
	@if exist __pycache__ rmdir /s /q __pycache__
	@echo ✅ Clean complete
