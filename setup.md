# MerchFine LLMOps — Complete Setup Guide

> **Tested Python:** 3.12.8 | **OS:** Windows 10/11 | **GPU:** NVIDIA RTX 4070 (CUDA 12.4)
> Last verified: 2026-04-13

---

## Prerequisites

| Tool | Required Version | Check Command |
|------|-----------------|---------------|
| **Python** | 3.12.x (NOT 3.13) | `python --version` |
| **CUDA Toolkit** | 12.4+ | `nvcc --version` |
| **Git** | Any | `git --version` |
| **Ollama** | Latest | `ollama --version` |

> ⚠️ **Python 3.13 will NOT work** — `unsloth`, `bitsandbytes`, `torch` are not yet compatible. Use 3.12.x.

---

## Step 0: Install Python 3.12 (if needed)

```bash
# Download from: https://www.python.org/downloads/release/python-3128/
# Choose: Windows installer (64-bit)
# During install: CHECK "Add Python to PATH"

# Verify
python --version
# Expected: Python 3.12.8
```

If you have multiple Python versions, use the full path:
```
C:\Users\rajiv\AppData\Local\Programs\Python\Python312\python.exe
```

---

## Step 1: Create Virtual Environment

```bash
cd D:\Fine_tuning

# Create venv with Python 3.12
python -m venv venv

# Activate it
venv\Scripts\activate

# Verify correct Python
python --version    # Should show 3.12.x
where python        # Should show D:\Fine_tuning\venv\Scripts\python.exe
```

---

## Step 2: Install PyTorch with CUDA (GPU training)

> ⚠️ **This MUST be done first, separately.** Order matters.

```bash
# For NVIDIA GPU with CUDA 12.4:
pip install torch==2.10.0 torchvision==0.25.0 torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify GPU detection:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**If you DON'T have a GPU** (CPU-only evaluation/dashboard):
```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 3: Install Unsloth (QLoRA training engine)

```bash
pip install unsloth

# Verify
python -c "import unsloth; print('Unsloth OK')"
```

> This requires CUDA/GPU. If you're on CPU-only, skip this step — training won't work but evaluation and dashboard will.

---

## Step 4: Install All Dependencies

```bash
cd D:\Fine_tuning\merchfine
pip install -r requirements.txt
```

**If you get version conflicts**, install in this exact order:
```bash
# Core ML stack (after torch is installed)
pip install transformers==5.5.0 peft==0.18.1 trl==0.24.0 bitsandbytes==0.49.2 datasets==4.3.0 accelerate==1.13.0

# Evaluation frameworks
pip install sentence-transformers==5.4.0 ragas==0.4.3 rouge-score==0.1.2 deepeval==3.4.6

# MLflow + Experiment tracking
pip install mlflow==3.11.1 optuna==4.8.0

# RAG stack
pip install llama-index-core==0.14.20 llama-index-vector-stores-chroma==0.5.5 llama-index-embeddings-huggingface==0.7.0 chromadb==1.5.7 rank-bm25==0.2.2

# LLM orchestration
pip install langchain==1.2.15 langchain-core==1.2.28 langchain-openai==1.1.12 langgraph==1.1.6 litellm==1.82.5

# Observability
pip install langfuse==4.2.0

# UI + API
pip install streamlit==1.56.0 plotly==6.7.0 fastapi==0.135.3 uvicorn==0.44.0

# Guardrails + utilities
pip install guardrails-ai==0.10.0 pydantic==2.12.5 pydantic-settings==2.13.1 python-dotenv==1.0.1 rich==14.3.4 psutil==7.2.2

# Ollama client
pip install ollama==0.6.1
```

---

## Step 5: Environment Setup

```bash
cd D:\Fine_tuning\merchfine

# Copy .env.example to .env and fill in your keys
copy .env.example .env

# Edit .env — at minimum set:
# OPENAI_API_KEY=sk-...      (for LLM-as-judge eval)
# HF_TOKEN=hf_...            (for gated models like Gemma)
# OLLAMA_HOST=http://127.0.0.1:11434
```

---

## Step 6: Verify Installation

```bash
cd D:\Fine_tuning\merchfine

# Quick import test — should print "12/12 PASS"
python -c "
from core.config import get_settings; print('L0 Config OK')
from data.schema import TrainingSample, SampleCategory; print('L1 Schema OK')
from training.sweep import HyperparameterSweep; print('L2 Training OK')
from core.model_switcher import ModelSwitcher; print('L3 Registry OK')
from rag.retriever import HybridRetriever; print('L5 RAG OK')
from agents.planner import DemandPlannerAgent; print('L6 Agents OK')
from guardrails.input_guard import InputSanitizer; print('L7 Guardrails OK')
from evaluation.quality_gate import QualityGateEngine; print('L8 Eval OK')
from observability.langfuse import LangfuseTracker; print('L9 Observability OK')
from orchestrator.pipeline import PipelineOrchestrator; print('L10 Orchestrator OK')
from core.exceptions import RAGRetrievalError, RAGIndexError; print('Exceptions OK')
import sentence_transformers, rouge_score, deepeval; print('Eval deps OK')
print('ALL OK')
"
```

---

## Step 7: Run the System

```bash
# Activate venv first!
cd D:\Fine_tuning\merchfine
D:\Fine_tuning\venv\Scripts\activate

# ── Dashboard ──
streamlit run ui/app.py

# ── Inference Server ──
python run_inference.py --api

# ── Run Evaluation ──
python -c "
import asyncio
from evaluation.quality_gate import QualityGateEngine
engine = QualityGateEngine()
tc = engine.load_test_cases()
preds = [{'query':t['query'],'response':t['expected_response'],'expected':t['expected_response'],'context':t.get('context',[]),'category':t.get('category','')} for t in tc]
report = asyncio.run(engine.evaluate_run(preds))
for k,v in report['metrics'].items():
    s = v['score']
    status = 'PASS' if v['passed'] else 'FAIL'
    print(f'  {k}: {s:.4f} [{status}]')
"

# ── API Chat Test ──
curl -X POST "http://127.0.0.1:8000/api/chat" ^
     -H "Content-Type: application/json" ^
     -d "{\"message\": \"Forecast demand for SKU Classic Crew Tee next 4 weeks\", \"use_rag\": false}"
```

---

## Troubleshooting

### `torch` / `torchvision` version mismatch
```bash
# Always install together from the same index:
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu124
```

### `No module named 'unsloth'`
Only available on CUDA environments. For CPU-only, skip training — evaluation and dashboard work fine without it.

### `UnicodeEncodeError` in terminal
```bash
# Add to your .env or run before commands:
set PYTHONIOENCODING=utf-8
```

### `sentence-transformers` import error
Must have matching torch + torchvision versions. Reinstall both together (see Step 2).

### RAGAS returns 0.0 scores
Either `ragas` isn't installed or it needs an OpenAI API key for LLM-based metrics. With our ROUGE-L fallback, it works offline but check `OPENAI_API_KEY` in `.env` for full RAGAS.

---

## Version Lock File

Save this as `requirements-lock.txt` for exact reproducibility:

```
# Core (install these FIRST via --index-url)
torch==2.10.0+cu124
torchvision==0.25.0+cu124

# ML/Training
transformers==5.5.0
peft==0.18.1
trl==0.24.0
bitsandbytes==0.49.2
datasets==4.3.0
accelerate==1.13.0
unsloth==2026.4.4
sentence-transformers==5.4.0
safetensors==0.7.0
tokenizers==0.22.2

# Evaluation
ragas==0.4.3
deepeval==3.4.6
rouge-score==0.1.2
optuna==4.8.0

# LLM/RAG
langchain==1.2.15
langchain-core==1.2.28
langchain-openai==1.1.12
langgraph==1.1.6
litellm==1.82.5
ollama==0.6.1
openai==2.30.0
chromadb==1.5.7
rank-bm25==0.2.2
llama-index-core==0.14.20

# Observability
mlflow==3.11.1
langfuse==4.2.0

# UI/API
streamlit==1.56.0
plotly==6.7.0
fastapi==0.135.3
uvicorn==0.44.0

# Core Python
pydantic==2.12.5
pydantic-settings==2.13.1
python-dotenv==1.0.1
numpy==2.4.3
pandas==2.3.3
scikit-learn==1.8.0
rich==14.3.4
psutil==7.2.2
PyYAML==6.0.3
```