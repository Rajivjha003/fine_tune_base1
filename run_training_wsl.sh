#!/bin/bash
# ============================================================
# MerchFine — WSL GPU Training Script
#
# Runs the full fine-tuning pipeline inside WSL where Triton
# (required by Unsloth) is natively supported.
#
# Usage from Windows PowerShell:
#   wsl -d Ubuntu bash /mnt/d/Fine_tuning/merchfine/run_training_wsl.sh
#
# Usage from WSL terminal:
#   cd /mnt/d/Fine_tuning/merchfine && bash run_training_wsl.sh
# ============================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_DIR="/mnt/d/Fine_tuning/merchfine"
VENV_DIR="${PROJECT_DIR}/venv_wsl"

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  MerchFine WSL GPU Training Pipeline${NC}"
echo -e "${CYAN}================================================${NC}"

# ── Step 1: Check CUDA ──────────────────────────────────────
echo -e "\n${YELLOW}[1/6] Checking CUDA availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
    echo -e "${GREEN}  CUDA detected via nvidia-smi${NC}"
else
    echo -e "${RED}  ERROR: nvidia-smi not found. Ensure NVIDIA drivers are installed in WSL.${NC}"
    echo -e "${YELLOW}  Try: sudo apt install nvidia-utils-XXX (match your driver version)${NC}"
    exit 1
fi

# ── Step 2: Python venv ─────────────────────────────────────
echo -e "\n${YELLOW}[2/6] Setting up Python virtual environment...${NC}"
cd "$PROJECT_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating new venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
echo -e "${GREEN}  Python: $(python3 --version) at $(which python3)${NC}"

# ── Step 3: Install dependencies ─────────────────────────────
echo -e "\n${YELLOW}[3/6] Installing dependencies...${NC}"

# Check if torch is installed with CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${GREEN}  PyTorch with CUDA already installed${NC}"
else
    echo "  Installing PyTorch with CUDA 12.4..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
fi

# Check if unsloth is installed
if python3 -c "import unsloth" 2>/dev/null; then
    echo -e "${GREEN}  Unsloth already installed${NC}"
else
    echo "  Installing Unsloth..."
    pip install unsloth -q
fi

# Install remaining requirements
echo "  Installing project requirements..."
pip install -r requirements.txt -q 2>/dev/null || true

echo -e "${GREEN}  All dependencies ready${NC}"

# ── Step 4: Verify GPU from Python ───────────────────────────
echo -e "\n${YELLOW}[4/6] Verifying GPU from Python...${NC}"
python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'  VRAM: {props.total_memory / (1024**3):.1f} GB')
    free, total = torch.cuda.mem_get_info(0)
    print(f'  VRAM free: {free / (1024**3):.1f} GB')
    print(f'  CUDA version: {torch.version.cuda}')
else:
    echo 'ERROR: CUDA not available from Python'
    exit 1
"

# ── Step 5: Run data pipeline (if no processed data) ─────────
echo -e "\n${YELLOW}[5/6] Checking training data...${NC}"
TRAIN_FILES=$(find "${PROJECT_DIR}/data/processed" -name "train_v1.0_*.jsonl" 2>/dev/null | wc -l)
if [ "$TRAIN_FILES" -eq 0 ]; then
    echo "  No processed training data found. Running data pipeline..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from data.pipeline import DataPipeline
from core.config import get_settings
settings = get_settings()
pipeline = DataPipeline()
raw_file = settings.data_dir / 'raw' / 'train_merchfine_v1.jsonl'
if raw_file.exists():
    manifest = pipeline.process(raw_file, version='1.0')
    print(manifest.summary())
else:
    print(f'ERROR: Raw data not found at {raw_file}')
    sys.exit(1)
"
    echo -e "${GREEN}  Data pipeline complete${NC}"
else
    echo -e "${GREEN}  Found ${TRAIN_FILES} processed training file(s)${NC}"
fi

# ── Step 6: Run full training ────────────────────────────────
echo -e "\n${YELLOW}[6/6] Starting QLoRA fine-tuning...${NC}"
echo -e "${CYAN}  Model: gemma-3-4b (unsloth/gemma-3-4b-it)${NC}"
echo -e "${CYAN}  LoRA: r=16, alpha=16, dropout=0.05${NC}"
echo -e "${CYAN}  Epochs: 3, batch=1, grad_accum=8${NC}"
echo ""

python3 -c "
import sys, time
sys.path.insert(0, '.')

from core.config import get_settings
from training.finetune import QLoRATrainer
from pathlib import Path

settings = get_settings()

# Find latest processed training file
processed = settings.data_dir / 'processed'
train_files = sorted(processed.glob('train_v1.0_*.jsonl'))
if not train_files:
    train_files = sorted(processed.glob('train*.jsonl'))
if not train_files:
    print('ERROR: No training data files found')
    sys.exit(1)

dataset_path = train_files[-1]
print(f'Dataset: {dataset_path}')
print(f'Dataset size: {dataset_path.stat().st_size} bytes')

trainer = QLoRATrainer(model_key='gemma-3-4b')
print(f'Trainer initialized for {trainer.model_key}')
print(f'  HF model: {trainer.model_spec.hf_id}')
print(f'  LoRA r={trainer.profile.lora_r}, alpha={trainer.profile.lora_alpha}')
print(f'  Seq length: {trainer.profile.max_seq_length}')
print(f'  Epochs: {trainer.profile.num_train_epochs}')
print()

result = trainer.train(dataset_path=dataset_path)

print()
print('=' * 60)
if result.success:
    print('TRAINING SUCCESSFUL')
    print(f'  Train Loss:    {result.train_loss:.4f}')
    print(f'  Eval Loss:     {result.eval_loss:.4f}')
    print(f'  Time:          {result.train_time_seconds/60:.1f} minutes')
    print(f'  VRAM Peak:     {result.vram_peak_gb:.2f} GB')
    print(f'  Samples:       {result.num_samples}')
    print(f'  Output Dir:    {result.output_dir}')
    print(f'  MLflow Run:    {result.mlflow_run_id or \"local (no server)\"}')
else:
    print(f'TRAINING FAILED: {result.error}')
    sys.exit(1)
print('=' * 60)
"

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}  Training pipeline complete!${NC}"
echo -e "${GREEN}================================================${NC}"
