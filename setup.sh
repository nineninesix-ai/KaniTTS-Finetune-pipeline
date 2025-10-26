#!/usr/bin/env bash
# Setup script for an ML environment with CUDA-aware PyTorch install.
# - Creates a Python venv
# - Logs in to Hugging Face & Weights & Biases
# - Detects CUDA and installs matching torch/torchvision/torchaudio
# - Installs transformers/accelerate/datasets
# - Tries to install flash-attn (optional; non-fatal if it fails)

set -euo pipefail

# -------- COLORS --------
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# -------- HEADER --------
clear
echo -e "${CYAN}${BOLD}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║          N I N E N I N E S I X  😼                         ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo -e "${MAGENTA}"
echo "              /\\_/\\  "
echo "             ( -.- )───┐"
echo "              > ^ <    │"
echo -e "${NC}"
echo ""
echo -e "${GREEN}${BOLD}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║   Nano Codec TTS Finetuning - Environment Setup            ║${NC}"
echo -e "${GREEN}${BOLD}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📦 This script will set up two Python environments:${NC}"
echo -e "${BLUE}   • venv_finetuning - For training models${NC}"
echo -e "${BLUE}   • venv_eval - For evaluation and inference${NC}"
echo ""
echo -e "${YELLOW}⚠️  Requirements: Ubuntu Linux with NVIDIA GPU (CUDA)${NC}"
echo ""
sleep 2

# -------- PYTHON INSTALLATION --------

echo -e "${GREEN}${BOLD}[STEP 1/6] Installing Python 3.10...${NC}"
echo -e "${BLUE}📥 Adding deadsnakes PPA and installing Python packages${NC}"

add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.10 python3.10-dev python3.10-venv

echo -e "${GREEN}✓ Python 3.10 installed successfully${NC}"
echo ""

# -------- finetuning setup --------

echo -e "${GREEN}${BOLD}[STEP 2/6] Creating Finetuning Environment...${NC}"
echo -e "${BLUE}🔧 Creating Python virtual environment: venv_finetuning${NC}"
python3.10 -m venv venv_finetuning
source venv_finetuning/bin/activate

echo -e "${BLUE}⬆️  Upgrading pip and build tools${NC}"
python -m pip install -U pip setuptools wheel packaging ninja
echo -e "${GREEN}✓ Virtual environment ready${NC}"
echo ""

echo -e "${GREEN}${BOLD}[STEP 3/6] Installing CLI Tools...${NC}"
echo -e "${BLUE}📦 Installing HuggingFace Hub, Git LFS, and Weights & Biases${NC}"
pip install -U "huggingface_hub[cli]" git-lfs wandb
echo -e "${GREEN}✓ CLI tools installed${NC}"
echo ""


echo -e "${GREEN}${BOLD}[STEP 4/6] Detecting CUDA and Installing PyTorch...${NC}"

TORCH_VER="2.8.0"
TV_VER="0.23.0"
TA_VER="2.8.0"

detect_cuda() {
  local ver=""
  if command -v nvidia-smi &>/dev/null; then
    ver=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -n 1 || true)
  elif command -v nvcc &>/dev/null; then
    ver=$(nvcc --version | grep -o 'release [0-9]\+\.[0-9]\+' | awk '{print $2}')
  fi
  echo "$ver"
}

CUDA_VER="$(detect_cuda)"

if [[ -z "${CUDA_VER}" ]]; then
  echo -e "${YELLOW}⚠️  No CUDA detected — installing CPU wheels for PyTorch${NC}"
  echo -e "${BLUE}📥 Installing PyTorch ${TORCH_VER} (CPU only)${NC}"
  pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VER}+cpu" \
    "torchvision==${TV_VER}+cpu" \
    "torchaudio==${TA_VER}+cpu"

else
  echo -e "${GREEN}🎮 Detected CUDA ${CUDA_VER}${NC}"
  CUDA_MAJOR="$(echo "$CUDA_VER" | cut -d. -f1)"
  CUDA_MINOR="$(echo "$CUDA_VER" | cut -d. -f2)"
  CUDA_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"  # e.g. cu128

  # 🧠 Special rule: if CUDA 13.x detected, use cu128 wheels (latest supported)
  if [[ "${CUDA_MAJOR}" -eq 13 ]]; then
    echo -e "${YELLOW}⚙️  CUDA 13 detected — using compatibility mode (PyTorch cu128 wheels)${NC}"
    CUDA_TAG="cu128"
  fi

  WHL_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
  echo -e "${BLUE}📥 Trying PyTorch wheels for ${CUDA_TAG} (${WHL_INDEX})${NC}"

  set +e
  pip install --index-url "${WHL_INDEX}" --extra-index-url https://pypi.org/simple \
    "torch==${TORCH_VER}+${CUDA_TAG}" \
    "torchvision==${TV_VER}+${CUDA_TAG}" \
    "torchaudio==${TA_VER}+${CUDA_TAG}"
  code=$?
  set -e

  if [[ $code -ne 0 ]]; then
    echo -e "${YELLOW}⚠️  ${CUDA_TAG} wheels not found. Falling back to cu121.${NC}"
    pip install --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple \
      "torch==${TORCH_VER}+cu121" \
      "torchvision==${TV_VER}+cu121" \
      "torchaudio==${TA_VER}+cu121"
  fi
fi

echo ""
echo -e "${BLUE}🔍 Verifying PyTorch installation...${NC}"
python - <<'PY'
import torch
print(f"OK: torch {torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
PY
echo -e "${GREEN}✓ PyTorch installed successfully${NC}"
echo ""

echo -e "${GREEN}${BOLD}[STEP 5/6] Installing ML Libraries...${NC}"
echo -e "${BLUE}📦 Installing transformers, accelerate, datasets, omegaconf, peft${NC}"
pip install "transformers==4.56.0" "accelerate==1.10.1" "datasets==3.6.0" omegaconf peft
echo -e "${GREEN}✓ ML libraries installed${NC}"
echo ""

echo -e "${GREEN}${BOLD}[STEP 6/6] Installing Flash Attention (Optional)...${NC}"
echo -e "${BLUE}⚡ Attempting to install flash-attn 2.8.3${NC}"
echo -e "${YELLOW}ℹ️  This may take a while and requires compilation. Non-fatal if it fails.${NC}"

ARCH="$(python - <<'PY'
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"{major}.{minor}")
PY
)"
if [[ -n "${ARCH}" ]]; then
  export TORCH_CUDA_ARCH_LIST="${ARCH}"
  echo -e "${BLUE}🔧 Setting TORCH_CUDA_ARCH_LIST=${ARCH} for optimal compilation${NC}"
fi



set +e
pip install "flash-attn==2.8.3" --no-build-isolation
FA_CODE=$?
set -e
if [[ $FA_CODE -ne 0 ]]; then
  echo -e "${YELLOW}⚠️  Flash-attn build failed. Continuing with PyTorch SDPA attention.${NC}"
else
  echo -e "${GREEN}✓ Flash Attention installed successfully${NC}"
fi
echo ""

echo -e "${GREEN}${BOLD}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║   Finetuning Environment Setup Complete! ✓                 ║${NC}"
echo -e "${GREEN}${BOLD}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}📊 Environment Summary:${NC}"
python - <<'PY'
import torch, transformers, accelerate, datasets
print(f"  • PyTorch:      {torch.__version__} (CUDA: {torch.version.cuda})")
print(f"  • Transformers: {transformers.__version__}")
print(f"  • Accelerate:   {accelerate.__version__}")
print(f"  • Datasets:     {datasets.__version__}")
PY
echo ""
deactivate
echo -e "${MAGENTA}😼 Finetuning environment ready! Activate with:${NC}"
echo -e "${CYAN}   source venv_finetuning/bin/activate${NC}"
echo ""


# -------- EVAL environment setup --------

echo -e "${GREEN}${BOLD}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║   Setting Up Evaluation Environment...                     ║${NC}"
echo -e "${GREEN}${BOLD}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}🔧 Creating Python virtual environment: venv_eval${NC}"
python3.10 -m venv venv_eval
source venv_eval/bin/activate

echo -e "${BLUE}📦 Installing PyTorch and core dependencies${NC}"
pip install torch "datasets==3.6.0" omegaconf peft

echo -e "${BLUE}📦 Installing audio processing libraries${NC}"
pip install scipy librosa numpy

echo -e "${BLUE}📦 Installing NeMo Toolkit (this may take a while)${NC}"
pip install nemo_toolkit[all]

echo -e "${BLUE}📦 Installing latest transformers from GitHub${NC}"
pip install -U "git+https://github.com/huggingface/transformers.git"

echo ""
echo -e "${GREEN}${BOLD}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║   All Environments Ready! 🎉                               ║${NC}"
echo -e "${GREEN}${BOLD}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}${BOLD}📝 Quick Start:${NC}"
echo ""
echo -e "${MAGENTA}For Training:${NC}"
echo -e "${CYAN}  source venv_finetuning/bin/activate${NC}"
echo -e "${CYAN}  python3 lora_finetun.py${NC}"
echo ""
echo -e "${MAGENTA}For Evaluation:${NC}"
echo -e "${CYAN}  source venv_eval/bin/activate${NC}"
echo -e "${CYAN}  python eval_research.py${NC}"
echo ""
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}"
echo "              /\\_/\\  "
echo "             ( ^.^ ) NINENINESIX"
echo "              > ^ <  "
echo -e "${NC}"
echo -e "${GREEN}${BOLD}Happy Finetuning! 🚀${NC}"
echo ""
