# Makefile for NINENINESIX Nano Codec TTS Finetuning Pipeline
# Simplifies common development tasks

.PHONY: help setup login train eval clean test-config

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
MAGENTA := \033[0;35m
NC := \033[0m # No Color

# Python virtual environment paths
VENV_FINETUNE := venv_finetuning
VENV_EVAL := venv_eval
PYTHON_FINETUNE := $(VENV_FINETUNE)/bin/python
PYTHON_EVAL := $(VENV_EVAL)/bin/python

help:
	@echo ""
	@echo "$(CYAN)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║          N I N E N I N E S I X  😼                         ║$(NC)"
	@echo "$(CYAN)║   Nano Codec TTS Finetuning Pipeline - Make Commands      ║$(NC)"
	@echo "$(CYAN)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@echo "  $(YELLOW)make setup$(NC)       - Run complete environment setup (requires sudo)"
	@echo "  $(YELLOW)make login$(NC)       - Login to HuggingFace and Weights & Biases"
	@echo "  $(YELLOW)make train$(NC)       - Run finetuning experiments"
	@echo "  $(YELLOW)make eval$(NC)        - Run evaluation and generate audio samples"
	@echo "  $(YELLOW)make test-config$(NC) - Test configuration files for errors"
	@echo "  $(YELLOW)make clean$(NC)       - Clean up generated files and caches"
	@echo "  $(YELLOW)make help$(NC)        - Show this help message"
	@echo ""
	@echo "$(MAGENTA)Quick Start:$(NC)"
	@echo "  1. $(CYAN)make setup$(NC)  # Install everything (one time only)"
	@echo "  2. $(CYAN)make login$(NC)  # Authenticate with HF and W&B"
	@echo "  3. $(CYAN)make train$(NC)  # Start finetuning"
	@echo "  4. $(CYAN)make eval$(NC)   # Evaluate trained models"
	@echo ""

setup:
	@echo "$(GREEN)Starting environment setup...$(NC)"
	@echo "$(YELLOW)⚠️  This requires sudo access$(NC)"
	@chmod +x setup.sh
	@sudo ./setup.sh
	@echo "$(GREEN)✓ Setup complete!$(NC)"

login:
	@echo "$(GREEN)Configuring authentication...$(NC)"
	@echo ""
	@echo "$(CYAN)Setting up git credential helper...$(NC)"
	@git config --global credential.helper store
	@echo "$(GREEN)✓ Git credential helper configured$(NC)"
	@echo ""
	@echo "$(CYAN)Logging in to HuggingFace...$(NC)"
	@echo "$(YELLOW)Please enter your HuggingFace token when prompted:$(NC)"
	@bash -c "source $(VENV_FINETUNE)/bin/activate && hf auth login"
	@echo ""
	@echo "$(CYAN)Logging in to Weights & Biases...$(NC)"
	@echo "$(YELLOW)Please enter your W&B API key when prompted:$(NC)"
	@bash -c "source $(VENV_FINETUNE)/bin/activate && wandb login"
	@echo ""
	@echo "$(GREEN)✓ Authentication complete!$(NC)"

train:
	@echo "$(GREEN)Starting finetuning experiments...$(NC)"
	@echo "$(CYAN)Using configuration: config/experiments.yaml$(NC)"
	@echo "$(CYAN)Dataset config: config/dataset_config.yaml$(NC)"
	@echo ""
	@if [ ! -d "$(VENV_FINETUNE)" ]; then \
		echo "$(YELLOW)⚠️  Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	@bash -c "source $(VENV_FINETUNE)/bin/activate && python3 lora_finetun.py"
	@echo ""
	@echo "$(GREEN)✓ Training complete!$(NC)"
	@echo "$(MAGENTA)Checkpoints saved to: ./checkpoints/$(NC)"

eval:
	@echo "$(GREEN)Starting evaluation...$(NC)"
	@echo "$(CYAN)Using configuration: config/eval_config.yaml$(NC)"
	@echo "$(CYAN)Evaluation set: config/eval_set.yaml$(NC)"
	@echo ""
	@if [ ! -d "$(VENV_EVAL)" ]; then \
		echo "$(YELLOW)⚠️  Eval environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	@if [ ! -d "checkpoints" ]; then \
		echo "$(YELLOW)⚠️  No checkpoints found. Run 'make train' first.$(NC)"; \
		exit 1; \
	fi
	@bash -c "source $(VENV_EVAL)/bin/activate && python eval_research.py"
	@echo ""
	@echo "$(GREEN)✓ Evaluation complete!$(NC)"
	@echo "$(MAGENTA)Audio samples saved to: ./audio_samples/$(NC)"

test-config:
	@echo "$(GREEN)Testing configuration files...$(NC)"
	@echo ""
	@bash -c "source $(VENV_FINETUNE)/bin/activate && python -c '\
		from config_loader import config_loader; \
		print(\"✓ model_config.yaml\"); \
		config_loader.get_model_config(); \
		print(\"✓ dataset_config.yaml\"); \
		config_loader.get_dataset_config(); \
		print(\"✓ experiments.yaml\"); \
		config_loader.get_experiments_config(); \
		print(\"✓ inference_config.yaml\"); \
		config_loader.get_inference_config(); \
		print(\"✓ eval_config.yaml\"); \
		config_loader.get_eval_config(); \
		print(\"✓ eval_set.yaml\"); \
		config_loader.get_eval_set(); \
		print(\"\n$(GREEN)All configuration files valid!$(NC)\")'"

clean:
	@echo "$(GREEN)Cleaning up...$(NC)"
	@echo "$(CYAN)Removing Python cache files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "$(CYAN)Removing temporary files...$(NC)"
	@rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete!$(NC)"

# Advanced targets
.PHONY: clean-all upload-model activate-finetune activate-eval

clean-all: clean
	@echo "$(YELLOW)⚠️  This will remove checkpoints, audio samples, and virtual environments!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to cancel, or Enter to continue...$(NC)"
	@read confirm
	@rm -rf checkpoints audio_samples $(VENV_FINETUNE) $(VENV_EVAL)
	@echo "$(GREEN)✓ Full cleanup complete!$(NC)"

upload-model:
	@echo "$(GREEN)Uploading model to HuggingFace Hub...$(NC)"
	@read -p "Enter checkpoint name (e.g., model_id from experiments.yaml): " model_id; \
	read -p "Enter HuggingFace repo (e.g., username/repo-name): " hf_repo; \
	bash -c "source $(VENV_FINETUNE)/bin/activate && huggingface-cli upload $$hf_repo ./checkpoints/$$model_id --private"
	@echo "$(GREEN)✓ Model uploaded!$(NC)"

activate-finetune:
	@echo "$(CYAN)To activate finetuning environment, run:$(NC)"
	@echo "  source $(VENV_FINETUNE)/bin/activate"

activate-eval:
	@echo "$(CYAN)To activate evaluation environment, run:$(NC)"
	@echo "  source $(VENV_EVAL)/bin/activate"
