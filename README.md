# KaniTTS Finetuning Pipeline

[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/NzP3rjB4SB?style=flat)](https://discord.gg/NzP3rjB4SB) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


<a href="https://www.nineninesix.ai/" target="_blank">
  <img src="https://www.nineninesix.ai/kitty.png" alt="996.ai" width="120" height="120">
</a>


```
===============================================
          N I N E N I N E S I X  😼
===============================================

          /\_/\
         ( -.- )───┐
          > ^ <    │
===============================================
```

---

This is a **complete, production-ready workflow** for finetuning **LFM2-based text-to-speech models** (like KaniTTS) with **NeMo NanoCodec** on your own speakers and languages.

- **Finetune** on your custom voice datasets
- **Run multiple experiments** with different hyperparameter configurations in one go
- **Automatically evaluate** each trained model on a test set
- **Upload results to HuggingFace Hub** for easy comparison and sharing
- **Perfect for hyperparameter optimization** (like Gridsearch)

---

## 📋 Requirements

### Hardware & System
- **Linux with NVIDIA GPU** (recommended: RTX 5090 or similar)
- **CUDA 12.8+** (the setup script handles PyTorch installation automatically)

### Dataset Preparation

> **IMPORTANT**: This pipeline requires a dataset tokenized with **NeMo NanoCodec**.

If you don't have a tokenized dataset yet, prepare it using our dataset processing pipeline: **[NanoCodec Dataset Pipeline](https://github.com/nineninesix-ai/nano-codec-dataset-pipeline)**

The dataset pipeline will:
- Convert your raw audio files to NeMo NanoCodec tokens
- Create the exact format needed for this finetuning pipeline
- Handle multi-speaker datasets automatically

## 🚀 Quick Start

### Step 1: Setup (One-Time)

```bash
# Clone the repository
git clone https://github.com/your-org/KaniTTS-Finetune-pipeline
cd KaniTTS-Finetune-pipeline

# Run setup (this will take 10-15 minutes, perfect time for a coffee break! ☕)
make setup
```

The setup script will:
- Install Python 3.10
- Create two virtual environments (training and evaluation)
- Detect your CUDA version and install matching PyTorch
- Install all required dependencies
- Optionally build Flash Attention 2 for faster training

### Step 2: Login to Services

```bash
make login
```

This will authenticate you with:
- **HuggingFace Hub** (for downloading base models and uploading results)
- **Weights & Biases** (for experiment tracking)

### Step 3: Configure Your Experiments

Edit the configuration files (detailed explanation below):
1. `config/dataset_config.yaml` - Your training datasets
2. `config/experiments.yaml` - Hyperparameters for your experiments

### Step 4: Train!

```bash
make train
```

Training can take several hours depending on your dataset size and GPU.

### Step 5: Evaluate

```bash
make eval
```

This will generate audio samples for all your trained models and upload them to HuggingFace Hub for comparison.

---

## 📊 Configuration Guide

This pipeline uses YAML configuration files.

### Basic Configuration

#### Dataset Configuration (`config/dataset_config.yaml`)

This file tells the pipeline where to find your training data and how to process it.

```yaml
max_duration_sec: 12  # Maximum audio duration in seconds

hf_datasets:
  - reponame: "your-username/your-dataset-repo"
    name: null  # Subset name (use null if no subsets)
    split: "train"
    text_col_name: text  # Column containing transcriptions
    nano_layer_1: nano_layer_1  # codec layer 1
    nano_layer_2: nano_layer_2  # codec layer 2
    nano_layer_3: nano_layer_3  # codec layer 3
    nano_layer_4: nano_layer_4  # codec layer 4
    encoded_len: encoded_len    # Audio length in frames
    speaker_id: "alice"  # OPTIONAL: speaker identifier May be `null` if you want to tune no support speaker_id model
    max_len: 10000      # OPTIONAL: limit number of samples
```

**Required Fields:**
- `reponame`: Your HuggingFace dataset repository (must be tokenized with NanoCodec)
- `text_col_name`: Column name containing text transcriptions
- `nano_layer_1/2/3/4`: Column names for the 4 codec layers
- `encoded_len`: Column with audio length in codec frames

**Optional Fields:**

**`speaker_id`** (HIGHLY RECOMMENDED for multi-speaker datasets):
- If specified: The pipeline prepends this to each text prompt (`"alice: Hello world"`)
- If omitted: Text is used as-is without speaker prefix
- **⚠️ IMPORTANT**: Each dataset in the list should represent **ONE speaker only**
  - ✅ Good: One dataset per speaker with `speaker_id: "alice"`
  - ❌ Bad: One dataset with multiple speakers and no speaker_id
  - Why? This helps the model learn speaker-specific characteristics

**`max_len`**:
- Limits the number of samples from this dataset
- Useful for balancing datasets or quick testing

**`categorical_filter`**:
```yaml
categorical_filter:
  column_name: "speaker"
  value: "speaker_001"
```
- Filters dataset to only include rows where `column_name == value`

#### What the Pipeline Does

When you run training, the dataset processor:

1. **Loads** all datasets from HuggingFace Hub
2. **Filters** by duration (removes samples longer than `max_duration_sec`)
3. **Renames** columns to standard names
4. **Applies** categorical filters if specified
5. **Limits** samples if `max_len` is set
6. **Converts** 4 codec layers into interleaved token sequences
7. **Removes** consecutive duplicate frames (compression)
8. **Prepends** speaker IDs to text if specified
9. **Creates** training sequences with special tokens
10. **Concatenates** all datasets and shuffles

The processing uses **multiprocessing** for speed, automatically detecting your CPU count.

#### Examples
##### Single Speaker Dataset

```yaml
max_duration_sec: 12

hf_datasets:
  - reponame: "my-username/alice-voice-nano"
    name: null
    split: "train"
    text_col_name: text
    nano_layer_1: nano_layer_1
    nano_layer_2: nano_layer_2
    nano_layer_3: nano_layer_3
    nano_layer_4: nano_layer_4
    encoded_len: encoded_len
    speaker_id: "alice"
```

##### Multi-Speaker Dataset

```yaml
max_duration_sec: 12

hf_datasets:
  # Alice's voice
  - reponame: "my-username/alice-voice-nano"
    name: null
    split: "train"
    text_col_name: text
    nano_layer_1: nano_layer_1
    nano_layer_2: nano_layer_2
    nano_layer_3: nano_layer_3
    nano_layer_4: nano_layer_4
    encoded_len: encoded_len
    speaker_id: "alice"
    max_len: 5000

  # Bob's voice
  - reponame: "my-username/bob-voice-nano"
    name: null
    split: "train"
    text_col_name: text
    nano_layer_1: nano_layer_1
    nano_layer_2: nano_layer_2
    nano_layer_3: nano_layer_3
    nano_layer_4: nano_layer_4
    encoded_len: encoded_len
    speaker_id: "bob"
    max_len: 5000
```

> **Key Point**: Notice each dataset has a **different `speaker_id`**. This is crucial for the model to learn to distinguish between speakers!


### Experiment Configuration (`config/experiments.yaml`)

This file defines your training experiments. You can run multiple experiments with different hyperparameters in a single training run.

#### Basic Structure

```yaml
base_model: "nineninesix/kani-tts-450m-0.2-pt"
project_name: "my-tts-experiments"

experiments:
  - base:
      model_id: "alice-tts-v1-lora16"
      run_name: "alice-experiment-001"
      desc: "Standard LoRA rank 16"
    lora_args:
      r: 16
      lora_alpha: 16
      lora_dropout: 0.1
      target_modules: [q_proj, v_proj, w1, w2, w3]
      bias: "none"
      task_type: CAUSAL_LM
      use_rslora: true
    trainer_args:
      num_train_epochs: 2
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 4
      learning_rate: 5e-5
      lr_scheduler_type: cosine
      warmup_ratio: 0.1
```

#### Understanding LoRA

LoRA (Low-Rank Adaptation) is a technique that lets you finetune large models efficiently:
- Instead of updating all 450 million parameters, LoRA adds small "adapter" layers
- Much faster training and lower memory requirements
- The adapters are merged back into the model after training

**This pipeline is designed specifically for LFM2-based models** like KaniTTS.

#### LoRA Hyperparameters Explained

**`r` (rank)**: Controls the size of adapter matrices
- Lower (8-16): Faster, less memory, simpler adaptations
- Higher (32-64): More expressive, better quality, slower
- **Recommended**: Start with 16

**`lora_alpha` (scaling factor)**: Usually set equal to `r`
- Controls how much the LoRA adapters influence the model
- **Recommended**: Same value as `r`

**`lora_dropout`**: Regularization to prevent overfitting
- **Recommended**: 0.1 for most cases

**`target_modules`**: Which parts of the model to finetune

Available modules and what they do:
- `q_proj, k_proj, v_proj`: **Attention query/key/value** (core attention mechanism)
- `out_proj`: **Attention output** projection
- `w1, w2, w3`: **Feed-forward network** layers (most of the model's capacity)
- `in_proj`: **Input projection** (less commonly used)

**Common patterns:**

```yaml
# Minimal (fastest, good for testing)
target_modules: [q_proj, v_proj]

# Balanced (recommended for most cases)
target_modules: [q_proj, v_proj, w1, w2, w3]

# Comprehensive (best quality, slower)
target_modules: [q_proj, k_proj, v_proj, out_proj, w1, w2, w3]

# Full (everything, highest quality)
target_modules: [q_proj, k_proj, v_proj, out_proj, w1, w2, w3, in_proj]
```

**`use_rslora`**: Improved LoRA variant
- **Recommended**: `true` (better training stability)

#### Training Hyperparameters Explained

**`num_train_epochs`**: How many times to go through the dataset
- **Recommended**: 1-3 epochs for most cases
- More epochs = better fit but risk of overfitting

**`per_device_train_batch_size`**: Samples per GPU
- **Recommended**: 1 (TTS models use long sequences)
- Increase if you have lots of VRAM

**`gradient_accumulation_steps`**: Accumulate gradients before update
- Effective batch size = `batch_size × accumulation_steps`
- **Recommended**: 4-8 (gives effective batch size of 4-8)

**`learning_rate`**: How fast the model learns
- **Recommended**: 2e-5 to 5e-5
- Too high = unstable training
- Too low = slow convergence

**`lr_scheduler_type`**: Learning rate schedule
- **Recommended**: `cosine` (gradually decreases LR)
- Alternative: `linear`, `constant`

**`warmup_ratio`**: Fraction of training for warmup
- **Recommended**: 0.1 (10% of training)
- Gradually increases LR at the start for stability

**`weight_decay`**: Regularization
- **Recommended**: 0.01-0.02
- Prevents overfitting

**`optim`**: Optimizer
- **Recommended**: `adamw_torch` (standard choice)

**`bf16`**: Use bfloat16 precision
- **Recommended**: `true` (faster, less memory, supported by modern GPUs like Blackwell)

#### Why Multiple Experiments?

You can **run many experiments in one go**:

```yaml
experiments:
  # Experiment 1: Conservative
  - base:
      model_id: "alice-conservative"
      run_name: "exp-conservative"
    lora_args:
      r: 8
      target_modules: [q_proj, v_proj]
    trainer_args:
      learning_rate: 2e-5
      num_train_epochs: 1

  # Experiment 2: Balanced
  - base:
      model_id: "alice-balanced"
      run_name: "exp-balanced"
    lora_args:
      r: 16
      target_modules: [q_proj, v_proj, w1, w2, w3]
    trainer_args:
      learning_rate: 5e-5
      num_train_epochs: 2

  # Experiment 3: Aggressive
  - base:
      model_id: "alice-aggressive"
      run_name: "exp-aggressive"
    lora_args:
      r: 32
      target_modules: [q_proj, k_proj, v_proj, out_proj, w1, w2, w3]
    trainer_args:
      learning_rate: 5e-5
      num_train_epochs: 3
```

The pipeline will:
1. Load and preprocess your dataset **once**
2. Run each experiment **sequentially**
3. Save each model to `./checkpoints/<model_id>`
4. Clear GPU memory between experiments
5. Log everything to Weights & Biases

You can then evaluate all of them and pick the best one. This is perfect for hyperparameter optimization with tools like **Optuna**.

#### Base Model

The default base model is:
```yaml
base_model: "nineninesix/kani-tts-400m-0.3-pt"
```

This model is pretrained on multiple languages:
- 🇬🇧 English (en)
- 🇰🇷 Korean (ko)
- 🇨🇳 Chinese (zh)
- 🇪🇸 Spanish (es)
- 🇩🇪 German (de)
- 🇸🇦 Arabic (ar)
- 🇯🇵 Japanese (ja)
- 🇰🇬 Kyrgyz (ky)

**Coming soon**: More advanced versions with support for additional languages including French, Portuguese, and more.

### Evaluation Configuration (`config/eval_config.yaml` & `config/eval_set.yaml`)

#### eval_config.yaml

```yaml
paths:
  audio_output_dir: "./audio_samples"
  checkpoints_dir: "./checkpoints"

huggingface:
  upload_dataset: true
  repo_name: "your-username/tts-evaluation-results"
  private: true

processing:
  num_proc: 4
```

- `audio_output_dir`: Where to save generated audio files
- `checkpoints_dir`: Where your trained models are saved
- `upload_dataset`: Whether to upload results to HuggingFace Hub
- `repo_name`: Your HuggingFace dataset repo for evaluation results
- `private`: Keep evaluation dataset private

#### eval_set.yaml

This file contains the prompts for evaluation:

```yaml
eval_set:
  - prompt_1: "alice: Hello, this is a test of the text to speech system."
  - prompt_2: "alice: The quick brown fox jumps over the lazy dog."
  - prompt_3: "alice: Machine learning is transforming the world of artificial intelligence."
```

**When to include speaker ID:**
- ✅ Include (`alice:`) if you trained with `speaker_id` in dataset config
- ❌ Omit if you didn't use speaker IDs during training

#### How Evaluation Works

When you run `make eval`, the pipeline:

1. **Loads the base model** (untrained)
2. **Generates audio** for each prompt in `eval_set.yaml`
3. **Loads each finetuned model** from `./checkpoints/`
4. **Generates audio** for each prompt with each model
5. **Creates a HuggingFace dataset** with:
   - `experiment_id`: Model identifier
   - `train_configuration`: Complete hyperparameter config (!)
   - `sentence_id`: Prompt identifier
   - `sentence`: The text prompt
   - `audio`: The generated audio file
6. **Uploads to HuggingFace Hub** (if enabled)

#### Why This Is Powerful

On HuggingFace Hub, you can:
- 🎧 **Listen** to all generated samples side-by-side
- 📝 **See** which text was used
- ⚙️ **View** the exact hyperparameters for each model
- 📊 **Compare** base model vs all finetuned variants
- 🔗 **Share** results with your team

This makes it easy to pick the best model or iterate on your experiments.

### Other Configuration Files

#### model_config.yaml

Defines the token space and codec settings. **You usually don't need to modify this.**

- Token IDs for special tokens (start_of_speech, end_of_speech, etc.)
- Codec parameters (sample rate, codebook size, etc.)
- This is a centralized config used by both training and inference

#### inference_config.yaml

Controls inference behavior:
- `max_new_tokens`: Maximum audio length
- `temperature`: Sampling randomness (0.6 = balanced)
- `top_p`: Nucleus sampling threshold
- `repetition_penalty`: Discourage repetition

---

## 🏗️ How It Works Under the Hood

### The Audio Codec

This pipeline uses **NeMo NanoCodec** which compresses audio into discrete tokens:
- **4 codebooks** of 4096 tokens each
- **22kHz sample rate** (high quality)
- **0.6 kbps** (extreme compression)
- **12.5 fps** (12.5 codec frames per second)

The model learns to predict these codec tokens from text, then the codec reconstructs them into audio.

### Token Space

The model's vocabulary is extended beyond text:
- **0-64399**: Text tokens (standard tokenizer)
- **64400+**: Special control tokens (start/end markers)
- **64410+**: Audio codec tokens (4 codebooks × 4032 tokens each)

Training sequences look like:
```
[start_of_human] <text_tokens> [end_of_text] [end_of_human]
[start_of_ai] [start_of_speech] <audio_tokens> [end_of_speech] [end_of_ai]
```

### Parallel Processing

Dataset preprocessing uses multiprocessing:
- Splits each dataset into shards
- Processes shards in parallel on CPU
- Automatically detects CPU count
- Configurable with `n_shards_per_dataset` parameter

This makes preprocessing fast even for large datasets!

---

## 📓 Google Colab Notebook
The notebook is available in the `notebooks/` directory, so you can run experiments without a local GPU.

---

## 🛠️ Additional Commands

```bash
# Validate configuration files
make test-config

# Clean cache files
make clean

# Upload a trained model to HuggingFace
make upload-model

# Show all available commands
make help
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use lower LoRA rank (`r: 8` instead of `r: 16`)
- Reduce `max_duration_sec` in dataset config

### Flash Attention Build Failed
- Non-fatal - training will use PyTorch SDPA instead
- Flash Attention provides ~20% speedup but is optional
- Ensure CUDA toolkit version matches PyTorch

### Dataset Loading Issues
- Ensure your HuggingFace token has access to the dataset
- Check column names in `dataset_config.yaml` match your dataset
- Verify dataset was tokenized with NanoCodec (4 layers + encoded_len)
- Use `make test-config` to validate config files

### Training Errors
- Check that `base_model` path is correct in `experiments.yaml`
- Ensure you're logged in to HuggingFace and W&B (`make login`)
- Verify your dataset has the required columns after processing

---

## 💬 Support & Community

Need help or want to share your results?

- 📝 **GitHub Issues**: [Report bugs or request features](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline/issues)
- 💬 **Discord Community**: [![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/NzP3rjB4SB?style=flat)](https://discord.gg/NzP3rjB4SB)

Join our Discord to:
- Get help from the community
- Share your trained models
- Discuss tips and tricks
- Stay updated on new releases

---

## 🙏 Acknowledgments

This pipeline is built on top of open-source projects:

- [NVIDIA NeMo NanoCodec](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps) - neural audio codec
- [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) - backbone LLM
- **HuggingFace** transformers and Hub
- PyTorch, LoRA (PEFT)
  
---

## 📄 License

This project is under Apache 2. See [LICENSE](LICENSE) file for details.
