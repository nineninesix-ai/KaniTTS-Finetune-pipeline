from dataclasses import dataclass
from datasets import Dataset, Audio
from omegaconf import OmegaConf
from inference import NemoAudioPlayer, KaniModel
import os
from scipy.io.wavfile import write
from config_loader import config_loader

# Load centralized configs
eval_cfg = config_loader.get_eval_config()
model_cfg = config_loader.get_model_config()

# Create audio output directory
audio_dir = eval_cfg.paths.audio_output_dir
os.makedirs(audio_dir, exist_ok=True)

def load_config(config_path: str = './config/experiments.yaml'):
    """Load configuration from a YAML file using OmegaConf.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Any: The loaded OmegaConf DictConfig.
    """
    resolved_path = os.path.abspath(config_path)
    print(f'📁 CONFIG: Loading configuration from {resolved_path}')
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")
    config = OmegaConf.load(resolved_path)
    return config

# Load experiment and evaluation configs
exp_cfg = config_loader.get_experiments_config()
eval_set = config_loader.get_eval_set()

# Initialize player without config (will use centralized config)
player = NemoAudioPlayer()


# * experiment_id: str
# * train_configuration: json
# * sentence_id: str
# * sentence: str
# * audio

dataset = []
sample_rate = model_cfg.codec.sample_rate

# --- base model check ---
print("=== Loading base model ===")
model = KaniModel(config=None, model_name=exp_cfg.base_model, player=player)
for sentence in eval_set.eval_set:
    try:
        id_ = list(sentence.keys())[0]
        print(f"--- Generating: {id_} ---")
        sentence = sentence[id_]
        wave, _ = model.run_model(sentence)
        audio_path = os.path.join(audio_dir, f"base_model: {exp_cfg.base_model.replace('/', '__')}_{id_}.wav")
        row = {
            "experiment_id": f"base_model: {exp_cfg.base_model}",
            "train_configuration": {'base_model': exp_cfg.base_model},
            "sentence_id": id_,
            "sentence": sentence,
            "audio": audio_path
        }
        dataset.append(row)
        write(audio_path, sample_rate, wave)
    except Exception as e:
        print(f"Error for sentence {id_}: {e}")


# --- finetuned models check ---

checkpoints_dir = eval_cfg.paths.checkpoints_dir
for exp_id in exp_cfg.experiments:
    model_path = os.path.join(checkpoints_dir, exp_id.base.model_id)
    print(f"=== Loading model from {model_path} ===")
    model = KaniModel(config=None, model_name=model_path, player=player)
    for sentence in eval_set.eval_set:
        try:
            id_ = list(sentence.keys())[0]
            print(f"--- Generating: {id_} ---")
            sentence = sentence[id_]
            wave, _ = model.run_model(sentence)
            audio_path = os.path.join(audio_dir, f"{exp_id.base.model_id}_{id_}.wav")
            row = {
                "experiment_id": exp_id.base.model_id,
                "train_configuration": OmegaConf.to_container(exp_id),
                "sentence_id": id_,
                "sentence": sentence,
                "audio": audio_path
            }
            dataset.append(row)
            write(audio_path, sample_rate, wave)
        except Exception as e:
            print(f"Error for sentence {id_}: {e}")

dataset = Dataset.from_list(dataset)
dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))

# Push to HuggingFace Hub if configured
if eval_cfg.huggingface.upload_dataset:
    dataset.push_to_hub(
        eval_cfg.huggingface.repo_name,
        private=eval_cfg.huggingface.private
    )

        