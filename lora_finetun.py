from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import torch
import wandb
from dataset_processor import DatasetProcessor
import time
from omegaconf import OmegaConf
import os
import gc


# ----- CONFIG ------ #

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
    print(f'✅ CONFIG: Successfully loaded configuration with {len(config.experiments)} experiments ✅')
    return config

cfg = load_config()
# ----- DATASET ----- #

dataset_ = DatasetProcessor(tokenizer_name = cfg.base_model, n_shards_per_dataset=5)
train_dataset = dataset_()
train_dataset = train_dataset.shuffle()
print(train_dataset)
time.sleep(10)


# ----- MODEL TRAIN ----- #

class ItemTrain:
    def __init__(self, base_model_name:str, project_name:str, experiment_cfg:OmegaConf)->None:
        self.base_model_name = base_model_name
        self.project_name = project_name
        self.cfg = experiment_cfg
        self.model_id = self.cfg.base.model_id
        self.run_name = self.cfg.base.run_name
        self.base_repo_id = 'checkpoints'

        self.lora_config = LoraConfig(**self.cfg.lora_args)
        self.training_args = TrainingArguments(**self.cfg.trainer_args,
                                        overwrite_output_dir=True,
                                        logging_steps=1,
                                        output_dir=f"./{self.base_repo_id}",
                                        report_to="wandb",
                                        save_strategy="no",
                                        remove_unused_columns=True,
                                        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name,
                                                    attn_implementation="flash_attention_2")
        self.model = get_peft_model(self.model , self.lora_config)

    def __call__(self)->None:
        print(f"=== TRAIN THE {self.model_id} MODEL ===")
        wandb.init(project=self.project_name, name = self.run_name)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(f"./{self.base_repo_id}/{self.model_id}")
        self.tokenizer.save_pretrained(f"./{self.base_repo_id}/{self.model_id}")
        wandb.finish()

# ----- Experiments ----- #
for item_cfg in cfg.experiments:
    experiment = None

    try:
        experiment = ItemTrain(base_model_name = cfg.base_model,
                                project_name = cfg.project_name,
                                experiment_cfg = item_cfg)
        experiment()

    except Exception as e:
        print(f'ERROR WITH {item_cfg.base.model_id}: {e}')

    finally:
        if experiment:
            del experiment.model
            del experiment.tokenizer
            del experiment
        torch.cuda.empty_cache()
        gc.collect()
        print(f"VRAM cleared. Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()}")
    