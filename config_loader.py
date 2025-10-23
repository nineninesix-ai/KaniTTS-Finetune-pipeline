"""
Configuration Loader Utility
Centralized config management for the TTS finetuning pipeline
"""
import os
from omegaconf import OmegaConf
from typing import Optional


class ConfigLoader:
    """Singleton class to load and cache all configuration files"""

    _instance = None
    _configs = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.config_dir = os.path.join(os.path.dirname(__file__), 'config')

    def _load_config(self, config_name: str):
        """Load a config file if not already cached"""
        if config_name not in self._configs:
            config_path = os.path.join(self.config_dir, f'{config_name}.yaml')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            self._configs[config_name] = OmegaConf.load(config_path)
            print(f'✅ CONFIG: Loaded {config_name}.yaml')
        return self._configs[config_name]

    def get_model_config(self):
        """Get model and token configuration"""
        return self._load_config('model_config')

    def get_dataset_config(self):
        """Get dataset configuration"""
        return self._load_config('dataset_config')

    def get_experiments_config(self):
        """Get experiments configuration"""
        return self._load_config('experiments')

    def get_inference_config(self):
        """Get inference configuration"""
        return self._load_config('inference_config')

    def get_eval_config(self):
        """Get evaluation configuration"""
        return self._load_config('eval_config')

    def get_eval_set(self):
        """Get evaluation set"""
        return self._load_config('eval_set')


def load_config(config_path: str):
    """
    Legacy function for backward compatibility
    Load configuration from a YAML file using OmegaConf

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        Any: The loaded OmegaConf DictConfig
    """
    resolved_path = os.path.abspath(config_path)
    print(f'📁 CONFIG: Loading configuration from {resolved_path}')
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")
    config = OmegaConf.load(resolved_path)
    return config


# Global config loader instance
config_loader = ConfigLoader()
