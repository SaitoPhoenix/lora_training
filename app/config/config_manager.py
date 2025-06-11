import confuse
import os
from utils.logger import logger


class ConfigManager:
    def __init__(self, config_path="config/training_config.yaml"):
        self.config = confuse.Configuration("lora_training")
        self.config.set_file(config_path)
        self._setup_cache_dirs()
        self._log_config()

    def _setup_cache_dirs(self):
        """Set up cache directories for models and datasets"""
        self.cache_dir = os.path.expanduser("~/.cache/huggingface")
        self.model_cache_dir = os.path.join(self.cache_dir, "hub")
        self.dataset_cache_dir = os.path.join(self.cache_dir, "datasets")

        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs(self.dataset_cache_dir, exist_ok=True)

        logger.info(f"Model cache directory: {self.model_cache_dir}")
        logger.info(f"Dataset cache directory: {self.dataset_cache_dir}")

    def _log_config(self):
        """Log the main configuration parameters"""
        logger.info("Training configuration:")
        logger.info(f"- Dataset: {self.get('TTS_dataset')}")
        logger.info(f"- Base model: {self.get('model_name')}")
        logger.info(f"- Epochs: {self.get('epochs')}")
        logger.info(f"- Batch size: {self.get('batch_size')}")
        logger.info(f"- Learning rate: {self.get('learning_rate')}")
        logger.info(
            f"- Max sequence length: {self.get('token_config.max_sequence_length')}"
        )

        logger.info("LoRA configuration:")
        logger.info(f"- Rank: {self.get('lora_config.rank')}")
        logger.info(f"- Alpha: {self.get('lora_config.alpha')}")
        logger.info(f"- Dropout: {self.get('lora_config.dropout')}")

    def get(self, key, default=None):
        """Get a configuration value using dot notation"""
        try:
            return self.config[key].get()
        except confuse.exceptions.NotFoundError:
            return default

    @property
    def token_config(self):
        """Get token configuration"""
        return {
            "start_of_text": self.get("token_config.start_of_text"),
            "end_of_text": self.get("token_config.end_of_text"),
            "start_of_speech": self.get("token_config.start_of_speech"),
            "end_of_speech": self.get("token_config.end_of_speech"),
            "start_of_human": self.get("token_config.start_of_human"),
            "end_of_human": self.get("token_config.end_of_human"),
            "start_of_ai": self.get("token_config.start_of_ai"),
            "end_of_ai": self.get("token_config.end_of_ai"),
            "pad_token": self.get("token_config.pad_token"),
            "max_sequence_length": self.get("token_config.max_sequence_length"),
        }

    @property
    def lora_config(self):
        """Get LoRA configuration"""
        return {
            "rank": self.get("lora_config.rank"),
            "alpha": self.get("lora_config.alpha"),
            "dropout": self.get("lora_config.dropout"),
        }

    @property
    def training_config(self):
        """Get training configuration"""
        return {
            "dataset": self.get("TTS_dataset"),
            "model_name": self.get("model_name"),
            "epochs": self.get("epochs"),
            "batch_size": self.get("batch_size"),
            "learning_rate": self.get("learning_rate"),
            "save_steps": self.get("save_steps"),
            "number_processes": self.get("number_processes"),
            "save_folder": self.get("save_folder"),
            "project_name": self.get("project_name"),
            "run_name": self.get("run_name"),
        }
