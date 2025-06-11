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
        logger.info(f"- Dataset: {self.dataset['TTS_dataset']}")
        logger.info(f"- Base model: {self.model['base_model']}")
        logger.info(f"- Epochs: {self.training['epochs']}")
        logger.info(f"- Batch size: {self.training['batch_size']}")
        logger.info(f"- Learning rate: {self.training['learning_rate']}")
        logger.info(f"- Max sequence length: {self.token['max_sequence_length']}")

        logger.info("LoRA configuration:")
        logger.info(f"- Rank: {self.lora['rank']}")
        logger.info(f"- Alpha: {self.lora['alpha']}")
        logger.info(f"- Dropout: {self.lora['dropout']}")

    def get_config(self):
        """Get full configuration"""
        return self.config

    @property
    def token(self):
        """Get token configuration"""
        return self.config["token_config"].get()

    @property
    def lora(self):
        """Get LoRA configuration"""
        return self.config["lora_config"].get()

    @property
    def training(self):
        """Get training configuration"""
        return self.config["training_config"].get()

    @property
    def paths(self):
        """Get paths configuration"""
        return self.config["paths"].get()

    @property
    def model(self):
        """Get model configuration"""
        return self.config["model"].get()

    @property
    def dataset(self):
        """Get dataset configuration"""
        return self.config["dataset"].get()
