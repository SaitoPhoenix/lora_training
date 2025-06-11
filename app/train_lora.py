from config.config_manager import ConfigManager
from managers.model_manager import ModelManager
from managers.dataset_manager import DatasetManager
from managers.training_manager import TrainingManager


def main():
    # Initialize configuration
    config_manager = ConfigManager()

    # Initialize model
    model_manager = ModelManager(config_manager)
    model_manager.prepare_for_training()

    # Initialize dataset
    dataset_manager = DatasetManager(config_manager)
    dataset_manager.load_and_preprocess(model_manager.tokenizer)

    # Initialize and run training
    training_manager = TrainingManager(config_manager, model_manager, dataset_manager)
    training_manager.train()


if __name__ == "__main__":
    main()
