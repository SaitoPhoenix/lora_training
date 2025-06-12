from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
from utils.logger import logger
from callbacks.training_progress import TrainingProgressCallback


class TrainingManager:
    def __init__(self, config_manager, model_manager, dataset_manager):
        self.config = config_manager
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
        self.trainer = None

    def _setup_wandb(self):
        """Initialize Weights & Biases"""
        logger.info("Initializing Weights & Biases...")
        wandb.init(
            project=self.config.wandb["project_name"],
            name=self.config.wandb["run_name"],
            dir=self.config.paths["wandb_dir"],
        )

    def _setup_training_args(self):
        """Set up training arguments"""
        logger.info("Setting up training arguments...")
        # Add output_dir to training args
        training_args = self.config.training.copy()
        training_args["output_dir"] = f"./{self.config.paths['save_folder']}"

        return TrainingArguments(**training_args)

    def train(self):
        """Run the training process"""
        # Initialize wandb
        self._setup_wandb()

        # Set up training arguments
        training_args = self._setup_training_args()

        # Initialize trainer
        logger.info("Initializing trainer...")
        self.trainer = Trainer(
            model=self.model_manager.model,
            args=training_args,
            train_dataset=self.dataset_manager.train_dataset,
            eval_dataset=self.dataset_manager.validation_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.model_manager.tokenizer,
                mlm=False,
                pad_to_multiple_of=8,  # Pad to multiple of 8 for better performance
            ),
            callbacks=[TrainingProgressCallback()],
        )

        # Start training
        logger.info("Starting training...")
        logger.info(
            f"Training will be logged to Weights & Biases project: {self.config.wandb['project_name']}, "
            f"run: {self.config.wandb['run_name']}"
        )
        self.trainer.train()

        # Save the model
        self._save_model()

    def _save_model(self):
        """Save the trained model"""
        save_path = (
            f"./{self.config.paths['save_folder']}/{self.config.wandb['run_name']}"
        )
        logger.info(f"Saving model to {save_path}...")
        self.model_manager.save_model(save_path)
