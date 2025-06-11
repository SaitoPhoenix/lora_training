from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
from utils.logger import get_logger as logger
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
            project=self.config.get("paths.project_name"),
            name=self.config.get("paths.run_name"),
        )

    def _setup_training_args(self):
        """Set up training arguments"""
        logger.info("Setting up training arguments...")
        return TrainingArguments(
            overwrite_output_dir=True,
            num_train_epochs=self.config.get("training_args.epochs"),
            per_device_train_batch_size=self.config.get("training_args.batch_size"),
            logging_steps=1,
            bf16=True,  # Enable bfloat16 mixed precision
            output_dir=f"./{self.config.get('paths.save_folder')}",
            report_to="wandb",
            save_steps=self.config.get("training_args.save_steps"),
            remove_unused_columns=False,
            learning_rate=self.config.get("training_args.learning_rate"),
            gradient_accumulation_steps=4,
            warmup_steps=100,
            gradient_checkpointing=True,
            optim="adamw_torch",
            max_grad_norm=1.0,
            dataloader_pin_memory=True,
            dataloader_num_workers=self.config.get("training_args.number_processes"),
        )

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
            train_dataset=self.dataset_manager.dataset,
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
            f"Training will be logged to Weights & Biases project: {self.config.get('paths.project_name')}, "
            f"run: {self.config.get('paths.run_name')}"
        )
        self.trainer.train()

        # Save the model
        self._save_model()

    def _save_model(self):
        """Save the trained model"""
        save_path = f"./{self.config.get('paths.save_folder')}/lora_adapter"
        logger.info(f"Saving model to {save_path}...")
        self.model_manager.save_model(save_path)
