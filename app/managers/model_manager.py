import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils.logger import logger


class ModelManager:
    def __init__(self, config_manager):
        self.config = config_manager
        self.tokenizer = None
        self.model = None
        self._initialize_tokenizer()
        self._initialize_model()

    def _initialize_tokenizer(self):
        """Initialize the tokenizer"""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_details["base_model"],
            resume_download=True,
            cache_dir=self.config.model_cache_dir,
        )

        # Set padding token to EOS token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set padding token to EOS token")

        self._analyze_tokenizer()

    def _analyze_tokenizer(self):
        """Analyze the tokenizer's vocabulary"""
        logger.info("\nTokenizer Analysis:")
        all_tokens = self.tokenizer.get_vocab()

        special_tokens = [
            token
            for token in all_tokens.keys()
            if token.startswith("<|") and token.endswith("|>")
        ]
        custom_tokens = [
            token for token in all_tokens.keys() if token.startswith("<custom_token_")
        ]

        logger.info(f"Total vocabulary size: {len(all_tokens)}")
        logger.info(f"Number of special tokens: {len(special_tokens)}")
        logger.info(f"Number of custom tokens: {len(custom_tokens)}")

    def _initialize_model(self):
        """Initialize the base model and apply LoRA configuration"""
        logger.info("Loading base model (this might take a while)...")
        model_args = self.config.model_config.copy()
        model_args["cache_dir"] = self.config.model_cache_dir

        # Explicitly set use_cache based on gradient checkpointing config
        # This is crucial for compatibility with LoRA and gradient checkpointing
        if self.config.training.get("gradient_checkpointing"):
            logger.info("Gradient checkpointing is enabled. Setting use_cache=False.")
            model_args["use_cache"] = False
        else:
            model_args["use_cache"] = True

        # Handle different torch dtype options
        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }

        if model_args["torch_dtype"] in dtype_mapping:
            model_args["torch_dtype"] = dtype_mapping[model_args["torch_dtype"]]
            logger.info(f"Using torch dtype: {model_args['torch_dtype']}")
        else:
            logger.warning(
                f"Unsupported torch dtype: {model_args['torch_dtype']}. Defaulting to float16"
            )
            model_args["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_details["base_model"], **model_args
        )

        # Enable gradient checkpointing and disable cache for LoRA training
        if self.config.training.get("gradient_checkpointing"):
            logger.info(
                "Gradient checkpointing is enabled. Activating it on the model and disabling cache..."
            )
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        else:
            self.model.config.use_cache = True
            logger.info("`use_cache` is enabled for the model.")

        # Configure and apply LoRA
        self._configure_lora()

    def _configure_lora(self):
        """Configure and apply LoRA to the model"""
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(**self.config.lora)

        logger.info("Wrapping model with LoRA...")
        self.model = get_peft_model(self.model, lora_config)
        self._ensure_trainable_layers()

    def _ensure_trainable_layers(self):
        """Ensure critical layers are trainable"""
        logger.info("Ensuring critical layers are trainable...")

        for layer_name in self.config.model_details["trainable_layers"]:
            if hasattr(self.model.base_model.model, layer_name):
                for param in getattr(
                    self.model.base_model.model, layer_name
                ).parameters():
                    param.requires_grad = True
                logger.info(f"Set {layer_name}.parameters().requires_grad = True")
            elif hasattr(self.model, layer_name):
                for param in getattr(self.model, layer_name).parameters():
                    param.requires_grad = True
                logger.info(
                    f"Set {layer_name}.parameters().requires_grad = True (direct on model)"
                )

        self._log_trainable_parameters()

    def _log_trainable_parameters(self):
        """Log information about trainable parameters"""
        trainable_params = 0
        all_param = 0
        for name, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                logger.info(
                    f"Trainable parameter: {name}, dtype: {param.dtype}, device: {param.device}"
                )
        logger.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}%"
        )

    def prepare_for_training(self):
        """Prepare the model for training"""
        self.model.train()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Model moved to CUDA device")

        logger.info(f"Model training mode: {self.model.training}")
        logger.info(f"Model device: {next(self.model.parameters()).device}")

    def save_model(self, path):
        """Save the model and tokenizer"""
        logger.info(f"Saving model to {path}...")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
