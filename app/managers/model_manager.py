import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
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
            self.config.model["base_model"],
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

        if special_tokens:
            logger.info("\nSpecial tokens found:")
            for token in special_tokens:
                logger.info(f"- {token} (ID: {all_tokens[token]})")

        if custom_tokens:
            logger.info("\nCustom tokens found:")
            custom_tokens.sort(key=lambda x: int(x.split("_")[-1].rstrip(">")))
            for token in custom_tokens[:10]:
                logger.info(f"- {token} (ID: {all_tokens[token]})")
            if len(custom_tokens) > 10:
                logger.info(f"... and {len(custom_tokens) - 10} more custom tokens")

    def _initialize_model(self):
        """Initialize the base model and apply LoRA configuration"""
        logger.info("Loading base model (this might take a while)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model["base_model"],
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            resume_download=True,
            cache_dir=self.config.model_cache_dir,
        )

        # Enable gradient computation for all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        # Configure and apply LoRA
        self._configure_lora()

    def _configure_lora(self):
        """Configure and apply LoRA to the model"""
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=self.config.lora["rank"],
            lora_alpha=self.config.lora["alpha"],
            lora_dropout=self.config.lora["dropout"],
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
            bias="none",
            modules_to_save=None,
            task_type="CAUSAL_LM",
            use_rslora=True,
            inference_mode=False,
            init_lora_weights="gaussian",
            use_dora=False,
        )

        logger.info("Wrapping model with LoRA...")
        self.model = get_peft_model(self.model, lora_config)
        self._ensure_trainable_layers()

    def _ensure_trainable_layers(self):
        """Ensure critical layers are trainable"""
        logger.info("Ensuring critical layers are trainable...")

        # Handle lm_head
        if hasattr(self.model.base_model.model, "lm_head"):
            for param in self.model.base_model.model.lm_head.parameters():
                param.requires_grad = True
            logger.info("Set lm_head.parameters().requires_grad = True")
        elif hasattr(self.model, "lm_head"):
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            logger.info(
                "Set lm_head.parameters().requires_grad = True (direct on model)"
            )

        # Handle embed_tokens
        if hasattr(self.model.base_model.model, "embed_tokens"):
            for param in self.model.base_model.model.embed_tokens.parameters():
                param.requires_grad = True
            logger.info("Set embed_tokens.parameters().requires_grad = True")
        elif hasattr(self.model, "embed_tokens"):
            for param in self.model.embed_tokens.parameters():
                param.requires_grad = True
            logger.info(
                "Set embed_tokens.parameters().requires_grad = True (direct on model)"
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
