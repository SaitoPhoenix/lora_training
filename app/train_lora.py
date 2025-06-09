from datasets import load_dataset  # Loads your preprocessed dataset
from peft import LoraConfig, get_peft_model  # For LoRA configuration and model wrapping
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)  # Core Hugging Face components
import yaml  # To load configuration from config.yaml
import wandb  # For experiment tracking
import torch
from tqdm import tqdm
import logging
import sys
import time
from datetime import datetime
import requests
import shutil
import tempfile
import os.path as osp
import numpy as np
import torchaudio.transforms as T
import json
import os
from snac import SNAC
from peft import PeftModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Set up cache directories
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "hub")
DATASET_CACHE_DIR = os.path.join(CACHE_DIR, "datasets")

# Create cache directories if they don't exist
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

logger.info(f"Model cache directory: {MODEL_CACHE_DIR}")
logger.info(f"Dataset cache directory: {DATASET_CACHE_DIR}")


def download_with_progress(url, filename, desc=None):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    with (
        open(filename, "wb") as file,
        tqdm(
            desc=desc,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)


def check_disk_space(path, required_size_mb):
    """Check if there's enough disk space"""
    free_space = shutil.disk_usage(path).free
    required_space = required_size_mb * 1024 * 1024  # Convert MB to bytes
    return free_space >= required_space


# Load configuration from config.yaml
logger.info("Loading configuration from config.yaml...")
config_file = "config.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from config
logger.info("Extracting configuration parameters...")
dsn = config["TTS_dataset"]
model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

# Extract token configuration
token_config = config["token_config"]
TOKENISER_LENGTH = token_config["tokeniser_length"]
START_OF_TEXT = token_config["start_of_text"]
END_OF_TEXT = token_config["end_of_text"]
START_OF_SPEECH = token_config["start_of_speech"]
END_OF_SPEECH = token_config["end_of_speech"]
START_OF_HUMAN = token_config["start_of_human"]
END_OF_HUMAN = token_config["end_of_human"]
START_OF_AI = token_config["start_of_ai"]
END_OF_AI = token_config["end_of_ai"]
PAD_TOKEN = token_config["pad_token"]
AUDIO_TOKENS_START = token_config["audio_tokens_start"]
MAX_SEQUENCE_LENGTH = token_config["max_sequence_length"]

# Extract LoRA configuration
lora_config = config["lora_config"]
lora_rank = lora_config["rank"]
lora_alpha = lora_config["alpha"]
lora_dropout = lora_config["dropout"]

logger.info("Training configuration:")
logger.info(f"- Dataset: {dsn}")
logger.info(f"- Base model: {model_name}")
logger.info(f"- Epochs: {epochs}")
logger.info(f"- Batch size: {batch_size}")
logger.info(f"- Learning rate: {learning_rate}")
logger.info(f"- Max sequence length: {MAX_SEQUENCE_LENGTH}")

logger.info("LoRA configuration:")
logger.info(f"- Rank: {lora_rank}")
logger.info(f"- Alpha: {lora_alpha}")
logger.info(f"- Dropout: {lora_dropout}")

# Initialize tokenizer and base model
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, resume_download=True, cache_dir=MODEL_CACHE_DIR
)


def analyze_tokenizer_tokens(tokenizer):
    """Analyze the tokenizer's vocabulary for custom and special tokens"""
    logger.info("\nTokenizer Analysis:")

    # Get all tokens
    all_tokens = tokenizer.get_vocab()

    # Count special tokens
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
        # Sort custom tokens by their number
        custom_tokens.sort(key=lambda x: int(x.split("_")[-1].rstrip(">")))
        for token in custom_tokens[:10]:  # Show first 10 custom tokens
            logger.info(f"- {token} (ID: {all_tokens[token]})")
        if len(custom_tokens) > 10:
            logger.info(f"... and {len(custom_tokens) - 10} more custom tokens")


# Analyze tokenizer after initialization
analyze_tokenizer_tokens(tokenizer)

logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARNING)

# Set padding token to EOS token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Set padding token to EOS token")

logger.info("Loading base model (this might take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    resume_download=True,
    cache_dir=MODEL_CACHE_DIR,
)

# Enable gradient computation for all parameters
for param in model.parameters():
    param.requires_grad = True

# LoRA Configuration
logger.info("Configuring LoRA...")
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
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
    # Add llama.cpp specific settings
    init_lora_weights="gaussian",  # llama.cpp expects Gaussian initialization
    use_dora=False,  # Disable DoRA as it's not supported by llama.cpp
)

# Wrap the model with PEFT LoRA
logger.info("Wrapping model with LoRA...")
model = get_peft_model(model, lora_config)

# Ensure critical layers are trainable
logger.info("Ensuring critical layers are trainable...")
if hasattr(model.base_model.model, "lm_head"):
    for param in model.base_model.model.lm_head.parameters():
        param.requires_grad = True
    logger.info("Set lm_head.parameters().requires_grad = True")
elif hasattr(model, "lm_head"):
    for param in model.lm_head.parameters():
        param.requires_grad = True
    logger.info("Set lm_head.parameters().requires_grad = True (direct on model)")

if hasattr(model.base_model.model, "embed_tokens"):
    for param in model.base_model.model.embed_tokens.parameters():
        param.requires_grad = True
    logger.info("Set embed_tokens.parameters().requires_grad = True")
elif hasattr(model, "embed_tokens"):
    for param in model.embed_tokens.parameters():
        param.requires_grad = True
    logger.info("Set embed_tokens.parameters().requires_grad = True (direct on model)")

# Print trainable parameters info
trainable_params = 0
all_param = 0
for name, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        logger.info(
            f"Trainable parameter: {name}, dtype: {param.dtype}, device: {param.device}"
        )
logger.info(
    f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}%"
)

# Ensure model is in training mode and on the correct device
model.train()
if torch.cuda.is_available():
    model = model.cuda()
    logger.info("Model moved to CUDA device")

# Verify model state
logger.info(f"Model training mode: {model.training}")
logger.info(f"Model device: {next(model.parameters()).device}")


def download_file(url, local_path, chunk_size=8192):
    """Download a file with progress bar and proper error handling"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(local_path, "wb") as f,
            tqdm(
                desc=osp.basename(local_path),
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    size = f.write(chunk)
                    pbar.update(size)
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False


def download_dataset(dataset_name, cache_dir):
    """Download dataset files manually"""
    logger.info(f"Attempting to download dataset {dataset_name} manually...")

    # Create a temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the dataset files
        base_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main"
        files_to_download = ["README.md", "data/train-00000-of-00001.parquet"]

        for file in files_to_download:
            url = f"{base_url}/{file}"
            local_path = osp.join(temp_dir, file)

            # Create directory if it doesn't exist
            os.makedirs(osp.dirname(local_path), exist_ok=True)

            logger.info(f"Downloading {file}...")
            if download_file(url, local_path):
                # Move to cache directory
                cache_path = osp.join(cache_dir, dataset_name.replace("/", "--"), file)
                os.makedirs(osp.dirname(cache_path), exist_ok=True)
                shutil.move(local_path, cache_path)
                logger.info(f"Successfully downloaded {file}")
            else:
                raise Exception(f"Failed to download {file}")


def validate_raw_dataset(dataset) -> bool:
    """Validate the raw dataset structure and content"""
    logger.info("Validating raw dataset...")

    # Check required columns
    required_columns = {"audio", "text"}
    if not all(col in dataset.features for col in required_columns):
        missing = required_columns - set(dataset.features.keys())
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Validate a sample of examples
    sample_size = min(5, len(dataset))
    logger.info(f"Validating {sample_size} random examples...")

    for i, example in enumerate(dataset.select(range(sample_size))):
        # Validate text
        if not isinstance(example["text"], str):
            raise ValueError(
                f"Example {i}: text is not a string, got {type(example['text'])}"
            )
        if len(example["text"].strip()) == 0:
            raise ValueError(f"Example {i}: text is empty")

        # Validate audio
        if not isinstance(example["audio"], dict):
            raise ValueError(
                f"Example {i}: audio is not a dictionary, got {type(example['audio'])}"
            )

        required_audio_keys = {"array", "sampling_rate"}
        if not all(key in example["audio"] for key in required_audio_keys):
            missing = required_audio_keys - set(example["audio"].keys())
            raise ValueError(f"Example {i}: audio missing required keys: {missing}")

        # Validate audio array
        if not isinstance(example["audio"]["array"], np.ndarray):
            raise ValueError(
                f"Example {i}: audio array is not numpy array, got {type(example['audio']['array'])}"
            )
        if example["audio"]["array"].size == 0:
            raise ValueError(f"Example {i}: audio array is empty")

        # Validate sampling rate
        if not isinstance(example["audio"]["sampling_rate"], (int, float)):
            raise ValueError(
                f"Example {i}: sampling rate is not a number, got {type(example['audio']['sampling_rate'])}"
            )
        if example["audio"]["sampling_rate"] <= 0:
            raise ValueError(
                f"Example {i}: invalid sampling rate: {example['audio']['sampling_rate']}"
            )

    logger.info("Raw dataset validation passed!")
    return True


def validate_processed_dataset(dataset, tokenizer) -> bool:
    """Validate the processed dataset structure and content"""
    logger.info("Validating processed dataset...")

    # Check required columns
    required_columns = {"input_ids", "attention_mask", "labels"}
    if not all(col in dataset.features for col in required_columns):
        missing = required_columns - set(dataset.features.keys())
        raise ValueError(f"Processed dataset missing required columns: {missing}")

    # Validate a sample of examples
    sample_size = min(5, len(dataset))
    logger.info(f"Validating {sample_size} random processed examples...")

    for i, example in enumerate(dataset.select(range(sample_size))):
        # Validate input_ids
        if not isinstance(example["input_ids"], (list, np.ndarray)):
            raise ValueError(
                f"Example {i}: input_ids is not a list/array, got {type(example['input_ids'])}"
            )
        if len(example["input_ids"]) == 0:
            raise ValueError(f"Example {i}: input_ids is empty")

        # Validate attention_mask
        if not isinstance(example["attention_mask"], (list, np.ndarray)):
            raise ValueError(
                f"Example {i}: attention_mask is not a list/array, got {type(example['attention_mask'])}"
            )
        if len(example["attention_mask"]) != len(example["input_ids"]):
            raise ValueError(
                f"Example {i}: attention_mask length doesn't match input_ids"
            )

        # Validate labels
        if not isinstance(example["labels"], (list, np.ndarray)):
            raise ValueError(
                f"Example {i}: labels is not a list/array, got {type(example['labels'])}"
            )
        if len(example["labels"]) != len(example["input_ids"]):
            raise ValueError(f"Example {i}: labels length doesn't match input_ids")

        # Validate tokenization
        try:
            decoded = tokenizer.decode(example["input_ids"])
            if len(decoded.strip()) == 0:
                raise ValueError(f"Example {i}: decoded text is empty")
        except Exception as e:
            raise ValueError(f"Example {i}: failed to decode tokens: {str(e)}")

    logger.info("Processed dataset validation passed!")
    return True


def tokenise_audio(waveform, sample_rate, snac_model):
    """Tokenize audio using SNAC model"""
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0).to("cuda")
    with torch.inference_mode():
        codes = snac_model.encode(waveform)
    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
        all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))
    return all_codes


def remove_duplicate_frames(codes_list):
    """Remove duplicate frames from audio codes"""
    if len(codes_list) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")
    result = codes_list[:7]
    for i in range(7, len(codes_list), 7):
        current_first = codes_list[i]
        previous_first = result[-7]
        if current_first != previous_first:
            result.extend(codes_list[i : i + 7])
    return result


def process_example(example, snac_model, tokenizer):
    """Process a single example with SNAC tokenization"""
    # Tokenize audio using SNAC
    audio_codes = tokenise_audio(
        example["audio"]["array"], example["audio"]["sampling_rate"], snac_model
    )

    # Remove duplicate frames
    audio_codes = remove_duplicate_frames(audio_codes)

    # Tokenize text
    text_ids = tokenizer.encode(example["text"], add_special_tokens=True)
    text_ids.append(END_OF_TEXT)

    # Create input sequence
    input_ids = (
        [START_OF_HUMAN]
        + text_ids
        + [END_OF_HUMAN]
        + [START_OF_AI]
        + [START_OF_SPEECH]
        + audio_codes
        + [END_OF_SPEECH]
        + [END_OF_AI]
    )

    # Truncate if too long
    if len(input_ids) > MAX_SEQUENCE_LENGTH:
        input_ids = input_ids[:MAX_SEQUENCE_LENGTH]

    # Pad if too short
    if len(input_ids) < MAX_SEQUENCE_LENGTH:
        input_ids = input_ids + [PAD_TOKEN] * (MAX_SEQUENCE_LENGTH - len(input_ids))

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": input_ids,
    }


def preprocess_dataset(dataset, tokenizer):
    """Preprocess the dataset to match the model's expected input format"""
    logger.info("Preprocessing dataset...")

    # Validate raw dataset first
    validate_raw_dataset(dataset)

    # Load SNAC model for audio tokenization
    logger.info("Loading SNAC model for audio tokenization...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")

    # Create a partial function with the required parameters
    from functools import partial

    process_fn = partial(process_example, snac_model=snac_model, tokenizer=tokenizer)

    # Process the dataset
    logger.info("Processing dataset with SNAC tokenization...")
    processed_dataset = dataset.map(
        process_fn, remove_columns=["audio", "text"], desc="Processing dataset"
    )

    # Validate processed dataset
    validate_processed_dataset(processed_dataset, tokenizer)

    logger.info(f"Processed dataset features: {processed_dataset.features}")
    return processed_dataset


# Load your preprocessed dataset
logger.info(f"Loading dataset {dsn}...")
try:
    # Check disk space before downloading (328MB for the dataset)
    if not check_disk_space(DATASET_CACHE_DIR, 328):
        logger.error(f"Not enough disk space in {DATASET_CACHE_DIR}. Required: 328MB")
        raise OSError("Insufficient disk space")

    # Try to download the dataset manually first
    try:
        download_dataset(dsn, DATASET_CACHE_DIR)
    except Exception as e:
        logger.warning(f"Manual download failed: {str(e)}")
        logger.info("Falling back to automatic download...")

    # Load the dataset
    logger.info("Loading dataset from downloaded files...")
    ds = load_dataset(dsn, split="train", cache_dir=DATASET_CACHE_DIR)

    # Verify dataset loaded correctly
    if ds is None or len(ds) == 0:
        raise ValueError("Dataset loaded but appears to be empty")

    logger.info(f"Successfully loaded dataset with {len(ds)} examples")
    logger.info(f"Dataset features: {ds.features}")

    # Preprocess the dataset
    ds = preprocess_dataset(ds, tokenizer)

except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    logger.error("Please check your internet connection and disk space")
    raise

# Initialize Weights & Biases
logger.info("Initializing Weights & Biases...")
wandb.init(project=project_name, name=run_name)

# Training Arguments
logger.info("Setting up training arguments...")
training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_steps=1,
    bf16=True,  # Enable bfloat16 mixed precision
    output_dir=f"./{base_repo_id}",
    report_to="wandb",
    save_steps=save_steps,
    remove_unused_columns=False,
    learning_rate=learning_rate,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    gradient_checkpointing=True,
    optim="adamw_torch",
    max_grad_norm=1.0,
    dataloader_pin_memory=True,
    dataloader_num_workers=number_processes,
    # # Add these for better mixed precision training
    # fp16=False,  # Disable fp16 since we're using bf16
    # bf16_full_eval=True,  # Use bf16 for evaluation
    # torch_compile=True,  # Enable torch.compile for better performance
)


class TrainingProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_log_time = None
        self.total_steps = None
        self.current_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.training_started = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.total_steps = state.max_steps
        logger.info(f"Starting training for {self.total_steps} total steps")
        logger.info(f"Training will run for {args.num_train_epochs} epochs")
        self.training_started = True

    def on_step_end(self, args, state, control, **kwargs):
        self.current_step = state.global_step

        # Log every 10 steps or if 30 seconds have passed
        current_time = time.time()
        if (self.current_step % 10 == 0) or (current_time - self.last_log_time > 30):
            elapsed = current_time - self.start_time
            steps_per_second = self.current_step / elapsed
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps / steps_per_second if steps_per_second > 0 else 0

            # Calculate progress percentage
            progress = (self.current_step / self.total_steps) * 100

            # Get current learning rate
            current_lr = kwargs.get("optimizer").param_groups[0]["lr"]

            # Get current loss if available
            current_loss = (
                state.log_history[-1].get("loss", "N/A") if state.log_history else "N/A"
            )

            # Format loss string based on type
            if isinstance(current_loss, float):
                self.best_loss = min(self.best_loss, current_loss)
                loss_str = f"{current_loss:.4f}"
                best_loss_str = f"{self.best_loss:.4f}"
            else:
                loss_str = str(current_loss)
                best_loss_str = f"{self.best_loss:.4f}"

            logger.info(
                f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%) | "
                f"Loss: {loss_str} | Best Loss: {best_loss_str} | "
                f"LR: {current_lr:.2e} | Speed: {steps_per_second:.1f} steps/s | "
                f"ETA: {datetime.fromtimestamp(current_time + eta).strftime('%H:%M:%S')}"
            )
            self.last_log_time = current_time

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch += 1
        logger.info(f"Completed epoch {self.epoch}/{args.num_train_epochs}")

        # Log epoch statistics
        if state.log_history:
            epoch_loss = state.log_history[-1].get("loss", "N/A")
            logger.info(f"Epoch {self.epoch} final loss: {epoch_loss}")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time / 3600:.2f} hours")
        logger.info(
            f"Final loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}"
        )
        logger.info(f"Best loss achieved: {self.best_loss:.4f}")


# Initialize and run the Trainer
logger.info("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # Pad to multiple of 8 for better performance
    ),
    callbacks=[TrainingProgressCallback()],
)

logger.info("Starting training...")
logger.info(
    f"Training will be logged to Weights & Biases project: {project_name}, run: {run_name}"
)
trainer.train()

# Save the LoRA adapter separately
logger.info("Saving LoRA adapter...")
model.save_pretrained(f"./{base_repo_id}/lora_adapter")
tokenizer.save_pretrained(f"./{base_repo_id}/lora_adapter")

# Convert to llama.cpp format
logger.info("Converting LoRA adapter to llama.cpp format...")


# Load the saved adapter
adapter_path = f"./{base_repo_id}/lora_adapter"
model = PeftModel.from_pretrained(model, adapter_path)

# Convert and save in llama.cpp format
output_path = f"./{base_repo_id}/lora_adapter_llamacpp.bin"

# Ensure the model is in evaluation mode for conversion
model.eval()

# Save in llama.cpp format with specific settings
model.save_pretrained(
    output_path,
    safe_serialization=False,  # llama.cpp expects raw binary format
    # max_shard_size="500MB",    # Split into manageable chunks
    torch_dtype=torch.float16,  # Use FP16 for better compatibility
)

# Create a metadata file for llama.cpp
metadata = {
    "lora_rank": lora_rank,
    "lora_alpha": lora_alpha,
    "target_modules": list(
        lora_config.target_modules
    ),  # Convert set to list for JSON serialization
    "base_model": model_name,
    "version": "1.0",
}

with open(f"./{base_repo_id}/lora_adapter_llamacpp.json", "w") as f:
    json.dump(metadata, f, indent=2)

logger.info("\nTraining complete! To use this LoRA adapter with llama.cpp:")
logger.info(f"1. Use the adapter file: {output_path}")
logger.info(f"2. Metadata file: {base_repo_id}/lora_adapter_llamacpp.json")
logger.info("3. Add --lora flag to llama.cpp server command")
logger.info("4. Example: --lora /path/to/lora_adapter_llamacpp.bin")
logger.info(
    "\nNote: Make sure your llama.cpp server is built with LoRA support enabled"
)
