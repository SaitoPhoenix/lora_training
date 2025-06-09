from datasets import load_dataset
import numpy as np
import logging
import sys
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
from snac import SNAC
import torch
import torchaudio.transforms as T

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml"""
    logger.info("Loading configuration from config.yaml...")
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


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


def process_example(example, snac_model, tokenizer, config):
    """Process a single example and return sequence length"""
    # Tokenize audio using SNAC
    audio_codes = tokenise_audio(
        example["audio"]["array"], example["audio"]["sampling_rate"], snac_model
    )

    # Remove duplicate frames
    audio_codes = remove_duplicate_frames(audio_codes)

    # Tokenize text
    text_ids = tokenizer.encode(example["text"], add_special_tokens=True)
    text_ids.append(config["token_config"]["end_of_text"])

    # Create input sequence
    input_ids = (
        [config["token_config"]["start_of_human"]]
        + text_ids
        + [config["token_config"]["end_of_human"]]
        + [config["token_config"]["start_of_ai"]]
        + [config["token_config"]["start_of_speech"]]
        + audio_codes
        + [config["token_config"]["end_of_speech"]]
        + [config["token_config"]["end_of_ai"]]
    )

    return len(input_ids)


def analyze_sequences():
    """Analyze sequence lengths in the dataset"""
    # Load configuration
    config = load_config()

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Load SNAC model
    logger.info("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")

    # Load dataset
    logger.info(f"Loading dataset {config['TTS_dataset']}...")
    dataset = load_dataset(config["TTS_dataset"], split="train")

    # Process examples and collect lengths
    logger.info("Processing examples to analyze sequence lengths...")
    lengths = []
    for example in tqdm(dataset):
        try:
            length = process_example(example, snac_model, tokenizer, config)
            lengths.append(length)
        except Exception as e:
            logger.error(f"Error processing example: {str(e)}")
            continue

    # Calculate statistics
    lengths = np.array(lengths)
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    avg_length = np.mean(lengths)
    median_length = np.median(lengths)
    std_length = np.std(lengths)

    # Print results
    logger.info("\nSequence Length Analysis:")
    logger.info(f"Number of sequences analyzed: {len(lengths)}")
    logger.info(f"Minimum length: {min_length}")
    logger.info(f"Maximum length: {max_length}")
    logger.info(f"Average length: {avg_length:.2f}")
    logger.info(f"Median length: {median_length}")
    logger.info(f"Standard deviation: {std_length:.2f}")

    # Print distribution percentiles
    percentiles = [50, 75, 90, 95, 99]
    logger.info("\nLength Distribution Percentiles:")
    for p in percentiles:
        logger.info(f"{p}th percentile: {np.percentile(lengths, p):.2f}")

    # Print warning if max length exceeds MAX_SEQUENCE_LENGTH
    if max_length > config["token_config"]["max_sequence_length"]:
        logger.warning(
            f"\nWARNING: Maximum sequence length ({max_length}) exceeds MAX_SEQUENCE_LENGTH ({config['token_config']['max_sequence_length']})"
        )
        logger.warning("Some sequences will be truncated during training")


if __name__ == "__main__":
    analyze_sequences()
