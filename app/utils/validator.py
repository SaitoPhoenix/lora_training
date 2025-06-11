import numpy as np
from utils.logger import get_logger as logger


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
