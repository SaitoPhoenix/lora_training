from datasets import load_dataset
import torch
import torchaudio.transforms as T
from snac import SNAC
from utils.logger import get_logger as logger
from utils.downloader import check_disk_space, download_dataset
from utils.validator import validate_raw_dataset, validate_processed_dataset


class DatasetManager:
    def __init__(self, config_manager):
        self.config = config_manager
        self.dataset = None
        self.snac_model = None
        self._initialize_snac()

    def _initialize_snac(self):
        """Initialize SNAC model for audio tokenization"""
        logger.info("Loading SNAC model for audio tokenization...")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model = self.snac_model.to("cuda")

    def _tokenise_audio(self, waveform, sample_rate):
        """Tokenize audio using SNAC model"""
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
        waveform = resample_transform(waveform)
        waveform = waveform.unsqueeze(0).to("cuda")

        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(
                codes[0][0][i].item()
                + self.config.get("token_config.audio_tokens_start")
            )
            all_codes.append(
                codes[1][0][2 * i].item()
                + self.config.get("token_config.audio_tokens_start")
                + 4096
            )
            all_codes.append(
                codes[2][0][4 * i].item()
                + self.config.get("token_config.audio_tokens_start")
                + (2 * 4096)
            )
            all_codes.append(
                codes[2][0][(4 * i) + 1].item()
                + self.config.get("token_config.audio_tokens_start")
                + (3 * 4096)
            )
            all_codes.append(
                codes[1][0][(2 * i) + 1].item()
                + self.config.get("token_config.audio_tokens_start")
                + (4 * 4096)
            )
            all_codes.append(
                codes[2][0][(4 * i) + 2].item()
                + self.config.get("token_config.audio_tokens_start")
                + (5 * 4096)
            )
            all_codes.append(
                codes[2][0][(4 * i) + 3].item()
                + self.config.get("token_config.audio_tokens_start")
                + (6 * 4096)
            )
        return all_codes

    def _remove_duplicate_frames(self, codes_list):
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

    def _process_example(self, example, tokenizer):
        """Process a single example with SNAC tokenization"""
        # Tokenize audio using SNAC
        audio_codes = self._tokenise_audio(
            example["audio"]["array"], example["audio"]["sampling_rate"]
        )

        # Remove duplicate frames
        audio_codes = self._remove_duplicate_frames(audio_codes)

        # Tokenize text
        text_ids = tokenizer.encode(example["text"], add_special_tokens=True)
        text_ids.append(self.config.get("token_config.end_of_text"))

        # Create input sequence
        input_ids = (
            [self.config.get("token_config.start_of_human")]
            + text_ids
            + [self.config.get("token_config.end_of_human")]
            + [self.config.get("token_config.start_of_ai")]
            + [self.config.get("token_config.start_of_speech")]
            + audio_codes
            + [self.config.get("token_config.end_of_speech")]
            + [self.config.get("token_config.end_of_ai")]
        )

        # Truncate if too long
        if len(input_ids) > self.config.get("token_config.max_sequence_length"):
            input_ids = input_ids[: self.config.get("token_config.max_sequence_length")]

        # Pad if too short
        if len(input_ids) < self.config.get("token_config.max_sequence_length"):
            input_ids = input_ids + [self.config.get("token_config.pad_token")] * (
                self.config.get("token_config.max_sequence_length") - len(input_ids)
            )

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": input_ids,
        }

    def load_and_preprocess(self, tokenizer):
        """Load and preprocess the dataset"""
        logger.info(f"Loading dataset {self.config.get('dataset.TTS_dataset')}...")
        try:
            # Check disk space before downloading
            if not check_disk_space(self.config.dataset_cache_dir, 328):
                logger.error(
                    f"Not enough disk space in {self.config.dataset_cache_dir}. Required: 328MB"
                )
                raise OSError("Insufficient disk space")

            # Try to download the dataset manually first
            try:
                download_dataset(
                    self.config.get("dataset.TTS_dataset"),
                    self.config.dataset_cache_dir,
                )
            except Exception as e:
                logger.warning(f"Manual download failed: {str(e)}")
                logger.info("Falling back to automatic download...")

            # Load the dataset
            logger.info("Loading dataset from downloaded files...")
            self.dataset = load_dataset(
                self.config.get("dataset.TTS_dataset"),
                split="train",
                cache_dir=self.config.dataset_cache_dir,
            )

            # Verify dataset loaded correctly
            if self.dataset is None or len(self.dataset) == 0:
                raise ValueError("Dataset loaded but appears to be empty")

            logger.info(
                f"Successfully loaded dataset with {len(self.dataset)} examples"
            )
            logger.info(f"Dataset features: {self.dataset.features}")

            # Validate raw dataset
            validate_raw_dataset(self.dataset)

            # Process the dataset
            logger.info("Processing dataset with SNAC tokenization...")
            self.dataset = self.dataset.map(
                lambda x: self._process_example(x, tokenizer),
                remove_columns=["audio", "text"],
                desc="Processing dataset",
            )

            # Validate processed dataset
            validate_processed_dataset(self.dataset, tokenizer)

            logger.info(f"Processed dataset features: {self.dataset.features}")
            return self.dataset

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            logger.error("Please check your internet connection and disk space")
            raise
