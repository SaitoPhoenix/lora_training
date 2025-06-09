# LoRA Adapter Trainer

A powerful and efficient LoRA (Low-Rank Adaptation) adapter trainer built using UV for environment management and project setup.

## Overview

This project provides a streamlined way to train LoRA adapters for language models, with a focus on ease of use and configuration through a YAML-based setup.

## Prerequisites

- Python 3.8 or higher
- UV package manager
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd lora_training
```

2. Set up the environment using UV:
```bash
uv venv
source .venv/bin/activate #or .venv/Scripts/activate if using Windows
uv sync
```

## Configuration

All training configurations can be modified in the `app/config/config.yaml` file. Here are the key configuration options:

```yaml
# Model Configuration
model:
  base_model: "path/to/base/model"
  adapter_name: "your_adapter_name"

# Training Parameters
training:
  learning_rate: 1e-4
  batch_size: 8
  num_epochs: 3
  gradient_accumulation_steps: 4

# LoRA Specific Settings
lora:
  r: 8
  alpha: 16
  dropout: 0.1
  target_modules: ["q_proj", "v_proj"]

# Data Configuration
data:
  train_file: "path/to/training/data"
  validation_file: "path/to/validation/data"
  max_length: 512
```

## Usage

1. Update the `app/config/config.yaml` file with your desired settings
2. Run the training script:
```bash
uv run app/train_lora.py
```

## Project Structure

```
lora_training/
├── app/
│   ├── config/
│   │   └── config.yaml    # Main configuration file
│   └── train_lora.py      # Training script
├── pyproject.toml         # Project dependencies and metadata
└── README.md             # This file
```

## Features

- UV-based environment management
- YAML-based configuration
- Flexible LoRA parameter tuning
- GPU acceleration support
- Progress tracking and logging

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Contact

[Your contact information]
