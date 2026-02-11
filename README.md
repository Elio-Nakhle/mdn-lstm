# MDN-LSTM: Mixed Density Network with LSTM

![CI](../../workflows/CI/badge.svg)

A Python project implementing a Mixed Density Network (MDN) with LSTM for probabilistic sequence prediction.

## Features

- **MDN-LSTM Model**: Combines LSTM temporal feature extraction with Gaussian Mixture Model output
- **Configurable Architecture**: Easy YAML-based configuration
- **CLI Interface**: Command-line tools for training, validation, and inference
- **PyInvoke Tasks**: Convenient task runner for common operations
- **PDM Package Management**: Modern Python dependency management

## Installation

### Prerequisites

- Python 3.10+
- [PDM](https://pdm-project.org/) package manager

### Setup

```bash
# Install PDM if you don't have it
pip install pdm

# Install project dependencies
pdm install --dev

# Or using invoke
pdm run invoke install
```

## Usage

### Configuration

Generate a default configuration file:

```bash
pdm run invoke init-config
# or
pdm run python -m mdn_lstm.cli init-config --output config.yaml
```

### Training

Train the model:

```bash
# Using invoke
pdm run invoke train --config config.yaml --epochs 10000

# Or using CLI directly
pdm run python -m mdn_lstm.cli train --config config.yaml --epochs 10000 --data data.csv --output models/
```

### Validation

Validate a trained model:

```bash
pdm run invoke validate --model-path models/mdn_lstm_model.pt --config config.yaml

# Or using CLI
pdm run python -m mdn_lstm.cli validate --model-path models/mdn_lstm_model.pt
```

### Inference

Run inference on a trained model:

```bash
# Single prediction
pdm run invoke infer --model-path models/mdn_lstm_model.pt --input-data "12,22,28,30,31,4,11,31"

# With multiple samples
pdm run python -m mdn_lstm.cli infer --model-path models/mdn_lstm_model.pt --input "12,22,28,30,31,4,11,31" --n-samples 10
```

### Export to ONNX

```bash
pdm run invoke export-model --model-path models/mdn_lstm_model.pt --output model.onnx
```

## Project Structure

```
mdn-lstm/
├── pyproject.toml          # PDM configuration and dependencies
├── tasks.py                # PyInvoke task definitions
├── config.yaml             # Default configuration
├── src/
│   └── mdn_lstm/
│       ├── __init__.py
│       ├── cli.py          # Command-line interface
│       ├── config.py       # Configuration dataclasses
│       ├── models/
│       │   ├── __init__.py
│       │   └── mdn.py      # MDN and MDN-LSTM model implementations
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset.py  # Data loading and preprocessing
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py  # Training loop and utilities
│       └── inference/
│           ├── __init__.py
│           └── predictor.py  # Inference utilities
└── tests/                  # Test files
```

## Available Invoke Tasks

```bash
pdm run invoke --list

# Available tasks:
#   check        Run all checks (lint, typecheck, test)
#   clean        Clean build artifacts and caches
#   export-model Export model to ONNX format
#   format-code  Format code with ruff
#   infer        Run inference with the trained model
#   init-config  Generate a default configuration file
#   install      Install project dependencies using PDM
#   lint         Run linters (ruff)
#   test         Run tests
#   train        Train the MDN-LSTM model
#   typecheck    Run type checking with mypy
#   validate     Validate the trained model
```

## Configuration Options

The configuration file (`config.yaml`) supports the following options:

### Model Configuration

```yaml
model:
  n_input: 8           # Number of input features
  n_output: 7          # Number of output dimensions
  n_hidden: 2          # LSTM hidden size
  n_gaussians: 3       # Number of Gaussian mixture components
  num_lstm_layers: 1   # Number of LSTM layers
  bidirectional: false # Use bidirectional LSTM
```

### Training Configuration

```yaml
training:
  epochs: 84000                    # Maximum training epochs
  learning_rate: 0.001             # Learning rate
  weight_decay: 0.001              # L2 regularization in optimizer
  batch_size: 32                   # Batch size
  train_split: 0.8                 # Train/val split ratio
  use_scheduler: false             # Use learning rate scheduler
  scheduler_patience: 100          # LR scheduler patience
  scheduler_factor: 0.5            # LR reduction factor
  early_stopping: true             # Enable early stopping
  early_stopping_patience: 2000    # Early stopping patience
  early_stopping_delta: 0.0005     # Minimum improvement delta
  l2_lambda: 0.001                 # L2 regularization in loss
```

### Data Configuration

```yaml
data:
  data_path: data.csv              # Training data file
  model_save_path: models          # Model output directory
  checkpoint_path: checkpoints     # Checkpoint directory
```

## Development

### Running Tests

```bash
pdm run invoke test
pdm run invoke test --coverage
```

### Linting and Formatting

```bash
pdm run invoke lint
pdm run invoke lint --fix
pdm run invoke format-code
```

### Type Checking

```bash
pdm run invoke typecheck
```

## License

Copyright (c) 2026 All Rights Reserved
