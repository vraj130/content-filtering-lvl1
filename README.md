# Content Filtering LV1 - Stage 1 Binary Classifier

High-recall binary classifier for AI-generated content detection (Stage 1 of two-stage filtering pipeline)

## Overview

This project implements the first stage of a two-stage content filtering pipeline designed to identify AI-generated content with high recall. The classifier uses **Gemma-3-1b** fine-tuned with **LoRA** (Low-Rank Adaptation) for efficient training on GPU.


### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have your dataset in the correct format:
   - Training data: `/workspace/unified_data/gemma_format/train.csv`
   - Validation data: `/workspace/unified_data/gemma_format/val.csv`
   - Test data: `/workspace/unified_data/gemma_format/test.csv`

### Training the Model

Run training with the default configuration:

```bash
python scripts/train.py --config config/training_config.yaml
```

Or simply:

```bash
python scripts/train.py
```

### Using Docker

Build and run with Docker:

```bash
docker build -t content-filtering-lv1 .
docker run --gpus all -v /path/to/data:/workspace/unified_data content-filtering-lv1 python scripts/train.py
```

## Configuration

All training parameters are configured in `config/training_config.yaml`:

- **Model settings**: Model ID, max length, attention implementation
- **LoRA parameters**: r=64, alpha=32, dropout=0.1
- **Training settings**: Batch size, learning rate, epochs, etc.
- **Class weights**: Balanced weighting for imbalanced datasets
- **Threshold **: Target recall for AI class

Edit this file to customize the training process.

## Training Output

After training completes, you'll find:

- **Trained model**: `models/gemma3_binary_model_final/`
- **Optimal threshold**: `models/gemma3_binary_model_final/optimal_threshold.json`
- **Checkpoints**: `outputs/checkpoints/gemma3_binary_classifier/`
- **TensorBoard logs**: `outputs/tensorboard/gemma3_binary_classifier/`


## Monitoring
Training progress is logged to:

- **TensorBoard**: `tensorboard --logdir outputs/tensorboard/`
- **Weights & Biases**: Configure in `training_config.yaml`
