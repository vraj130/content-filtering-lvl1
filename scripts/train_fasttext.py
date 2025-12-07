#!/usr/bin/env python3
"""
CLI script to train the FastText binary classifier.
Usage: python scripts/train_fasttext.py --config config/fasttext_training_config.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.fasttext_trainer import FastTextTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Train FastText Binary Classifier for Content Filtering'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/fasttext_training_config.yaml',
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Initialize and run trainer
    print("\n" + "="*80)
    print("Content Filtering - FastText Binary Classifier")
    print("AI vs Non-AI Text Classification")
    print("="*80 + "\n")
    
    trainer = FastTextTrainer(config)
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()

