#!/usr/bin/env python3
"""
CLI script to train the ModernBERT binary classifier
Usage: python scripts/train_modernbert.py --config config/modernbert_training_config.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.modernbert_trainer import ModernBERTTrainer


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Train ModernBERT Binary Classifier for Content Filtering'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/modernbert_training_config.yaml',
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Initialize and run trainer
    print("\n" + "="*80)
    print("Content Filtering LV1 - ModernBERT Binary Classifier")
    print("ModernBERT-large with Full Fine-tuning")
    print("="*80 + "\n")
    
    trainer = ModernBERTTrainer(config)
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()

