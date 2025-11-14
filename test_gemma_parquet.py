#!/usr/bin/env python3
"""
Parquet Testing Script for Instruction-Tuned Gemma Model
Processes parquet files with text normalization and saves predictions
"""

import os
import sys
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def normalize_quotes(text):
    """
    Normalize smart/curly quotes to straight ASCII quotes
    This ensures consistent tokenization regardless of input format
    """
    # Smart double quotes to straight
    text = text.replace('"', '"')  # U+8220 LEFT DOUBLE QUOTATION MARK
    text = text.replace('"', '"')  # U+8221 RIGHT DOUBLE QUOTATION MARK
    
    # Smart single quotes/apostrophes to straight
    text = text.replace(''', "'")  # U+8217 RIGHT SINGLE QUOTATION MARK
    text = text.replace(''', "'")  # U+8218 LEFT SINGLE QUOTATION MARK
    
    # Dashes
    text = text.replace('–', '-')  # U+2013 EN DASH
    text = text.replace('—', '-')  # U+2014 EM DASH
    
    return text


class ParquetTester:
    """Process parquet files with normalized text predictions"""
    
    def __init__(self, config_path, model_path):
        self.config_path = config_path
        self.model_path = model_path
        self.config = self.load_config()
        self.device = None
        self.tokenizer = None
        self.model = None
        
    def load_config(self):
        """Load configuration from YAML"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_device(self):
        """Setup GPU"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['gpu']['graphic_card']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
    
    def load_model(self):
        """Load trained model and tokenizer"""
        print(f"\n🤖 Loading model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='right'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Tokenizer loaded")
        
        # Load base model with quantization
        base_model_id = self.config['model']['hugging_face_model_id']
        print(f"🔧 Loading base model: {base_model_id}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config['quantization']['load_in_4bit'],
            bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype'])
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        # Load LoRA adapter
        print("🔌 Loading LoRA adapter...")
        # self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model = base_model
        self.model.eval()
        
        print("✅ Model loaded successfully\n")
    
    def create_prompt(self, text):
        """Create instruction prompt (same as training)"""
        prompt_template = self.config['instruction']['prompt_template']
        text = re.sub(r'\n', ' ', text.strip())
        prompt = prompt_template.format(text=text)
        return prompt
    
    def predict_single(self, text, normalize=True):
        """
        Predict class for a single text
        
        Args:
            text: Input text
            normalize: Whether to normalize quotes before prediction
        
        Returns:
            predicted_class (0 or 1), ai_probability (float), confidence (float), 
            answer_token (str), normalized_text (str)
        """
        # Normalize text if requested
        normalized_text = normalize_quotes(text) if normalize else text
        
        # Create prompt
        prompt = self.create_prompt(normalized_text)


        # print('='*80)
        # print(prompt[:200])
        # print('='*80)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(self.device)
        
        # Get token IDs for Yes/No
        yes_tokens = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens = self.tokenizer.encode("No", add_special_tokens=False)
        yes_token_id = yes_tokens[0]
        no_token_id = no_tokens[0]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )
        
        generated_ids = outputs.sequences[0]
        input_length = inputs.input_ids.shape[1]
        generated_text = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)
        # print(f"\n{'='*80}")
        # print(f"DEBUG - Full Generated Output: '{generated_text}'")
        # print(f"{'='*80}\n")

        # Get logits for Yes/No tokens
        first_token_logits = outputs.scores[0][0]
        yes_score = first_token_logits[yes_token_id].item()
        no_score = first_token_logits[no_token_id].item()
        

        # Calculate probabilities
        probs = F.softmax(torch.tensor([no_score, yes_score]), dim=0)
        yes_prob = probs[1].item()  # AI probability

        # print('='*80)
        # print("PROBS", probs)
        # print("YES PROB", yes_prob)
        # print("NO PROB", 1 - yes_prob)
        # print('='*80)

        # Predict
        predicted_class = 1 if yes_score > no_score else 0
        confidence = yes_prob if predicted_class == 1 else (1 - yes_prob)
        answer_token = "Yes" if predicted_class == 1 else "No"

        return predicted_class, yes_prob, confidence, answer_token, normalized_text
    
    def process_parquet(self, parquet_path, normalize=True, threshold=0.5):
        """
        Process all samples in a parquet file
        
        Args:
            parquet_path: Path to input parquet file
            normalize: Whether to normalize quotes
            threshold: Classification threshold
        
        Returns:
            DataFrame with predictions
        """
        print("="*80)
        print("PARQUET PROCESSING MODE")
        print("="*80)
        print(f"Input file: {parquet_path}")
        print(f"Text normalization: {'ENABLED' if normalize else 'DISABLED'}")
        print(f"Classification threshold: {threshold}")
        print("="*80)
        
        # Load parquet file
        df = pd.read_parquet(parquet_path)
        print(f"\n✅ Loaded {len(df):,} samples from parquet file")
        
        # Check if labels exist
        has_labels = 'label' in df.columns
        if has_labels:
            print(f"📊 Dataset has labels - accuracy will be calculated")
        else:
            print(f"⚠️  No labels found - predictions only")
        
        # Process each sample
        results = []
        correct_count = 0
        total_with_labels = 0
        
        print(f"\n🔄 Processing samples...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            text = row['text']
            true_label = row.get('label', None)
            

              # Run prediction
            pred_class, ai_prob, confidence, answer, normalized_text = self.predict_single(
                text, 
                normalize=normalize
            )

      
            # Calculate if correct
            is_correct = None
            if true_label is not None:
                total_with_labels += 1
                is_correct = (pred_class == true_label)
                if is_correct:
                    correct_count += 1
            

            # Store result
            result = {
                'text': text,
                'normalized_text': normalized_text if normalize else None,
                'text_changed': (text != normalized_text) if normalize else False,
                'true_label': true_label,
                'predicted_label': pred_class,
                'ai_probability': ai_prob,
                'confidence': confidence,
                'answer_token': answer,
                'is_correct': is_correct,
            }
            results.append(result)

            # print('='*80)
            # print("Predicted label: ", pred_class)
            
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Print summary
        print("\n" + "="*80)
        print("PROCESSING COMPLETE - SUMMARY")
        print("="*80)
        print(f"Total samples processed: {len(results_df):,}")
        
        if total_with_labels > 0:
            accuracy = correct_count / total_with_labels
            print(f"Samples with labels: {total_with_labels:,}")
            print(f"Correct predictions: {correct_count:,}/{total_with_labels:,} ({accuracy*100:.2f}%)")
        
        # Count predictions
        ai_count = (results_df['predicted_label'] == 1).sum()
        nonai_count = len(results_df) - ai_count
        print(f"\nPrediction distribution:")
        print(f"  Predicted AI: {ai_count:,}/{len(results_df):,} ({ai_count/len(results_df)*100:.1f}%)")
        print(f"  Predicted Non-AI: {nonai_count:,}/{len(results_df):,} ({nonai_count/len(results_df)*100:.1f}%)")
        
        # Average metrics
        avg_ai_prob = results_df['ai_probability'].mean()
        avg_confidence = results_df['confidence'].mean()
        print(f"\nAverage AI probability: {avg_ai_prob:.4f}")
        print(f"Average confidence: {avg_confidence:.4f}")
        
        # Text normalization stats
        if normalize:
            changed_count = results_df['text_changed'].sum()
            print(f"\nText normalization:")
            print(f"  Samples with changes: {changed_count:,}/{len(results_df):,} ({changed_count/len(results_df)*100:.1f}%)")
        
        print("="*80)
        
        return results_df
    
    def save_predictions(self, results_df, output_path=None, parquet_name="predictions"):
        """
        Save predictions to parquet file
        
        Args:
            results_df: DataFrame with predictions
            output_path: Output directory (default: outputs/parquet_predictions)
            parquet_name: Base name for output file
        """
        if output_path is None:
            output_path = "outputs/parquet_predictions"
        
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions parquet
        output_file = os.path.join(output_path, f"{parquet_name}_{timestamp}.parquet")
        results_df.to_parquet(output_file, index=False)
        print(f"\n💾 Saved predictions to: {output_file}")
        
        # Also save CSV for easy viewing
        csv_file = os.path.join(output_path, f"{parquet_name}_{timestamp}.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"📊 Saved CSV to: {csv_file}")
        
        return output_file


def main():
    """Main entry point"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process parquet file with text normalization and predictions'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input parquet file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/instruction_training_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='outputs/checkpoints/gemma3_20k_balanced_it_v2/final_model',
        help='Path to trained model'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Normalize quotes before prediction (default: True)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_false',
        dest='normalize',
        help='Disable quote normalization'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/parquet_predictions',
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='predictions',
        help='Base name for output files'
    )
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"PARQUET PREDICTION PROCESSING")
    print(f"{'='*80}")
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Normalize: {args.normalize}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output}")
    print(f"{'='*80}\n")
    
    # Initialize tester
    tester = ParquetTester(args.config, args.model)
    tester.setup_device()
    tester.load_model()
    
    # Process parquet file
    results_df = tester.process_parquet(
        args.input, 
        normalize=args.normalize,
        threshold=args.threshold
    )
    
    # Save predictions
    tester.save_predictions(results_df, args.output, args.name)
    
    print("\n✅ Processing complete!")


if __name__ == "__main__":
    main()

