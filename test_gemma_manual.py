#!/usr/bin/env python3
"""
Manual Testing Script for Instruction-Tuned Gemma Model
Allows quick testing on custom text samples defined directly in code
"""

import os
import sys
import re
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class ManualTester:
    """Quick tester for manual text samples"""
    
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
        base_model_id = self.config['model']['hugging_face_model_id']
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
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
        self.model = base_model
        self.model.eval()
        
        print("✅ Model loaded successfully\n")

    @staticmethod
    def _single_token_variants(tok, word: str):
        cands = set()
        for pref in ["", " ", "\n"]:
            for w in [word, word.lower()]:
                ids = tok.encode(pref + w, add_special_tokens=False)
                if len(ids) == 1:
                    cands.add(ids[0])
        # Fallback: if nothing single-token, take first piece of " word"
        if not cands:
            ids = tok.encode(" " + word, add_special_tokens=False)
            if ids:
                cands.add(ids[0])
        return sorted(cands)
    
    def create_prompt(self, text):
        """Create instruction prompt (same as training)"""

        prompt_template = self.config['instruction']['prompt_template']
        text = re.sub(r'\n', ' ', text.strip())
        prompt = prompt_template.format(text=text)
        return prompt
    
    def predict_single(self, text):
        """
        Predict class for a single text
        
        Returns:
            predicted_class (0 or 1), ai_probability (float), confidence (float), answer_token (str)
        """
        # Create prompt
        prompt = self.create_prompt(text)

        print('='*80)
        print(prompt)
        print('='*80)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(self.device)
        
        # Get token IDs for Yes/No
        yes_ids = self._single_token_variants(self.tokenizer, self.config['instruction']['answer_yes'])
        no_ids  = self._single_token_variants(self.tokenizer, self.config['instruction']['answer_no'])
        
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
        
        # DEBUG: Decode and print the full generated output
        generated_ids = outputs.sequences[0]
        input_length = inputs.input_ids.shape[1]
        generated_text = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)
        print(f"\n{'='*80}")
        print(f"DEBUG - Full Generated Output: '{generated_text}'")
        print(f"{'='*80}\n")
        
        # Get logits for Yes/No tokens
        first_token_logits = outputs.scores[0][0]
        yes_logit = torch.logsumexp(first_token_logits[yes_ids], dim=0)
        no_logit  = torch.logsumexp(first_token_logits[no_ids], dim=0)

        probs = F.softmax(torch.stack([no_logit, yes_logit]), dim=0)
        yes_prob = probs[1].item()

        # Optional: produce class label and answer
        threshold = 0.5  # or make configurable
        predicted_class = int(yes_prob >= threshold)
        answer_token = "Yes" if predicted_class == 1 else "No"
        confidence = yes_prob if predicted_class == 1 else (1 - yes_prob)

        return predicted_class, yes_prob, confidence, answer_token
    
    def test_samples(self, samples):
        """
        Test custom samples
        
        Args:
            samples: List of strings OR list of dicts with 'text' and optional 'label' keys
                Examples:
                - ["text1", "text2", ...]
                - [{"text": "...", "label": 1}, {"text": "..."}, ...]
        """
        print("="*80)
        print("MANUAL TESTING MODE")
        print("="*80)
        print(f"Testing {len(samples)} sample(s)\n")
        
        results = []
        correct_count = 0
        total_with_labels = 0
        
        for idx, sample in enumerate(samples, 1):
            # Parse sample
            if isinstance(sample, str):
                text = sample
                expected_label = None
            elif isinstance(sample, dict):
                text = sample['text']
                expected_label = sample.get('label', None)
            else:
                print(f"⚠️  Sample {idx}: Invalid format, skipping")
                continue
            
            # Run prediction
            pred_class, ai_prob, confidence, answer = self.predict_single(text)
            
            # Determine result
            pred_label_name = "AI" if pred_class == 1 else "Non-AI"
            
            if expected_label is not None:
                total_with_labels += 1
                expected_name = "AI" if expected_label == 1 else "Non-AI"
                is_correct = (pred_class == expected_label)
                if is_correct:
                    correct_count += 1
            else:
                expected_name = None
                is_correct = None
            
            # Store result
            result = {
                'sample_num': idx,
                'text': text,
                'text_preview': text[:100] + "..." if len(text) > 100 else text,
                'predicted_class': pred_class,
                'predicted_label': pred_label_name,
                'answer_token': answer,
                'ai_probability': ai_prob,
                'confidence': confidence,
                'expected_label': expected_label,
                'expected_name': expected_name,
                'is_correct': is_correct,
            }
            results.append(result)
            
            # Print result
            self.print_sample_result(result)
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total samples tested: {len(results)}")
        
        if total_with_labels > 0:
            accuracy = correct_count / total_with_labels
            print(f"Samples with labels: {total_with_labels}")
            print(f"Correct predictions: {correct_count}/{total_with_labels} ({accuracy*100:.1f}%)")
        
        # Count predictions
        ai_count = sum(1 for r in results if r['predicted_class'] == 1)
        nonai_count = len(results) - ai_count
        print(f"\nPrediction distribution:")
        print(f"  Predicted AI: {ai_count}/{len(results)} ({ai_count/len(results)*100:.1f}%)")
        print(f"  Predicted Non-AI: {nonai_count}/{len(results)} ({nonai_count/len(results)*100:.1f}%)")
        
        # Average confidence
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_ai_prob = sum(r['ai_probability'] for r in results) / len(results)
        print(f"\nAverage AI probability: {avg_ai_prob:.4f}")
        print(f"Average confidence: {avg_confidence:.4f}")
                
        return results
    
    def print_sample_result(self, result):
        """Pretty print a single sample result"""
        print(f"\n{'─'*80}")
        print(f"Sample {result['sample_num']}")
        print(f"{'─'*80}")
        
        # Text preview
        print(f"📝 Text: \"{result['text_preview']}\"")
        
        # Prediction
        pred_emoji = "🤖" if result['predicted_class'] == 1 else "👤"
        print(f"\n{pred_emoji} Prediction: {result['predicted_label']} ({result['answer_token']})")
        print(f"   AI Probability: {result['ai_probability']:.4f}")
        print(f"   Confidence: {result['confidence']:.4f}")
        
        # Expected vs actual (if label provided)
        if result['expected_label'] is not None:
            if result['is_correct']:
                print(f"   ✅ CORRECT (Expected: {result['expected_name']})")
            else:
                print(f"   ❌ INCORRECT (Expected: {result['expected_name']})")


def main():
    """Main entry point - EDIT THE SAMPLES BELOW TO TEST YOUR OWN TEXT"""
    
    # Configuration
    config_path = "config/instruction_training_config.yaml"
    model_path = "outputs/checkpoints/gemma3_20k_balanced_it_v2/final_model"
    
    # ========================================================================
    # DEFINE YOUR TEST SAMPLES HERE
    # ========================================================================
    
    # Option 1: Simple list of strings (no labels)


    test_samples = [
        
    """Meetup : Ottawa Weekly Meetup Discussion article for the meetup : Ottawa Weekly Meetup WHEN: 12 September 2011 07:30:00PM (-0400) WHERE: Pub Italia: 434 Preston St, Ottawa, ON Type: Discussion & Skill training Date: Monday September 12, 7:30pm until at least 9:00pm. Venue: Pub Italia (likely in a booth in the abbey - back of the pub) (We've settled on Monday evenings as the best time for most people, so we'll try this as the standard time and place for a while. Proposal for a more cost-effective or convenient location are eagerly solicited.) Skill: Mind mapping (I'll give an overview of mind maps and several potential applications, particularly as a tool for granularizing skills) Discussion post: Reflections on rationality a year out (A nice post on the question of what one should expect to gain as a result of actively participating in a rationalist community.) Discussion article for the meetup : Ottawa Weekly Meetup""",

    """Meetup : Ottawa Weekly Meetup

Discussion article for the meetup : Ottawa Weekly Meetup
WHEN: 12 September 2011 07:30:00PM (-0400)

WHERE: Pub Italia: 434 Preston St, Ottawa, ON

Type: Discussion & Skill training

Date: Monday September 12, 7:30pm until at least 9:00pm.

Venue: Pub Italia

(likely in a booth in the abbey - back of the pub)

(We've settled on Monday evenings as the best time for most people, so we'll try this as the standard time and place for a while. Proposal for a more cost-effective or convenient location are eagerly solicited.)

Skill: Mind mapping

(I'll give an overview of mind maps and several potential applications, particularly as a tool for granularizing skills)

Discussion post: Reflections on rationality a year out

(A nice post on the question of what one should expect to gain as a result of actively participating in a rationalist community.)


Discussion article for the meetup : Ottawa Weekly Meetup""", 
    """[link] How many humans will have their brain preserved? Forecasts and trends http://lessdead.com/how-many-humans-will-have-their-brain-preserved-forecasts-and-trends Summary: > Doubling time for the number of people that got cryopreserved has been pretty consistently 9 years since the beginning."""
    ]
    
    # Initialize tester
    tester = ManualTester(config_path, model_path)
    tester.setup_device()
    tester.load_model()
    
    # Test samples
    results = tester.test_samples(test_samples)
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()

