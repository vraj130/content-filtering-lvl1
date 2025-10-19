import os
import logging
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase
import wandb
import requests
import json
from dotenv import load_dotenv

from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
                         Trainer, EarlyStoppingCallback, DataCollatorForLanguageModeling, DataCollatorWithPadding,
                         AutoModelForCausalLM, TrainerCallback)
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


load_dotenv()


@dataclass
class DataCollatorForInstructionTuning:
    """
    Data collator for instruction fine-tuning
    Pads input_ids, attention_mask, and labels to the same length
    """
    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: int = None
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract labels first (before padding)
        labels = [feature["labels"] for feature in features] if "labels" in features[0] else None
        
        # Remove labels from features for padding
        features_without_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        
        # Pad input_ids and attention_mask
        batch = self.tokenizer.pad(
            features_without_labels,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Pad labels manually
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            
            # Pad labels with -100 (ignore index)
            padded_labels = []
            for label in labels:
                padding_length = max_label_length - len(label)
                padded_label = label + [-100] * padding_length
                padded_labels.append(padded_label)
            
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch




class TrainingLoggingCallback(TrainerCallback):
    """Custom callback for comprehensive training logging"""
    
    def __init__(self, logger):
        self.logger = logger
        self.epoch_start_time = None
        self.training_start_time = time.time()
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Log when training begins"""
        self.training_start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info("Training Started - Instruction Fine-Tuning")
        self.logger.info("=" * 80)
        self.logger.info(f"Total epochs: {args.num_train_epochs}")
        self.logger.info(f"Training batch size: {args.per_device_train_batch_size}")
        self.logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        self.logger.info(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        self.logger.info(f"Learning rate: {args.learning_rate}")
        self.logger.info(f"Weight decay: {args.weight_decay}")
        self.logger.info("-" * 80)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Log when epoch begins"""
        self.epoch_start_time = time.time()
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Epoch {int(state.epoch) + 1}/{args.num_train_epochs} Started")
        self.logger.info(f"{'='*80}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log when epoch ends"""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        self.logger.info(f"\nEpoch {int(state.epoch)}/{args.num_train_epochs} Completed")
        self.logger.info(f"Epoch time: {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes)")
        
        # Log training loss if available
        if state.log_history:
            for log in reversed(state.log_history):
                if 'loss' in log and 'epoch' in log:
                    if abs(log['epoch'] - state.epoch) < 0.01:  # Match current epoch
                        self.logger.info(f"Training Loss: {log['loss']:.4f}")
                        if 'learning_rate' in log:
                            self.logger.info(f"Learning Rate: {log['learning_rate']:.2e}")
                        break
        self.logger.info("-" * 80)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log when training ends"""
        total_time = time.time() - self.training_start_time
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Training Completed")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        self.logger.info(f"Total epochs completed: {state.epoch}")
        self.logger.info("-" * 80)


class InstructionFineTuner:
    """Training pipeline for Gemma3 instruction-based binary classifier"""
    
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.device = None
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.dataset = None
        self.dataset_tokenized = None
        self.class2id = config['class_mapping']
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.logger = None
        
    def setup_environment(self):
        """Setup GPU and environment variables"""
        graphic_card = self.config['gpu']['graphic_card']
        os.environ["CUDA_VISIBLE_DEVICES"] = graphic_card
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.device = torch.device(f"cuda:{graphic_card}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(0)
        print(f"Using device: {self.device}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    def setup_logging(self):
        """Setup comprehensive logging to both console and file"""
        # Create logs directory if it doesn't exist
        log_dir = "outputs/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"instruction_training_{timestamp}.log")
        
        # Create logger
        self.logger = logging.getLogger('InstructionFineTuner')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler (detailed logging)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (less verbose)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Log initial setup information
        self.logger.info("="*80)
        self.logger.info("Logging Initialized - Instruction Fine-Tuning")
        self.logger.info("="*80)
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Timestamp: {timestamp}")
        self.logger.info("-"*80)
        
    def setup_wandb(self):
        """Initialize Weights & Biases logging"""
        if not self.config['wandb']['enabled']:
            return
        
        wandb.init(
            project=self.config['wandb']['project'],
            name=self.config['wandb']['run_name'],
            config={
                "model": self.config['model']['hugging_face_model_id'],
                "max_length": self.config['model']['max_length'],
                "batch_size": self.config['training']['per_device_train_batch_size'],
                "gradient_accumulation_steps": self.config['training']['gradient_accumulation_steps'],
                "learning_rate": self.config['training']['learning_rate'],
                "num_epochs": self.config['training']['num_train_epochs'],
                "lora_r": self.config['lora']['r'],
                "lora_alpha": self.config['lora']['lora_alpha'],
                "training_type": "instruction_finetuning",
            }
        )
    
    def create_prompt(self, text, label=None):
        """
        Create instruction prompt for the model
        
        Args:
            text: Document text
            label: Label (0 or 1), optional for inference
            
        Returns:
            Formatted prompt string
        """
        # Get prompt template from config
        prompt_template = self.config['instruction']['prompt_template']
        
        if label is not None:
            # Training: include the answer
            answer = self.config['instruction']['answer_yes'] if label == 1 else self.config['instruction']['answer_no']
            prompt = f"{prompt_template.format(text=text)}{answer}<end_of_turn>"
        else:
            # Inference: just the question
            prompt = prompt_template.format(text=text)
        
        return prompt
    
    def load_data(self):
        """Load training, validation, and test datasets"""
        self.logger.info("Loading datasets...")
        data_dir = self.config['data']['data_dir']
        self.logger.info(f"Data directory: {data_dir}")
        
        train_df = pd.read_csv(f"{data_dir}/{self.config['data']['train_file']}")
        val_df = pd.read_csv(f"{data_dir}/{self.config['data']['val_file']}")
        test_df = pd.read_csv(f"{data_dir}/{self.config['data']['test_file']}")
        
        print(f"✅ Loaded unified dataset:")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Valid: {len(val_df)} samples")
        print(f"   Test:  {len(test_df)} samples")
        
        self.logger.info(f"Dataset loaded successfully:")
        self.logger.info(f"  Train: {len(train_df)} samples")
        self.logger.info(f"  Validation: {len(val_df)} samples")
        self.logger.info(f"  Test: {len(test_df)} samples")

        if self.config.get('quick_test', False):
            print(f"\n⚡⚡⚡ QUICK TEST MODE ENABLED ⚡⚡⚡")
            print(f"   Original sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            
            # Use tiny subsets - stratified by label to maintain class balance
            train_df = train_df.groupby('label', group_keys=False).apply(
                lambda x: x.head(25)  # 25 per class = 50 total
            ).reset_index(drop=True)
            
            val_df = val_df.groupby('label', group_keys=False).apply(
                lambda x: x.head(10)  # 10 per class = 20 total
            ).reset_index(drop=True)
            
            test_df = test_df.groupby('label', group_keys=False).apply(
                lambda x: x.head(10)  # 10 per class = 20 total
            ).reset_index(drop=True)
            
            print(f"   Test subset sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            print(f"   ⚠️  This is ONLY for testing the pipeline, not real training!")
            self.logger.warning("QUICK TEST MODE: Using tiny stratified data subset")

        
        # Format data with prompts
        self.logger.info("Formatting data with instruction prompts...")
        print(f"\n🎯 Formatting data with instruction prompts...")
        
        for df in [train_df, val_df, test_df]:
            df['prompted_text'] = df.apply(
                lambda row: self.create_prompt(row['text'], row['label']), 
                axis=1
            )
        
        # Verify label distribution
        print(f"\n🏷️  Label distribution:")
        self.logger.info("Label distribution:")
        for split_name, split_df in [('Train', train_df), ('Valid', val_df), ('Test', test_df)]:
            ai_count = (split_df['label'] == 1).sum()
            nonai_count = (split_df['label'] == 0).sum()
            print(f"   {split_name}: non-AI={nonai_count} ({nonai_count/len(split_df)*100:.1f}%), AI={ai_count} ({ai_count/len(split_df)*100:.1f}%)")
            self.logger.info(f"  {split_name}: non-AI={nonai_count} ({nonai_count/len(split_df)*100:.1f}%), AI={ai_count} ({ai_count/len(split_df)*100:.1f}%)")
        
        # Convert to Hugging Face Dataset
        self.dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'valid': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df)
        })
        
        print(f"\nDataset loaded:\n{self.dataset}")
        self.logger.info("Dataset conversion to HuggingFace format complete")
        
        # Log example prompt
        print(f"\n📝 Example prompt (first 500 chars):")
        example_prompt = train_df['prompted_text'].iloc[0]
        print(example_prompt[:500])
        self.logger.info("Example prompt:")
        self.logger.info(example_prompt[:500])
    
    def setup_tokenizer(self):
        """Initialize and configure tokenizer"""
        self.logger.info(f"Loading tokenizer: {self.config['model']['hugging_face_model_id']}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['hugging_face_model_id'],
            padding_side='right',
            device_map=self.config['gpu']['gpu_device'],
            add_bos=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info("Set pad_token to eos_token")
        
        self.logger.info("Tokenizer setup complete")
    
    def tokenize_dataset(self):
        """Tokenize the dataset"""
        max_length = self.config['model']['max_length']
        self.logger.info(f"Tokenizing dataset with max_length={max_length}...")
        self.logger.info("Creating labels with -100 masking for prompt tokens...")
        print(f"\n🔧 Tokenizing dataset with label masking...")
        
        def preprocess_function(sample):
            """Tokenize and create labels with -100 masking"""
            # Tokenize the full prompted text
            tokenized = self.tokenizer(
                sample['prompted_text'], 
                truncation=True, 
                max_length=max_length,
                padding=False  # Pad in collator
            )
            
            # Create labels - copy of input_ids
            labels = tokenized['input_ids'].copy()
            
            # Find where the answer starts
            # We need to mask everything before the answer
            answer_yes = self.config['instruction']['answer_yes']
            answer_no = self.config['instruction']['answer_no']
            
            # Tokenize answers to find their token IDs
            yes_tokens = self.tokenizer.encode(answer_yes, add_special_tokens=False)
            no_tokens = self.tokenizer.encode(answer_no, add_special_tokens=False)
            
            # Find where answer starts in the sequence
            answer_start_idx = None
            for i in range(len(labels) - max(len(yes_tokens), len(no_tokens))):
                if (labels[i:i+len(yes_tokens)] == yes_tokens or 
                    labels[i:i+len(no_tokens)] == no_tokens):
                    answer_start_idx = i
                    break
            
            # Mask everything before the answer with -100
            if answer_start_idx is not None:
                labels[:answer_start_idx] = [-100] * answer_start_idx
            else:
                # If we can't find the answer, mask everything (shouldn't happen)
                self.logger.warning(f"Could not find answer tokens in sequence")
            
            tokenized['labels'] = labels
            
            return tokenized
        
        self.dataset_tokenized = self.dataset.map(
            preprocess_function, 
            remove_columns=self.dataset['train'].column_names
        )
        print(f"\nTokenized dataset:\n{self.dataset_tokenized}")
        self.logger.info("Dataset tokenization complete")
        self.logger.info(f"  Train: {len(self.dataset_tokenized['train'])} samples")
        self.logger.info(f"  Validation: {len(self.dataset_tokenized['valid'])} samples")
        self.logger.info(f"  Test: {len(self.dataset_tokenized['test'])} samples")
        self.logger.info("  Label masking applied: loss computed only on answer tokens")
    
    def load_model(self):
        """Load and configure the base model with quantization"""
        self.logger.info(f"Loading base model: {self.config['model']['hugging_face_model_id']}")
        self.logger.info(f"Quantization: 4-bit ({self.config['quantization']['bnb_4bit_quant_type']})")
        print(f"\n🤖 Loading base model...")
        print(f"   Model: {self.config['model']['hugging_face_model_id']}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config['quantization']['load_in_4bit'],
            bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype'])
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['hugging_face_model_id'],
            dtype=torch.bfloat16,
            device_map=self.config['gpu']['gpu_device'],
            attn_implementation=self.config['model']['attention_implementation'],
            quantization_config=bnb_config
        )
        
        self.logger.info("Base model loaded successfully")
        print(f"✅ Base model loaded (architecture unchanged)")
        
        # DO NOT replace lm_head - this is the key difference from classification approach
        self.logger.info("Keeping original lm_head for language modeling")
    
    def setup_lora(self):
        """Configure and apply LoRA"""
        self.logger.info("Configuring LoRA...")
        self.logger.info(f"  r={self.config['lora']['r']}, alpha={self.config['lora']['lora_alpha']}, dropout={self.config['lora']['lora_dropout']}")
        self.logger.info(f"  Target modules: {', '.join(self.config['lora']['target_modules'])}")
        self.logger.info(f"  Task type: {self.config['lora']['task_type']} (CAUSAL_LM for instruction tuning)")
        print(f"\n🔧 Configuring LoRA...")
        print(f"   r={self.config['lora']['r']}, alpha={self.config['lora']['lora_alpha']}")
        print(f"   Task type: {self.config['lora']['task_type']}")
        
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        self.logger.info("Gradient checkpointing enabled, model prepared for k-bit training")
        
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type']  # CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.logger.info("LoRA adapters applied successfully")
        print(f"✅ LoRA adapters applied")
    
    def setup_trainer(self):
        """Configure the Hugging Face Trainer"""
        self.logger.info("Setting up Trainer...")
        self.logger.info(f"  Output directory: {self.config['training']['output_dir']}")
        self.logger.info(f"  Epochs: {self.config['training']['num_train_epochs']}")
        self.logger.info(f"  Learning rate: {self.config['training']['learning_rate']}")
        self.logger.info(f"  Batch size per device: {self.config['training']['per_device_train_batch_size']}")
        self.logger.info(f"  Gradient accumulation steps: {self.config['training']['gradient_accumulation_steps']}")
        self.logger.info(f"  Early stopping patience: {self.config['early_stopping']['patience']}")
        
        early_stop = EarlyStoppingCallback(
            early_stopping_patience=self.config['early_stopping']['patience'],
            early_stopping_threshold=self.config['early_stopping']['threshold']
        )
        
        # Create logging callback
        logging_callback = TrainingLoggingCallback(self.logger)
        
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            learning_rate=self.config['training']['learning_rate'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            weight_decay=self.config['training']['weight_decay'],
            eval_strategy=self.config['training']['eval_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            push_to_hub=self.config['training']['push_to_hub'],
            report_to=["tensorboard", "wandb"] if self.config['wandb']['enabled'] else ["tensorboard"],
            logging_dir=self.config['output']['tensorboard_dir'],
            logging_strategy=self.config['training']['logging_strategy'],
            logging_steps=self.config['training']['logging_steps'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            bf16=self.config['training']['bf16'],
            fp16=self.config['training']['fp16'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            dataloader_num_workers=self.config['training']['dataloader_num_workers'],
            dataloader_pin_memory=self.config['training']['dataloader_pin_memory'],
            dataloader_prefetch_factor=self.config['training']['dataloader_prefetch_factor']
        )
        
        # Use DataCollatorForLanguageModeling for instruction tuning
        data_collator = DataCollatorForInstructionTuning(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset_tokenized['train'],
            eval_dataset=self.dataset_tokenized['valid'],
            data_collator=data_collator,
            callbacks=[early_stop, logging_callback]
        )
        
        self.logger.info("Trainer configured successfully")
    
    def train(self):
        """Execute training"""
        print("\n" + "="*80)
        print("Starting training...")
        print("="*80 + "\n")
        
        self.trainer.train(resume_from_checkpoint=False)
    
    def save_model(self):
        """Save trained model"""
        output_dir = self.config['output']['model_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nSaving model...")
        self.trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
        self.logger.info(f"Model saved to {output_dir}")
    
    def _predict_single_chunk(self, text_chunk):

    # Create prompt
        prompt = self.create_prompt(text_chunk, label=None)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=32000  # Use larger context for inference
        ).to(self.device)
        
        # Get token IDs for Yes/No
        yes_tokens = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_tokens = self.tokenizer.encode("No", add_special_tokens=False)
        yes_token_id = yes_tokens[0]
        no_token_id = no_tokens[0]
        
        # Generate with constrained decoding (single token)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,  # Only generate one token
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )
        
        # Get logits for Yes/No tokens
        first_token_logits = outputs.scores[0][0]
        yes_score = first_token_logits[yes_token_id].item()
        no_score = first_token_logits[no_token_id].item()
        
        # Calculate probabilities
       
        probs = F.softmax(torch.tensor([no_score, yes_score]), dim=0)
        yes_prob = probs[1].item()
        
        # Predict: 1 if yes_score > no_score, else 0
        predicted_class = 1 if yes_score > no_score else 0
        confidence = yes_prob if predicted_class == 1 else (1 - yes_prob)
        
        return predicted_class, confidence
    
    def predict_text(self, text):
        """
        Predict the class for a given text with intelligent chunking
        
        For short texts (< 30K tokens): Single prediction
        For long texts (>= 30K tokens): Chunk-based OR voting
        
        Args:
            text: Input document text
            
        Returns:
            tuple: (predicted_class_label, answer_string, confidence, chunk_info)
        """
        # Configuration
        max_chunk_tokens = 30000  # Much larger than training (8192)
        overlap_tokens = 2000     # Overlap between chunks
        
        # Tokenize full document to check length
        doc_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(doc_tokens)
        
        # Case 1: Short document - single prediction
        if num_tokens <= max_chunk_tokens:
            predicted_class, confidence = self._predict_single_chunk(text)
            
            answer = "AI" if predicted_class == 1 else "Non-AI"
            chunk_info = {
                'num_chunks': 1,
                'total_tokens': num_tokens,
                'used_chunking': False
            }
            
            return self.id2class[predicted_class], answer, confidence, chunk_info
        
        # Case 2: Long document - chunk and vote
        if hasattr(self, 'logger'):
            self.logger.info(f"Long document ({num_tokens} tokens) - using chunking strategy")
        
        # Create chunks with overlap
        chunks = []
        chunk_predictions = []
        chunk_confidences = []
        
        start_idx = 0
        while start_idx < num_tokens:
            # Extract chunk
            end_idx = min(start_idx + max_chunk_tokens, num_tokens)
            chunk_tokens = doc_tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            # Predict on chunk
            pred_class, confidence = self._predict_single_chunk(chunk_text)
            chunk_predictions.append(pred_class)
            chunk_confidences.append(confidence)
            
            # Move to next chunk
            if end_idx >= num_tokens:
                break
            start_idx += (max_chunk_tokens - overlap_tokens)
        
        # OR Voting: If ANY chunk predicts AI (1) -> Final is AI (1)
        final_prediction = 1 if any(chunk_predictions) else 0
        
        # Confidence: Use max confidence among chunks that match final prediction
        matching_confidences = [
            conf for pred, conf in zip(chunk_predictions, chunk_confidences)
            if pred == final_prediction
        ]
        final_confidence = max(matching_confidences) if matching_confidences else 0.5
        
        answer = "AI" if final_prediction == 1 else "Non-AI"
        
        chunk_info = {
            'num_chunks': len(chunks),
            'total_tokens': num_tokens,
            'used_chunking': True,
            'chunk_predictions': chunk_predictions,
            'chunk_confidences': chunk_confidences,
            'ai_chunks': sum(chunk_predictions),
            'voting_rule': 'OR (any chunk=AI -> final=AI)'
        }
        
        if hasattr(self, 'logger'):
            self.logger.info(f"Chunking result: {len(chunks)} chunks, "
                            f"{sum(chunk_predictions)} predicted AI, "
                            f"final={answer}")
        
        return self.id2class[final_prediction], answer, final_confidence, chunk_info

    def test_predictions(self):
        """Test predictions on example texts"""
        print("\n" + "="*80)
        print("Testing predictions on examples...")
        print("="*80 + "\n")
        
        test_examples = [
            ("We discuss reinforcement learning from human feedback and its applications in AI safety.", "ai"),
            ("The recipe for chocolate chip cookies requires butter, sugar, and flour.", "nonai"),
            ("Constitutional AI aims to align language models with human values through iterative refinement.", "ai"),
            ("I went to the park yesterday and saw many dogs playing.", "nonai")
        ]
        
        for example, expected in test_examples:
            predicted_label, answer, confidence, chunk_info = self.predict_text(example)
            print(f"Text: {example}")
            print(f"Expected: {expected} | Predicted: {predicted_label} | "
                f"Confidence: {confidence:.4f} | Chunks: {chunk_info['num_chunks']}")
            print("-" * 80)

    def evaluate_test_set(self):
        """Evaluate on test set with chunking for long documents"""
        print("\n" + "="*80)
        print("Evaluating on test set with intelligent chunking...")
        print("="*80 + "\n")
        
        self.logger.info("Starting test set evaluation with chunking strategy...")
        
        # Get predictions for all test samples
        test_df = self.dataset['test'].to_pandas()
        predictions = []
        confidences = []
        chunk_stats = {
            'total_docs': len(test_df),
            'chunked_docs': 0,
            'single_chunk_docs': 0,
            'total_chunks': 0,
            'max_chunks': 0
        }
        true_labels = test_df['label'].tolist()
        
        print(f"Generating predictions for {len(test_df)} test samples...")
        self.logger.info(f"Generating predictions for {len(test_df)} samples...")
        
        for idx, row in test_df.iterrows():
            if idx % 50 == 0:
                print(f"  Progress: {idx}/{len(test_df)}")
                self.logger.info(f"  Progress: {idx}/{len(test_df)}")
            
            # Predict with chunking
            predicted_label, answer, confidence, chunk_info = self.predict_text(row['text'])
            predictions.append(1 if predicted_label == 'ai' else 0)
            confidences.append(confidence)
            
            # Track chunking statistics
            if chunk_info['used_chunking']:
                chunk_stats['chunked_docs'] += 1
                chunk_stats['total_chunks'] += chunk_info['num_chunks']
                chunk_stats['max_chunks'] = max(chunk_stats['max_chunks'], chunk_info['num_chunks'])
            else:
                chunk_stats['single_chunk_docs'] += 1
                chunk_stats['total_chunks'] += 1
        
        # Print chunking statistics
        print(f"\n📊 Chunking Statistics:")
        print(f"   Total documents: {chunk_stats['total_docs']}")
        print(f"   Single-chunk docs: {chunk_stats['single_chunk_docs']} "
            f"({chunk_stats['single_chunk_docs']/chunk_stats['total_docs']*100:.1f}%)")
        print(f"   Multi-chunk docs: {chunk_stats['chunked_docs']} "
            f"({chunk_stats['chunked_docs']/chunk_stats['total_docs']*100:.1f}%)")
        print(f"   Total chunks processed: {chunk_stats['total_chunks']}")
        print(f"   Max chunks in single doc: {chunk_stats['max_chunks']}")
        print(f"   Avg chunks per doc: {chunk_stats['total_chunks']/chunk_stats['total_docs']:.2f}")
        
        self.logger.info(f"Chunking stats: {chunk_stats}")
        
        # Compute detailed metrics
        print("\nClassification Report:")
        report = classification_report(true_labels, predictions, 
                                    target_names=['non-AI', 'AI'], digits=4)
        print(report)
        self.logger.info("Classification Report:")
        self.logger.info("\n" + report)
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_labels, predictions)
        print(f"                Predicted")
        print(f"                non-AI    AI")
        print(f"Actual non-AI   {cm[0,0]:<9} {cm[0,1]}")
        print(f"Actual AI       {cm[1,0]:<9} {cm[1,1]}")
        
        self.logger.info("Confusion Matrix:")
        self.logger.info(f"                Predicted")
        self.logger.info(f"                non-AI    AI")
        self.logger.info(f"Actual non-AI   {cm[0,0]:<9} {cm[0,1]}")
        self.logger.info(f"Actual AI       {cm[1,0]:<9} {cm[1,1]}")
        
        # Compute per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, labels=[0, 1], zero_division=0
        )
        
        print(f"\nPer-Class Metrics:")
        print(f"Non-AI: Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1={f1[0]:.4f}")
        print(f"AI:     Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1={f1[1]:.4f}")
        
        self.logger.info(f"Per-Class Metrics:")
        self.logger.info(f"Non-AI: Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1={f1[0]:.4f}")
        self.logger.info(f"AI:     Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1={f1[1]:.4f}")
        
        return {
            'ai_recall': recall[1],
            'ai_precision': precision[1],
            'ai_f1': f1[1],
            'nonai_recall': recall[0],
            'nonai_precision': precision[0],
            'nonai_f1': f1[0],
            'chunk_stats': chunk_stats
        }


    def notify_slack(self, message):
        """Send Slack notification"""
        if not self.config['slack']['enabled']:
            return
            
        webhook = self.config['slack']['webhook_url']
        
        try:
            response = requests.post(
                webhook,
                data=json.dumps({"text": message}),
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code == 200:
                print("✅ Slack notification sent!")
            else:
                print(f"⚠️ Slack notification failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Failed to send Slack notification: {e}")
    
    def run_full_pipeline(self):
        """Execute the complete training pipeline"""
        # Setup logging first
        self.setup_environment()
        self.setup_logging()
        
        self.logger.info("="*80)
        self.logger.info("Starting Instruction Fine-Tuning....")
        self.logger.info("="*80)
        self.logger.info(f"Model: {self.config['model']['hugging_face_model_id']}")
        self.logger.info(f"Task: Instruction-based Binary Classification (AI vs non-AI content)")
        self.logger.info(f"Method: Prompt-based fine-tuning with answer generation")
        self.logger.info("-"*80)
        
        self.setup_wandb()
        
        # Data preparation
        self.load_data()
        self.setup_tokenizer()
        self.tokenize_dataset()
        
        # Model setup
        self.load_model()
        self.setup_lora()
        
        # Training
        self.setup_trainer()
        self.train()
        
        # Save model
        self.save_model()
        
        # Testing
        self.test_predictions()
        test_metrics = self.evaluate_test_set()
        
        # Notifications
        notification_message = f"""
🎉 *Instruction Fine-Tuning Completed!*

📊 *Final Results:*
- Epochs completed: {self.trainer.state.epoch}
- AI Recall: {test_metrics['ai_recall']:.4f}
- AI Precision: {test_metrics['ai_precision']:.4f}
- AI F1: {test_metrics['ai_f1']:.4f}

💾 *Model saved to:* `{self.config['output']['model_dir']}`


"""
        self.notify_slack(notification_message)
        
        # Log to wandb
        if self.config['wandb']['enabled']:
            wandb.log({
                "final/ai_recall": test_metrics['ai_recall'],
                "final/ai_precision": test_metrics['ai_precision'],
                "final/ai_f1": test_metrics['ai_f1']
            })
            wandb.finish()
        
        print("\n✅ Training and evaluation complete!")
        print(f"💾 Model saved to: {self.config['output']['model_dir']}")
        
        # Final logging summary
        self.logger.info("="*80)
        self.logger.info("="*80)
        self.logger.info(f"Final Results:")
        self.logger.info(f"  Epochs completed: {self.trainer.state.epoch}")
        self.logger.info(f"  AI Recall: {test_metrics['ai_recall']:.4f}")
        self.logger.info(f"  AI Precision: {test_metrics['ai_precision']:.4f}")
        self.logger.info(f"  AI F1: {test_metrics['ai_f1']:.4f}")
        self.logger.info(f"  Model saved to: {self.config['output']['model_dir']}")
        self.logger.info("="*80)