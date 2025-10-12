"""
Core training logic for Gemma3 Binary Classifier
Extracted from lora_finetune_gemma3_1b.py for better organization
"""

import os
import logging
import time
from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from peft import (LoraConfig, PeftConfig, prepare_model_for_kbit_training, 
                  get_peft_model, PeftModelForSequenceClassification)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
                         Trainer, EarlyStoppingCallback, DataCollatorWithPadding,
                         AutoModelForCausalLM, TrainerCallback)
import bitsandbytes as bnb
import evaluate
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import wandb
import requests
import json


class Gemma3ForSequenceClassification(PeftModelForSequenceClassification):
    """Custom class for Gemma 3 binary classification with class weighting"""
    
    def __init__(self, peft_config: PeftConfig, model: AutoModelForCausalLM, 
                 class_weights_dict=None, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.num_labels = model.config.num_labels
        self.problem_type = "single_label_classification"
        self.class_weights_dict = class_weights_dict
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        logits = outputs.logits
        
        # Get logits from last real token (not padding)
        sequence_lengths = torch.sum(attention_mask, dim=1)
        last_token_indices = sequence_lengths - 1
        batch_size = logits.shape[0]
        logits = logits[torch.arange(batch_size, device=logits.device), last_token_indices, :]
        
        loss = None
        if labels is not None:
            # Use weighted loss for class imbalance
            if self.class_weights_dict is not None:
                weights = torch.tensor(
                    [self.class_weights_dict[0], self.class_weights_dict[1]], 
                    dtype=torch.float32, 
                    device=logits.device
                )
                loss_fct = nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


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
        self.logger.info("Training Started")
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
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics"""
        if metrics:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Evaluation Results - Epoch {int(state.epoch)}")
            self.logger.info(f"{'='*80}")
            
            # Log key metrics
            if 'eval_loss' in metrics:
                self.logger.info(f"Eval Loss: {metrics['eval_loss']:.4f}")
            if 'eval_accuracy' in metrics:
                self.logger.info(f"Accuracy: {metrics['eval_accuracy']:.4f}")
            
            # Log per-class metrics for AI class
            if 'eval_ai_recall' in metrics:
                self.logger.info(f"\nAI Class Metrics:")
                self.logger.info(f"  Recall:    {metrics['eval_ai_recall']:.4f}")
                if 'eval_ai_precision' in metrics:
                    self.logger.info(f"  Precision: {metrics['eval_ai_precision']:.4f}")
                if 'eval_ai_f1' in metrics:
                    self.logger.info(f"  F1 Score:  {metrics['eval_ai_f1']:.4f}")
            
            # Log per-class metrics for non-AI class
            if 'eval_nonai_recall' in metrics:
                self.logger.info(f"\nNon-AI Class Metrics:")
                self.logger.info(f"  Recall:    {metrics['eval_nonai_recall']:.4f}")
                if 'eval_nonai_precision' in metrics:
                    self.logger.info(f"  Precision: {metrics['eval_nonai_precision']:.4f}")
                if 'eval_nonai_f1' in metrics:
                    self.logger.info(f"  F1 Score:  {metrics['eval_nonai_f1']:.4f}")
            
            # Log overall F1, Precision, Recall if available
            if 'eval_f1' in metrics:
                self.logger.info(f"\nOverall Metrics:")
                self.logger.info(f"  F1 Score:  {metrics['eval_f1']:.4f}")
                if 'eval_precision' in metrics:
                    self.logger.info(f"  Precision: {metrics['eval_precision']:.4f}")
                if 'eval_recall' in metrics:
                    self.logger.info(f"  Recall:    {metrics['eval_recall']:.4f}")
            
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


class BinaryClassifierTrainer:
    """Training pipeline for Gemma3 binary classifier"""
    
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
        self.wrapped_model = None
        self.trainer = None
        self.dataset = None
        self.dataset_tokenized = None
        self.class_weights_dict = None
        self.class2id = config['class_mapping']
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.optimal_threshold = 0.5
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
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        # Create logger
        self.logger = logging.getLogger('BinaryClassifierTrainer')
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
        self.logger.info("Logging Initialized")
        self.logger.info("="*80)
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Timestamp: {timestamp}")
        self.logger.info("-"*80)
        
    def setup_wandb(self):
        """Initialize Weights & Biases logging"""
        if not self.config['wandb']['enabled']:
            return
            
        wandb.login(key=self.config['wandb']['api_key'])
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
                "target_recall": self.config['threshold']['target_recall'],
            }
        )
    
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
    
    def compute_class_weights(self):
        """Compute balanced class weights for imbalanced data"""
        if not self.config['class_weights']['use_balanced']:
            self.logger.info("Balanced class weights disabled - using uniform weights")
            return
            
        self.logger.info("Computing balanced class weights for imbalanced data...")
        print(f"\n📊 Computing class weights for imbalanced data...")
        train_labels = self.dataset['train']['label']
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        self.class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print(f"   Class weights:")
        print(f"   non-AI (0): {self.class_weights_dict[0]:.3f}")
        print(f"   AI (1): {self.class_weights_dict[1]:.3f}")
        print(f"   → AI class will have {self.class_weights_dict[1]/self.class_weights_dict[0]:.2f}x more weight in loss")
        
        self.logger.info("Class weights computed:")
        self.logger.info(f"  non-AI (0): {self.class_weights_dict[0]:.3f}")
        self.logger.info(f"  AI (1): {self.class_weights_dict[1]:.3f}")
        self.logger.info(f"  AI class weight multiplier: {self.class_weights_dict[1]/self.class_weights_dict[0]:.2f}x")
    
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
        
        def preprocess_function(sample):
            tokenized = self.tokenizer(sample['text'], truncation=True, max_length=max_length)
            tokenized['labels'] = sample['label']
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
    
    def load_model(self):
        """Load and configure the base model with quantization"""
        self.logger.info(f"Loading base model: {self.config['model']['hugging_face_model_id']}")
        self.logger.info(f"Quantization: 4-bit ({self.config['quantization']['bnb_4bit_quant_type']})")
        
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
        
        # Replace language modeling head with classification head
        self.model.lm_head = nn.Linear(
            self.model.config.hidden_size, 
            self.config['model']['num_labels'], 
            bias=False, 
            device=self.config['gpu']['gpu_device']
        )
        self.logger.info(f"Replaced LM head with classification head ({self.config['model']['num_labels']} labels)")
    
    def setup_lora(self):
        """Configure and apply LoRA"""
        self.logger.info("Configuring LoRA...")
        self.logger.info(f"  r={self.config['lora']['r']}, alpha={self.config['lora']['lora_alpha']}, dropout={self.config['lora']['lora_dropout']}")
        self.logger.info(f"  Target modules: {', '.join(self.config['lora']['target_modules'])}")
        
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        self.logger.info("Gradient checkpointing enabled, model prepared for k-bit training")
        
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type']
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.logger.info("LoRA adapters applied successfully")
        
        # Wrap model with custom classification class
        peft_config = PeftConfig(peft_type="LORA", task_type="SEQ_CLS", inference_mode=False)
        for key, value in lora_config.__dict__.items():
            setattr(peft_config, key, value)
        
        self.wrapped_model = Gemma3ForSequenceClassification(
            peft_config, self.model, class_weights_dict=self.class_weights_dict
        )
        self.wrapped_model.num_labels = self.config['model']['num_labels']
        self.logger.info("Model wrapped with custom classification head")
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Compute overall metrics
        metrics = clf_metrics.compute(predictions=predictions, references=labels)
        
        # Also compute per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, labels=[0, 1], zero_division=0
        )
        
        metrics['nonai_precision'] = precision[0]
        metrics['nonai_recall'] = recall[0]
        metrics['nonai_f1'] = f1[0]
        metrics['ai_precision'] = precision[1]
        metrics['ai_recall'] = recall[1]
        metrics['ai_f1'] = f1[1]
        
        return metrics
    
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
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.trainer = Trainer(
            model=self.wrapped_model,
            args=training_args,
            train_dataset=self.dataset_tokenized['train'],
            eval_dataset=self.dataset_tokenized['valid'],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stop, logging_callback]
        )
        
        self.logger.info("Trainer configured successfully")
    
    def train(self):
        """Execute training"""
        print("\n" + "="*80)
        print("Starting training...")
        print("="*80 + "\n")
        
        self.trainer.train(resume_from_checkpoint=False)
    
    def find_optimal_threshold(self, target_recall=None):
        """Find threshold that achieves target recall on AI class (label=1)"""
        if target_recall is None:
            target_recall = self.config['threshold']['target_recall']
            
        print(f"\n🎯 Finding optimal threshold for {target_recall*100:.0f}% recall on AI class...")
        
        predictions = self.trainer.predict(self.dataset_tokenized['valid'])
        logits = predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        
        # Probability of AI class (label=1)
        ai_probs = probs[:, 1]
        true_labels = predictions.label_ids
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(
            true_labels, ai_probs, pos_label=1
        )
        
        # Find threshold that achieves target recall
        valid_idx = np.where(recalls >= target_recall)[0]
        
        if len(valid_idx) == 0:
            print(f"  ⚠️  Could not achieve {target_recall*100:.0f}% recall")
            best_idx = np.argmax(recalls)
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            print(f"  Best recall achieved: {recalls[best_idx]:.4f} at threshold {optimal_threshold:.4f}")
        else:
            # Among valid thresholds, pick one with highest precision
            best_idx = valid_idx[np.argmax(precisions[valid_idx])]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            
            print(f"  ✅ Optimal threshold: {optimal_threshold:.4f}")
            print(f"     AI Recall: {recalls[best_idx]:.4f}")
            print(f"     AI Precision: {precisions[best_idx]:.4f}")
            print(f"     F1: {2 * precisions[best_idx] * recalls[best_idx] / (precisions[best_idx] + recalls[best_idx]):.4f}")
        
        self.optimal_threshold = optimal_threshold
        
        return optimal_threshold, {
            'threshold': float(optimal_threshold),
            'recall': float(recalls[best_idx]),
            'precision': float(precisions[best_idx])
        }
    
    def save_model(self, threshold_metrics):
        """Save trained model and optimal threshold"""
        output_dir = self.config['output']['model_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save threshold
        with open(f"{output_dir}/optimal_threshold.json", 'w') as f:
            json.dump(threshold_metrics, f, indent=2)
        
        # Save model
        print("\nSaving model...")
        self.trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
        print(f"Optimal threshold saved to {output_dir}/optimal_threshold.json")
    
    def predict_text(self, text, threshold=None):
        """
        Predict the class for a given text
        Returns: predicted class label and probability
        """
        if threshold is None:
            threshold = self.optimal_threshold
            
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config['model']['max_length']
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.wrapped_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            ai_prob = probs[0][1].item()  # Probability of AI class
            
            # Use custom threshold
            predicted_class = 1 if ai_prob >= threshold else 0
        
        return self.id2class[predicted_class], ai_prob
    
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
            predicted_label, ai_prob = self.predict_text(example, threshold=self.optimal_threshold)
            print(f"Text: {example}")
            print(f"Expected: {expected} | Predicted: {predicted_label} | AI prob: {ai_prob:.4f}")
            print("-" * 80)
    
    def evaluate_test_set(self):
        """Evaluate on test set with optimal threshold"""
        print("\n" + "="*80)
        print("Evaluating on test set with optimal threshold...")
        print("="*80 + "\n")
        
        # Get test predictions with optimal threshold
        test_predictions = self.trainer.predict(self.dataset_tokenized['test'])
        test_logits = test_predictions.predictions
        test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
        test_ai_probs = test_probs[:, 1]
        test_labels = test_predictions.label_ids
        
        # Apply optimal threshold
        test_preds = (test_ai_probs >= self.optimal_threshold).astype(int)
        
        # Compute detailed metrics
        print("Classification Report:")
        print(classification_report(test_labels, test_preds, target_names=['non-AI', 'AI'], digits=4))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, test_preds)
        print(f"                Predicted")
        print(f"                non-AI    AI")
        print(f"Actual non-AI   {cm[0,0]:<9} {cm[0,1]}")
        print(f"Actual AI       {cm[1,0]:<9} {cm[1,1]}")
        
        # Also show default 0.5 threshold results for comparison
        test_results = self.trainer.evaluate(self.dataset_tokenized['test'])
        print("\nTest Results (default 0.5 threshold):")
        for key, value in test_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
    
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
        self.logger.info("Starting Full Training Pipeline")
        self.logger.info("="*80)
        self.logger.info(f"Model: {self.config['model']['hugging_face_model_id']}")
        self.logger.info(f"Task: Binary Classification (AI vs non-AI content)")
        self.logger.info("-"*80)
        
        self.setup_wandb()
        
        # Data preparation
        self.load_data()
        self.compute_class_weights()
        self.setup_tokenizer()
        self.tokenize_dataset()
        
        # Model setup
        self.load_model()
        self.setup_lora()
        
        # Training
        self.setup_trainer()
        self.train()
        
        # Threshold optimization
        optimal_threshold, threshold_metrics = self.find_optimal_threshold()
        
        # Save model
        self.save_model(threshold_metrics)
        
        # Testing
        self.test_predictions()
        self.evaluate_test_set()
        
        # Notifications
        notification_message = f"""
🎉 *Gemma Training Complete!*

📊 *Final Results:*
- Epochs completed: {self.trainer.state.epoch}
- Best AI Recall: {threshold_metrics['recall']:.4f}
- AI Precision: {threshold_metrics['precision']:.4f}
- Optimal Threshold: {threshold_metrics['threshold']:.4f}

💾 *Model saved to:* `{self.config['output']['model_dir']}`

✅ Ready for next script!
"""
        self.notify_slack(notification_message)
        
        # Log to wandb
        if self.config['wandb']['enabled']:
            wandb.log({
                "final/optimal_threshold": threshold_metrics['threshold'],
                "final/ai_recall": threshold_metrics['recall'],
                "final/ai_precision": threshold_metrics['precision']
            })
            wandb.finish()
        
        print("\n✅ Training and evaluation complete!")
        print(f"💾 Model and optimal threshold saved to: {self.config['output']['model_dir']}")
        
        # Final logging summary
        self.logger.info("="*80)
        self.logger.info("Pipeline Completed Successfully!")
        self.logger.info("="*80)
        self.logger.info(f"Final Results:")
        self.logger.info(f"  Epochs completed: {self.trainer.state.epoch}")
        self.logger.info(f"  Best AI Recall: {threshold_metrics['recall']:.4f}")
        self.logger.info(f"  AI Precision: {threshold_metrics['precision']:.4f}")
        self.logger.info(f"  Optimal Threshold: {threshold_metrics['threshold']:.4f}")
        self.logger.info(f"  Model saved to: {self.config['output']['model_dir']}")
        self.logger.info("="*80)

