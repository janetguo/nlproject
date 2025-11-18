"""Direct Preference Optimization (DPO) training with experiment tracking."""

import json
import torch
import random
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import DPOTrainer
from datasets import Dataset
from typing import Dict, List


# Fixed random seed for reproducibility
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dpo_dataset(json_path: str) -> Dataset:
    """Load DPO preference pairs from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to HuggingFace dataset format
    dataset_dict = {
        'prompt': [],
        'chosen': [],
        'rejected': []
    }
    
    for item in data:
        dataset_dict['prompt'].append(item['prompt'])
        dataset_dict['chosen'].append(item['chosen'])
        dataset_dict['rejected'].append(item['rejected'])
    
    return Dataset.from_dict(dataset_dict)


def train_dpo_model(
    model_name: str = "codellama/CodeLlama-7b-hf",
    train_dataset_path: str = "dpo_preferences.json",
    output_dir: str = "./dpo_model",
    learning_rate: float = 5e-7,
    num_epochs: int = 1,
    batch_size: int = 4,
    max_length: int = 1024,
    beta: float = 0.1,  # DPO temperature parameter
    seed: int = RANDOM_SEED,
    log_file: str = None
):
    """
    Train model using DPO on preference pairs.
    
    Args:
        model_name: HuggingFace model identifier
        train_dataset_path: Path to JSON with preference pairs
        output_dir: Where to save fine-tuned model
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        batch_size: Batch size (keep small for 7B model)
        max_length: Max sequence length
        beta: DPO temperature (controls strength of preference learning)
        seed: Random seed for reproducibility
        log_file: Path to save training log (optional)
    """
    # Set all random seeds
    set_seed(seed)
    
    config = {
        'model_name': model_name,
        'train_dataset_path': train_dataset_path,
        'output_dir': output_dir,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'max_length': max_length,
        'beta': beta,
        'seed': seed,
        'gradient_accumulation_steps': 4
    }
    
    print("="*60)
    print("DPO TRAINING CONFIGURATION")
    print("="*60)
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print("="*60)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for DPO
    
    # Load dataset
    print("Loading preference dataset...")
    train_dataset = load_dpo_dataset(train_dataset_path)
    print(f"Loaded {len(train_dataset)} preference pairs")
    
    # Create reference model (frozen copy of base model)
    print("Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 4*4 = 16
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_code_hallucination",
        seed=seed,
        data_seed=seed,
        report_to="none"  # Disable wandb/tensorboard by default
    )
    
    # Initialize DPO trainer
    print("\nInitializing DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        beta=beta,
        max_length=max_length,
        max_prompt_length=512,
    )
    
    # Train
    print("\nStarting training...")
    print("="*60)
    dpo_trainer.train()
    
    # Save final model
    print("\nSaving model...")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration
    config_path = f"{output_dir}/training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved training config to: {config_path}")
    
    # Save training log if specified
    if log_file:
        training_log = {
            'config': config,
            'num_preference_pairs': len(train_dataset),
            'status': 'completed'
        }
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"Saved training log to: {log_file}")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to: {output_dir}")
    print(f"{'='*60}")


def train_with_lora(
    model_name: str = "codellama/CodeLlama-7b-hf",
    train_dataset_path: str = "dpo_preferences.json",
    output_dir: str = "./dpo_model_lora",
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    max_length: int = 1024,
    beta: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = RANDOM_SEED,
    log_file: str = None
):
    """
    More memory-efficient training using LoRA adapters.
    Recommended for resource-constrained environments.
    """
    from peft import LoraConfig, get_peft_model
    
    # Set all random seeds
    set_seed(seed)
    
    config = {
        'model_name': model_name,
        'train_dataset_path': train_dataset_path,
        'output_dir': output_dir,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'max_length': max_length,
        'beta': beta,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'seed': seed,
        'gradient_accumulation_steps': 4
    }
    
    print("="*60)
    print("DPO TRAINING WITH LORA")
    print("="*60)
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print("="*60)
    
    # Load base model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    train_dataset = load_dpo_dataset(train_dataset_path)
    print(f"Loaded {len(train_dataset)} preference pairs")
    
    # Reference model (base model without LoRA)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_code_hallucination_lora",
        seed=seed,
        data_seed=seed,
        report_to="none"
    )
    
    # DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        beta=beta,
        max_length=max_length,
        max_prompt_length=512,
    )
    
    print("\nStarting training...")
    print("="*60)
    dpo_trainer.train()
    
    # Save
    print("\nSaving model...")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration
    config_path = f"{output_dir}/training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved training config to: {config_path}")
    
    # Save training log if specified
    if log_file:
        training_log = {
            'config': config,
            'num_preference_pairs': len(train_dataset),
            'status': 'completed'
        }
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"Saved training log to: {log_file}")
    
    print(f"\n{'='*60}")
    print(f"LoRA adapters saved to: {output_dir}")
    print(f"To use: Load base model + adapters")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DPO model for code generation")
    parser.add_argument("--dataset", type=str, default="dpo_preferences.json",
                        help="Path to preference dataset")
    parser.add_argument("--output", type=str, default="./dpo_model",
                        help="Output directory for model")
    parser.add_argument("--model", type=str, default="codellama/CodeLlama-7b-hf",
                        help="Base model name")
    parser.add_argument("--lora", action="store_true",
                        help="Use LoRA for memory-efficient training")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--log-file", type=str,
                        help="Path to save training log")
    
    args = parser.parse_args()
    
    if args.lora:
        print("Using LoRA training (recommended for limited GPU memory)")
        train_with_lora(
            model_name=args.model,
            train_dataset_path=args.dataset,
            output_dir=args.output,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            beta=args.beta,
            seed=args.seed,
            log_file=args.log_file
        )
    else:
        print("Using full fine-tuning (requires more GPU memory)")
        train_dpo_model(
            model_name=args.model,
            train_dataset_path=args.dataset,
            output_dir=args.output,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            beta=args.beta,
            seed=args.seed,
            log_file=args.log_file
        )
