#!/usr/bin/env python3
"""
GLiNER Training Script for NEREL Dataset.

Fine-tunes a GLiNER model on NEREL data converted to GLiNER format.

Usage:
    python train_gliner.py \
        --train_data /path/to/train.json \
        --val_data /path/to/dev.json \
        --output_dir ./models/gliner-nerel \
        --base_model urchade/gliner_multi-v2.1 \
        --max_steps 10000 \
        --batch_size 8
"""

import json
import os
import argparse
from typing import List, Dict, Any, Optional

import torch
from gliner import GLiNER


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load GLiNER format dataset from JSON file."""
    with open(path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {path}")
    return data


def validate_dataset(data: List[Dict[str, Any]], name: str = "dataset") -> None:
    """Validate dataset format."""
    if not data:
        raise ValueError(f"{name} is empty")

    sample = data[0]
    if "tokenized_text" not in sample:
        raise ValueError(f"{name} missing 'tokenized_text' field")
    if "ner" not in sample:
        raise ValueError(f"{name} missing 'ner' field")

    # Count entities
    total_entities = sum(len(d.get("ner", [])) for d in data)
    print(f"  Total entities in {name}: {total_entities}")


def train_gliner(
    train_data_path: str,
    val_data_path: Optional[str],
    output_dir: str,
    base_model: str = "urchade/gliner_multi-v2.1",
    max_steps: int = 10000,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    others_lr: float = 5e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    save_steps: int = 1000,
    eval_steps: int = 500,
    logging_steps: int = 100,
    save_total_limit: int = 3,
    focal_loss_alpha: float = 0.75,
    focal_loss_gamma: float = 0.0,
    negatives: float = 1.0,
    device: str = "auto",
    seed: int = 42
):
    """
    Fine-tune GLiNER on custom dataset.

    Args:
        train_data_path: Path to training data JSON
        val_data_path: Path to validation data JSON (optional)
        output_dir: Directory to save trained model
        base_model: Base GLiNER model to fine-tune
        max_steps: Maximum training steps
        batch_size: Training batch size
        learning_rate: Encoder learning rate
        others_lr: Learning rate for other components
        warmup_ratio: Warmup ratio for learning rate scheduler
        weight_decay: Weight decay for regularization
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
        save_total_limit: Maximum checkpoints to keep
        focal_loss_alpha: Focal loss alpha parameter
        focal_loss_gamma: Focal loss gamma parameter
        negatives: Negative sampling ratio
        device: Device to train on
        seed: Random seed
    """
    # Set seed
    torch.manual_seed(seed)

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_data = load_dataset(train_data_path)
    validate_dataset(train_data, "train")

    val_data = None
    if val_data_path and os.path.exists(val_data_path):
        val_data = load_dataset(val_data_path)
        validate_dataset(val_data, "validation")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load base model
    print(f"\nLoading base model: {base_model}")
    model = GLiNER.from_pretrained(base_model)
    print("Model loaded successfully")

    # Train the model
    print("\nStarting training...")
    print(f"  Max steps: {max_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Output dir: {output_dir}")

    try:
        trainer = model.train_model(
            train_dataset=train_data,
            eval_dataset=val_data,
            output_dir=output_dir,

            # Training schedule
            max_steps=max_steps,
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,

            # Batch & optimization
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            others_lr=others_lr,
            weight_decay=weight_decay,
            others_weight_decay=weight_decay,
            max_grad_norm=1.0,

            # Loss configuration
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            loss_reduction="sum",
            negatives=negatives,

            # Logging & saving
            save_steps=save_steps,
            eval_steps=eval_steps if val_data else None,
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,

            # Misc
            seed=seed,
            report_to="none",  # Disable wandb by default
        )

        # Save final model
        print("\nSaving final model...")
        trainer.save_model(output_dir)

        # Save training config
        config = {
            "base_model": base_model,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "others_lr": others_lr,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "focal_loss_alpha": focal_loss_alpha,
            "focal_loss_gamma": focal_loss_gamma,
            "negatives": negatives,
            "train_data": train_data_path,
            "val_data": val_data_path,
            "train_examples": len(train_data),
            "val_examples": len(val_data) if val_data else 0,
        }
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w', encoding='UTF-8') as f:
            json.dump(config, f, indent=2)
        print(f"Training config saved to {config_path}")

        print("\nTraining complete!")
        print(f"Model saved to: {output_dir}")

        return trainer

    except Exception as e:
        print(f"\nTraining error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GLiNER on NEREL dataset"
    )

    # Data paths
    parser.add_argument(
        '--train_data',
        type=str,
        required=True,
        help="Path to training data JSON"
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default=None,
        help="Path to validation data JSON"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./models/gliner-nerel',
        help="Output directory for trained model"
    )

    # Model
    parser.add_argument(
        '--base_model',
        type=str,
        default='urchade/gliner_multi-v2.1',
        help="Base GLiNER model to fine-tune"
    )

    # Training hyperparameters
    parser.add_argument(
        '--max_steps',
        type=int,
        default=10000,
        help="Maximum training steps"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help="Encoder learning rate"
    )
    parser.add_argument(
        '--others_lr',
        type=float,
        default=5e-5,
        help="Learning rate for other components"
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help="Weight decay"
    )

    # Checkpointing
    parser.add_argument(
        '--save_steps',
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=500,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=100,
        help="Log every N steps"
    )
    parser.add_argument(
        '--save_total_limit',
        type=int,
        default=3,
        help="Maximum checkpoints to keep"
    )

    # Loss configuration
    parser.add_argument(
        '--focal_loss_alpha',
        type=float,
        default=0.75,
        help="Focal loss alpha"
    )
    parser.add_argument(
        '--focal_loss_gamma',
        type=float,
        default=0.0,
        help="Focal loss gamma"
    )
    parser.add_argument(
        '--negatives',
        type=float,
        default=1.0,
        help="Negative sampling ratio"
    )

    # Misc
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help="Device to train on (auto, cpu, cuda)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    train_gliner(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        others_lr=args.others_lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,
        negatives=args.negatives,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
