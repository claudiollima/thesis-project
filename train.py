#!/usr/bin/env python3
"""
Main training script for deepfake detection.

Usage:
    python train.py --train-dir data/train --val-dir data/val
    python train.py --train-dir data/train --val-dir data/val --model efficientnet --epochs 10
"""

import argparse
from pathlib import Path

import torch

from src.models.detector import create_detector
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detector")
    parser.add_argument("--train-dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--val-dir", type=str, required=True, help="Validation data directory")
    parser.add_argument("--model", type=str, default="efficientnet", choices=["convnext", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--checkpoint-dir", type=str, default="models")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone weights")
    args = parser.parse_args()

    # Create model
    print(f"Creating {args.model} model...")
    model = create_detector(
        model_type=args.model,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dataloaders
    print(f"\nLoading data from {args.train_dir} and {args.val_dir}...")
    train_loader, val_loader = create_dataloaders(
        train_root=args.train_dir,
        val_root=args.val_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb,
    )

    # Train
    trainer.train()

    print(f"\nTraining complete! Best AUC: {trainer.best_auc:.4f}")
    print(f"Model saved to {args.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()
