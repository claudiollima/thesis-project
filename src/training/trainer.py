"""
Training loop for deepfake detection.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


class Trainer:
    """Simple trainer for binary classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        epochs: int = 20,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
        project_name: str = "deepfake-detection",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        # Loss function with class weighting for imbalanced data
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=lr * 0.01,
        )

        # Best metrics
        self.best_auc = 0.0

        if use_wandb:
            wandb.init(project=project_name)
            wandb.watch(model)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.float().to(self.device).unsqueeze(1)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Store predictions
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

            pbar.set_postfix({"loss": loss.item()})

        # Calculate metrics
        all_preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc": accuracy_score(all_labels, all_preds_binary),
            "train_auc": roc_auc_score(all_labels, all_preds),
            "train_f1": f1_score(all_labels, all_preds_binary),
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            labels = labels.float().to(self.device).unsqueeze(1)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        all_preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
        metrics = {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc": accuracy_score(all_labels, all_preds_binary),
            "val_auc": roc_auc_score(all_labels, all_preds),
            "val_f1": f1_score(all_labels, all_preds_binary),
        }

        return metrics

    def train(self) -> None:
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()

            # Log metrics
            all_metrics = {**train_metrics, **val_metrics, "lr": self.scheduler.get_last_lr()[0]}
            
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, Train AUC: {train_metrics['train_auc']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, Val AUC: {val_metrics['val_auc']:.4f}")

            if self.use_wandb:
                wandb.log(all_metrics)

            # Save best model
            if val_metrics["val_auc"] > self.best_auc:
                self.best_auc = val_metrics["val_auc"]
                self.save_checkpoint(f"best_model.pt", val_metrics)
                print(f"New best model! AUC: {self.best_auc:.4f}")

        # Save final model
        self.save_checkpoint("final_model.pt", val_metrics)

    def save_checkpoint(self, filename: str, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_auc": self.best_auc,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
