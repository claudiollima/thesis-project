"""
Dataset classes for deepfake detection training.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class DeepfakeDataset(Dataset):
    """
    Generic deepfake detection dataset.
    
    Expects directory structure:
        root/
            real/
                image1.jpg
                image2.jpg
            fake/
                image1.jpg
                image2.jpg
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        split: str = "train",
    ):
        self.root = Path(root)
        self.transform = transform
        self.split = split

        self.samples: List[Tuple[Path, int]] = []

        # Load real images (label = 0)
        real_dir = self.root / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    self.samples.append((img_path, 0))

        # Load fake images (label = 1)
        fake_dir = self.root / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    self.samples.append((img_path, 1))

        print(f"Loaded {len(self.samples)} samples from {root}")
        print(f"  Real: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"  Fake: {sum(1 for _, l in self.samples if l == 1)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Default transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


def get_train_transforms(img_size: int = 224) -> A.Compose:
    """Training augmentations for robustness."""
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """Validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def create_dataloaders(
    train_root: str,
    val_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = DeepfakeDataset(
        train_root,
        transform=get_train_transforms(img_size),
        split="train",
    )
    val_dataset = DeepfakeDataset(
        val_root,
        transform=get_val_transforms(img_size),
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
