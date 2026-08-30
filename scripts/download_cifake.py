#!/usr/bin/env python3
"""
Download CIFAKE dataset - real CIFAR-10 images paired with AI-generated equivalents.

Dataset: https://huggingface.co/datasets/birgermoell/cifake
- Real: 60,000 CIFAR-10 images
- Fake: 60,000 Stable Diffusion generated images with same class labels

This is a perfect dataset for initial experiments:
1. Balanced real/fake
2. Small image size (32x32 or upscaled)
3. Well-documented baseline
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Download CIFAKE dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cifake",
        help="Output directory"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max samples per class (real/fake)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    # Create directories
    for split_dir in [train_dir, val_dir]:
        (split_dir / "real").mkdir(parents=True, exist_ok=True)
        (split_dir / "fake").mkdir(parents=True, exist_ok=True)
    
    print("Loading CIFAKE dataset from HuggingFace...")
    
    # Load dataset - using dragonintelligence version which is in standard format
    dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="test")
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Features: {dataset.features}")
    
    # Shuffle for random sampling
    dataset = dataset.shuffle(seed=42)
    
    # Track counts
    real_count = 0
    fake_count = 0
    max_per_class = args.max_samples
    train_size_per_class = int(max_per_class * args.train_ratio)
    
    print(f"\nSampling {max_per_class} images per class...")
    print(f"  Train: {train_size_per_class} per class")
    print(f"  Val: {max_per_class - train_size_per_class} per class")
    
    for sample in tqdm(dataset, desc="Processing"):
        img = sample["image"]
        label = sample["label"]  # 0=real, 1=fake
        
        # Skip if we have enough of this class
        if label == 0 and real_count >= max_per_class:
            continue
        if label == 1 and fake_count >= max_per_class:
            continue
        
        # Determine split
        if label == 0:
            is_train = real_count < train_size_per_class
            count = real_count
            real_count += 1
        else:
            is_train = fake_count < train_size_per_class
            count = fake_count
            fake_count += 1
        
        split_dir = train_dir if is_train else val_dir
        label_name = "real" if label == 0 else "fake"
        
        # Save image
        if isinstance(img, Image.Image):
            # Upscale from 32x32 to 224x224 for models
            img_upscaled = img.resize((224, 224), Image.Resampling.LANCZOS)
            img_upscaled.save(split_dir / label_name / f"{label_name}_{count:05d}.jpg", quality=95)
        
        # Check if done
        if real_count >= max_per_class and fake_count >= max_per_class:
            break
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset Summary:")
    for split in ["train", "val"]:
        split_dir = train_dir if split == "train" else val_dir
        real = len(list((split_dir / "real").glob("*")))
        fake = len(list((split_dir / "fake").glob("*")))
        print(f"  {split}: {real} real, {fake} fake")
    print("="*50)
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ CIFAKE dataset ready")
        print("Run: python train.py --train-dir data/cifake/train --val-dir data/cifake/val")
