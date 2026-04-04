#!/usr/bin/env python3
"""
Prepare a sample dataset using COCO-SD format.
Downloads AI-generated images from available sources.
"""

import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO


def download_sample_images():
    """Download sample real/fake images for testing."""
    output_dir = Path("data/test_sample")
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    # Create directories
    for split_dir in [train_dir, val_dir]:
        (split_dir / "real").mkdir(parents=True, exist_ok=True)
        (split_dir / "fake").mkdir(parents=True, exist_ok=True)
    
    print("Loading CIFAKE dataset...")
    # CIFAKE: CIFAR-10 images labeled as real vs AI-generated
    try:
        dataset = load_dataset("ceyda/cifake-real-and-fake", split="train")
        print(f"Loaded {len(dataset)} samples")
        
        # Split into train/val
        dataset = dataset.shuffle(seed=42)
        train_size = int(0.8 * len(dataset))
        
        for i, sample in enumerate(tqdm(dataset, desc="Processing")):
            split = "train" if i < train_size else "val"
            split_dir = train_dir if i < train_size else val_dir
            
            img = sample["image"]
            label = sample["label"]  # 0 = real, 1 = fake
            
            label_name = "fake" if label == 1 else "real"
            if isinstance(img, Image.Image):
                img.save(split_dir / label_name / f"{label_name}_{i:05d}.jpg")
        
    except Exception as e:
        print(f"CIFAKE failed: {e}")
        print("\nTrying alternative: jlbaker361/ai-vs-real...")
        
        try:
            dataset = load_dataset("jlbaker361/ai-vs-real", split="train")
            print(f"Loaded {len(dataset)} samples")
            
            dataset = dataset.shuffle(seed=42)
            # Take first 2000 samples for quick testing
            max_samples = min(2000, len(dataset))
            train_size = int(0.8 * max_samples)
            
            for i, sample in enumerate(tqdm(dataset.select(range(max_samples)), desc="Processing")):
                split = "train" if i < train_size else "val"
                split_dir = train_dir if i < train_size else val_dir
                
                img = sample["image"]
                label = sample["label"]  # 0 = real, 1 = AI
                
                label_name = "fake" if label == 1 else "real"
                if isinstance(img, Image.Image):
                    img = img.convert("RGB")
                    img.save(split_dir / label_name / f"{label_name}_{i:05d}.jpg")
        except Exception as e2:
            print(f"Alternative also failed: {e2}")
            return False
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset Summary:")
    for split in ["train", "val"]:
        split_dir = train_dir if split == "train" else val_dir
        real_count = len(list((split_dir / "real").glob("*")))
        fake_count = len(list((split_dir / "fake").glob("*")))
        print(f"  {split}: {real_count} real, {fake_count} fake")
    print("="*50)
    return True


if __name__ == "__main__":
    success = download_sample_images()
    if success:
        print("\n✓ Sample dataset ready at data/test_sample/")
        print("Run: python train.py --data-dir data/test_sample")
    else:
        print("\n✗ Failed to download dataset")
