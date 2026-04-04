#!/usr/bin/env python3
"""
Prepare sample dataset from Midjourney AI-generated images.
"""

import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def main():
    output_dir = Path("data/midjourney_sample")
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    # Create directories
    for split_dir in [train_dir, val_dir]:
        (split_dir / "real").mkdir(parents=True, exist_ok=True)
        (split_dir / "fake").mkdir(parents=True, exist_ok=True)
    
    print("Loading Midjourney AI-generated images dataset...")
    
    # Use streaming to avoid downloading entire dataset
    dataset = load_dataset(
        "ideepankarsharma2003/AIGeneratedImages_Midjourney",
        split="train",
        streaming=True,
    )
    
    # Collect samples
    max_samples = 1000  # Keep it small for quick testing
    train_size = int(0.8 * max_samples)
    
    real_count = 0
    fake_count = 0
    
    print(f"Downloading {max_samples} samples...")
    
    for i, sample in enumerate(tqdm(dataset, total=max_samples)):
        if i >= max_samples:
            break
            
        split = "train" if i < train_size else "val"
        split_dir = train_dir if i < train_size else val_dir
        
        img = sample["image"]
        label = sample["label"]  # Check what label format is used
        
        # Determine if real or fake based on label
        # Common conventions: 0=real, 1=fake or "real"/"fake" strings
        if isinstance(label, str):
            is_fake = label.lower() in ["fake", "ai", "generated", "synthetic", "1"]
        else:
            is_fake = bool(label)
        
        label_name = "fake" if is_fake else "real"
        
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
            filename = f"{label_name}_{i:05d}.jpg"
            img.save(split_dir / label_name / filename, quality=95)
            
            if is_fake:
                fake_count += 1
            else:
                real_count += 1
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset Summary:")
    print(f"  Total: {real_count} real, {fake_count} fake")
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
        print("\n✓ Sample dataset ready at data/midjourney_sample/")
        print("Run: python train.py --data-dir data/midjourney_sample")
