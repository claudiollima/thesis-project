#!/usr/bin/env python3
"""
Download a sample from GenImage dataset for quick pipeline testing.
Uses HuggingFace datasets library for efficient streaming.
"""

import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def main():
    output_dir = Path("data/genimage_sample")
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    # Create directories
    for split_dir in [train_dir, val_dir]:
        (split_dir / "real").mkdir(parents=True, exist_ok=True)
        (split_dir / "fake").mkdir(parents=True, exist_ok=True)
    
    print("Loading GenImage dataset (streaming)...")
    
    # Load a manageable subset - GenImage has real/AI pairs
    # Using stable-diffusion subset for quick testing
    dataset = load_dataset(
        "poloclub/diffusiondb",
        "2m_random_1k",  # Small 1k sample
        split="train",
        trust_remote_code=True,
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Split 80/20 train/val
    train_size = int(0.8 * len(dataset))
    
    # Save AI-generated images (fake)
    print("Saving AI-generated samples...")
    for i, sample in enumerate(tqdm(dataset, desc="Processing")):
        split = "train" if i < train_size else "val"
        split_dir = train_dir if i < train_size else val_dir
        
        # DiffusionDB contains AI-generated images
        img = sample["image"]
        if isinstance(img, Image.Image):
            img.save(split_dir / "fake" / f"diffusion_{i:05d}.jpg")
    
    # For real images, download a small set from LAION or use placeholder
    # For now, we'll use a second dataset for real images
    print("\nLoading real images from imagenet-sketch...")
    try:
        real_dataset = load_dataset(
            "imagenet_sketch",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        
        real_count = 0
        for sample in tqdm(real_dataset, desc="Real images", total=1000):
            if real_count >= 1000:
                break
            split = "train" if real_count < 800 else "val"
            split_dir = train_dir if real_count < 800 else val_dir
            
            img = sample["image"]
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
                img.save(split_dir / "real" / f"real_{real_count:05d}.jpg")
                real_count += 1
    except Exception as e:
        print(f"Could not load real dataset: {e}")
        print("Creating placeholder - you'll need to add real images manually")
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset Summary:")
    for split in ["train", "val"]:
        split_dir = train_dir if split == "train" else val_dir
        real_count = len(list((split_dir / "real").glob("*")))
        fake_count = len(list((split_dir / "fake").glob("*")))
        print(f"  {split}: {real_count} real, {fake_count} fake")
    print("="*50)


if __name__ == "__main__":
    main()
