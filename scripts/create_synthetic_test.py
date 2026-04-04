#!/usr/bin/env python3
"""
Create synthetic test dataset for pipeline verification.
Generates simple images with different characteristics for real/fake.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random


def create_real_like_image(size=224):
    """Create image with natural-looking noise patterns."""
    # Natural photos have complex, correlated noise
    img = np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
    
    # Add gaussian blur for natural smoothness
    pil_img = Image.fromarray(img)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Add some structure (gradients like in real photos)
    arr = np.array(pil_img)
    gradient = np.linspace(0, 50, size).reshape(1, -1, 1)
    arr = np.clip(arr + gradient, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)


def create_fake_like_image(size=224):
    """Create image with AI-like artifacts."""
    # AI images often have smoother regions, weird edges
    img = Image.new('RGB', (size, size), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    
    # Add geometric shapes (AI tends to create cleaner geometry)
    for _ in range(5):
        x1, y1 = random.randint(0, size//2), random.randint(0, size//2)
        x2, y2 = x1 + random.randint(20, 100), y1 + random.randint(20, 100)
        color = tuple(random.randint(50, 200) for _ in range(3))
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # Add slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return img


def main():
    output_dir = Path("data/synthetic_test")
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    # Create directories
    for split_dir in [train_dir, val_dir]:
        (split_dir / "real").mkdir(parents=True, exist_ok=True)
        (split_dir / "fake").mkdir(parents=True, exist_ok=True)
    
    random.seed(42)
    np.random.seed(42)
    
    n_train_per_class = 400
    n_val_per_class = 100
    
    print("Creating synthetic test dataset...")
    
    # Train set
    for i in range(n_train_per_class):
        real_img = create_real_like_image()
        fake_img = create_fake_like_image()
        
        real_img.save(train_dir / "real" / f"real_{i:04d}.jpg", quality=95)
        fake_img.save(train_dir / "fake" / f"fake_{i:04d}.jpg", quality=95)
    
    # Val set
    for i in range(n_val_per_class):
        real_img = create_real_like_image()
        fake_img = create_fake_like_image()
        
        real_img.save(val_dir / "real" / f"real_{i:04d}.jpg", quality=95)
        fake_img.save(val_dir / "fake" / f"fake_{i:04d}.jpg", quality=95)
    
    print(f"\nCreated synthetic test dataset:")
    print(f"  Train: {n_train_per_class} real, {n_train_per_class} fake")
    print(f"  Val: {n_val_per_class} real, {n_val_per_class} fake")
    print(f"\nLocation: {output_dir}")
    print("\nNote: This is for PIPELINE TESTING only.")
    print("Replace with real deepfake dataset for actual experiments.")


if __name__ == "__main__":
    main()
