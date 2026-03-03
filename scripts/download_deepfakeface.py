#!/usr/bin/env python3
"""
Download the DeepFakeFace dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/OpenRL/DeepFakeFace
Paper: "Robustness and Generalizability of Deepfake Detection: A Study with Diffusion Models"
"""

import os
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download DeepFakeFace dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/deepfakeface",
        help="Output directory for the dataset"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Specific subset to download (e.g., 'text2img', 'inpainting')"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading DeepFakeFace dataset to {output_dir}")

    # Download from HuggingFace
    snapshot_download(
        repo_id="OpenRL/DeepFakeFace",
        repo_type="dataset",
        local_dir=str(output_dir),
        ignore_patterns=["*.md", "*.gitattributes"] if args.subset else None,
    )

    print(f"\nDataset downloaded to: {output_dir}")
    print("Contents:")
    for item in output_dir.iterdir():
        if item.is_dir():
            n_files = len(list(item.glob("*")))
            print(f"  {item.name}/  ({n_files} items)")
        else:
            print(f"  {item.name}")


if __name__ == "__main__":
    main()
