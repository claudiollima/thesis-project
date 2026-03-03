# AI-Generated Image Detection Research

Master's thesis project on detecting AI-generated images using multi-signal approaches.

## Overview

This project implements and evaluates detection methods for AI-generated images, focusing on:
- Content-level detection (pixel artifacts, frequency analysis)
- Contextual signals (metadata, spread patterns)
- Multi-modal approaches

## Datasets

- OpenFake (3M+ real, ~1M synthetic)
- DF40
- GenImage
- SocialDF

## Project Structure

```
thesis-project/
├── data/               # Datasets (gitignored)
├── models/             # Trained models
├── src/
│   ├── data/           # Data loading utilities
│   ├── models/         # Model architectures
│   ├── training/       # Training scripts
│   └── evaluation/     # Evaluation metrics
├── notebooks/          # Jupyter notebooks
├── scripts/            # Utility scripts
└── experiments/        # Experiment configs & results
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
