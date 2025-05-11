# ExplAIn

A project for training language models on mathematical reasoning tasks using GRPO (Generative Reward-Penalized Optimization).

## Overview

This project implements GRPO training for the Qwen language model on the GSM8K dataset, focusing on mathematical problem-solving capabilities.

## Requirements

- Python >= 3.12
- PyTorch >= 2.7.0
- TRL >= 0.17.0
- Datasets >= 3.6.0

## Installation

1. Clone the repository
2. Install dependencies:
```bash
uv sync
```

## Usage

To train the model:

```bash
./run_grpo.sh
```

Or run directly with Python:

```bash
python grpo.py
```

## Project Structure

- `grpo.py`: Main training script implementing GRPO
- `dataloader.py`: Data loading and preprocessing for GSM8K dataset
- `run_grpo.sh`: Shell script for running the training
- `Qwen3-1.7B-GSM8K-GRPO/`: Directory for model checkpoints and outputs
