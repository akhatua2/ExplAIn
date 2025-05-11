#!/bin/bash

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# (Optional) Activate conda or virtualenv
# source activate explain

# Launch with Accelerate
accelerate launch --multi_gpu grpo.py
