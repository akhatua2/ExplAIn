# #!/bin/bash

# # Set visible GPUs
# export CUDA_VISIBLE_DEVICES=0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # Launch with Accelerate
#!/bin/bash

# Set critical environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/triton_cache"

# Create DeepSpeed config
cat > ds_config.json << 'EOL'
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 32,
  "steps_per_print": 10,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0
}
EOL

# Launch with DeepSpeed
deepspeed --num_gpus=4 grpo.py \
  --deepspeed ds_config.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_generations 4 \
  --bf16 true \
  --gradient_checkpointing true