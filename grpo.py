from trl import GRPOConfig, GRPOTrainer
from dataloader import GSM8KDataLoader
import wandb
import os
import torch
from transformers import HfArgumentParser
import argparse
import json

# Parse DeepSpeed CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--deepspeed", type=str, default=None)
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
parser.add_argument("--num_generations", type=int, default=1)
parser.add_argument("--bf16", type=str, default="true")
parser.add_argument("--gradient_checkpointing", type=str, default="true")
args = parser.parse_args()

# Handle boolean arguments
args.bf16 = args.bf16.lower() == "true"
args.gradient_checkpointing = args.gradient_checkpointing.lower() == "true"

# Initialize distributed training
local_rank = args.local_rank
if local_rank != -1:
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    is_main_process = local_rank == 0
else:
    is_main_process = True

# Load and format GSM8K dataset
dataloader = GSM8KDataLoader()
dataloader.load_data()
formatted_train = dataloader.format_prompts(split='train')

# Define custom reward function
def reward_math_solving(completions, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        ground_truth = kwargs['answer'][i].split('#### ')[-1].strip()
        model_answer = None

        if "\\boxed{" in completion and "}" in completion:
            start_idx = completion.find("\\boxed{") + len("\\boxed{")
            end_idx = completion.find("}", start_idx)
            model_answer = completion[start_idx:end_idx].strip()
            if model_answer:
                model_answer = model_answer.replace(',', '')
                try:
                    model_answer = str(int(model_answer))
                except ValueError:
                    model_answer = None

        reward = 0
        if model_answer:
            reward += 1
        if model_answer == ground_truth:
            reward += 10

        rewards.append(reward)

    avg_reward = sum(rewards) / len(rewards)
    if is_main_process:
        wandb.log({"avg_reward": avg_reward})

    return rewards

# Define GRPO training config with memory optimizations
# Create a safe config for wandb logging
safe_config = {
    "output_dir": "Qwen3-1.7B-GSM8K-GRPO",
    "num_train_epochs": 4,
    "num_generations": args.num_generations,
    "per_device_train_batch_size": args.per_device_train_batch_size,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "max_completion_length": 256,
    "max_prompt_length": 256,
    "bf16": args.bf16,
    "gradient_checkpointing": args.gradient_checkpointing,
}

# Initialize wandb (only on main process)
if is_main_process:
    wandb.init(
        project="gsm8k-grpo",
        name="qwen3-1.7b-gsm8k-grpo",
        config=safe_config,  # Use safe config instead
    )

# Now create the actual training args
training_args = GRPOConfig(
    output_dir="Qwen3-1.7B-GSM8K-GRPO",
    num_train_epochs=4,
    num_generations=args.num_generations,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    max_completion_length=512,
    max_prompt_length=256,
    bf16=args.bf16,
    logging_steps=10,
    remove_unused_columns=False,
    gradient_checkpointing=args.gradient_checkpointing,
    optim="adamw_torch_fused",
    deepspeed=args.deepspeed,  # Just pass the path, don't load the config
    local_rank=local_rank,
    ddp_find_unused_parameters=False,
)

# Initialize trainer without the deepspeed_config parameter
trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=reward_math_solving,
    args=training_args,
    train_dataset=formatted_train,
)

# Run training
if __name__ == "__main__":
    trainer.train()