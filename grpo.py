# train_grpo.py
from trl import GRPOConfig, GRPOTrainer
from dataloader import GSM8KDataLoader

# Load and format GSM8K dataset
dataloader = GSM8KDataLoader()
dataloader.load_data()
formatted_train = dataloader.format_prompts(split='train')

# Define custom reward function
def reward_math_solving(completions, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        ground_truth = kwargs['answer'][i].split('####')[-1].strip()
        model_answer = None
        if "<answer>" in completion and "</answer>" in completion:
            start_idx = completion.find("<answer>") + len("<answer>")
            end_idx = completion.find("</answer>")
            model_answer = completion[start_idx:end_idx].strip()
        reward = 0
        if model_answer:
            reward += 1
        if model_answer == ground_truth:
            reward += 10
        rewards.append(reward)
    return rewards

# Define GRPO training config
training_args = GRPOConfig(
    output_dir="Qwen3-1.7B-GSM8K-GRPO",
    num_train_epochs=4,
    num_generations=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    max_completion_length=256,
    max_prompt_length=512,
    bf16=True,
    logging_steps=10,
    remove_unused_columns=False,
)

# Initialize trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=reward_math_solving,
    args=training_args,
    train_dataset=formatted_train,
)

# Run training
if __name__ == "__main__":
    trainer.train()
