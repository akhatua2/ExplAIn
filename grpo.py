from dataloader import GSM8KDataLoader
from trl import GRPOConfig, GRPOTrainer

# Initialize and load the dataset
dataloader = GSM8KDataLoader()
dataset = dataloader.load_data()

# Format the prompts for training
formatted_train = dataloader.format_prompts(split='train')

# Define the reward function for math problem solving
def reward_math_solving(completions, **kwargs):
    """
    Reward function that evaluates math problem solutions
    Returns:
    - +10 for correct answer
    - +1 for having any non-empty answer
    - 0 otherwise
    """
    rewards = []
    for i, completion in enumerate(completions):
        # Get ground truth answer from kwargs
        ground_truth = kwargs['answer'][i].split('####')[-1].strip()
        print(f"\nSample {i}:")
        print(f"Ground Truth: {ground_truth}")
        
        # Extract model's answer from tags if present
        model_answer = None
        if "<answer>" in completion and "</answer>" in completion:
            start_idx = completion.find("<answer>") + len("<answer>")
            end_idx = completion.find("</answer>")
            model_answer = completion[start_idx:end_idx].strip()
        print(f"Model's answer: {model_answer}")
        
        # Calculate reward
        reward = 0
        if model_answer:  # Non-empty answer
            reward += 1
        if model_answer == ground_truth:  # Correct answer
            reward += 10
            
        print(f"Final reward: {reward}")
        rewards.append(reward)
    
    return rewards

# Configure training arguments
training_args = GRPOConfig(
    output_dir="Qwen3-1.7B-GSM8K-GRPO",
    num_train_epochs=4,
)

# Initialize trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=reward_math_solving,
    args=training_args,
    train_dataset=formatted_train,
)

# Start training
trainer.train()