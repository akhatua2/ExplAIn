from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer,SFTConfig

from dataloader import GSM8KDataLoader

# Load and format GSM8K dataset
dataloader = GSM8KDataLoader()
dataloader.load_data()
formatted_train = dataloader.format_prompts_sft(split='train')

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

training_args = SFTConfig(
    output_dir="Qwen3-1.7B-GSM8K-SFT",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_length=520,
    bf16=True,
    logging_steps=10,
    remove_unused_columns=False,
    report_to="wandb"
)

trainer = SFTTrainer(
    model,
    train_dataset=formatted_train,
    args=training_args,
)

# wandb.init(project="cs224r", name = "Qwen3-1.7B-GSM8K-SFT")

trainer.train()