from dataloader import GSM8KDataLoader
import re
from tqdm import tqdm
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_answer(text):
    """Extract the answer from the model's response using <answer> tags"""
    match = re.search(r'<answer>(.*?)</answer>', text)
    if match:
        return match.group(1).strip()
    return None

def extract_numeric_answer(text):
    """Extract the numeric answer from text"""
    # First try to get answer from tags
    answer = extract_answer(text)
    if answer:
        # Extract the last number from the answer
        numbers = re.findall(r'\d+', answer)
        if numbers:
            return int(numbers[-1])
    return None

def evaluate_model(model_name, num_samples=None, batch_size=8):
    """
    Evaluate a model on GSM8K test set, using all available GPUs
    
    Args:
        model_name (str): Name of the Hugging Face model
        num_samples (int): Number of samples to evaluate (None for full test set)
        batch_size (int): Batch size for processing examples
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print(f"Loading model {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set tokenizer to left padding for decoder-only models
    tokenizer.padding_side = 'left'
    
    # Load model on all available GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Use all available GPUs
        torch_dtype=torch.float16,  # Use half precision for better memory efficiency
        trust_remote_code=True
    )
    
    # Load data
    dataloader = GSM8KDataLoader()
    dataloader.load_data()
    test_data = dataloader.format_prompts(split='test')
    
    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))
    
    correct = 0
    total = len(test_data)
    
    # Create batches
    batches = [range(i, min(i + batch_size, len(test_data))) for i in range(0, len(test_data), batch_size)]
    
    # Run inference on test set in batches
    for batch_indices in tqdm(batches, desc="Evaluating"):
        # Get examples for this batch
        batch_examples = [test_data[i] for i in batch_indices]
        
        # Prepare batch inputs
        prompts = [example['prompt'] for example in batch_examples]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Process each example in the batch
        for i, output in enumerate(outputs):
            # Decode response
            response = tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract predicted answer
            pred_answer = extract_numeric_answer(response)
            
            # Get ground truth answer
            true_answer = int(batch_examples[i]['answer'].split('#### ')[-1])
            
            # Compare answers
            if pred_answer == true_answer:
                correct += 1
    
    # Calculate metrics
    accuracy = correct / total
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Model name")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for processing examples")
    
    args = parser.parse_args()
    
    print(f"Evaluating {args.model}")
    print(f"Using device_map='auto' to utilize all available GPUs")
    print(f"Batch size: {args.batch_size}")
    
    metrics = evaluate_model(args.model, args.samples, args.batch_size)
    
    print(f"Test Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")
