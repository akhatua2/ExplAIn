from dataloader import GSM8KDataLoader
import re
from tqdm import tqdm
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_answer_tags(text):
    """Extract the answer from the model's response using <answer></answer> tags"""
    # Find the last occurrence of answer tags in case the model repeats the prompt
    matches = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL))
    if matches:
        # Get the last match (most recent answer)
        return matches[-1].group(1).strip()
    return None

def extract_numeric_answer_tags(text):
    """Extract the numeric answer from text using <answer></answer> tags"""
    # First try to get answer from answer tags format
    answer = extract_answer_tags(text)
    if answer:
        # Look for numbers with dollar signs first (most specific)
        dollar_amounts = re.findall(r'\$(\d+(?:,\d+)*)', answer)
        if dollar_amounts:
            # Convert the last dollar amount to integer
            return int(dollar_amounts[-1].replace(',', ''))
        
        # If no dollar amounts, look for any numbers
        numbers = re.findall(r'\d+(?:,\d+)*', answer)
        if numbers:
            # Convert the last number to integer
            return int(numbers[-1].replace(',', ''))
    return None

def extract_answer(text):
    """Extract the answer from the model's response using boxed format"""
    # Find the last occurrence of boxed answer in case the model repeats the prompt
    matches = list(re.finditer(r'\\boxed{([^}]*)}', text))
    if matches:
        # Get the last match (most recent answer)
        return matches[-1].group(1).strip()
    return None

def extract_numeric_answer(text):
    """Extract the numeric answer from text"""
    # First try to get answer from boxed format
    answer = extract_answer(text)
    if answer:
        # Look for numbers with dollar signs first (most specific)
        dollar_amounts = re.findall(r'\$(\d+(?:,\d+)*)', answer)
        if dollar_amounts:
            # Convert the last dollar amount to integer
            return int(dollar_amounts[-1].replace(',', ''))
        
        # If no dollar amounts, look for any numbers
        numbers = re.findall(r'\d+(?:,\d+)*', answer)
        if numbers:
            # Convert the last number to integer
            return int(numbers[-1].replace(',', ''))
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
    # test_data = dataloader.format_prompts_llama_few_shot(split='test')
    test_data = dataloader.format_prompts(split='test')

    
    
    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))
    
    correct = 0
    total = len(test_data)
    
    # Create batches
    batches = [range(i, min(i + batch_size, len(test_data))) for i in range(0, len(test_data), batch_size)]
    
    per_sample_results = [] # To store results for each sample

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
                pad_token_id=tokenizer.pad_token_id,
                # return_dict_in_generate = True,
                # output_attentions = True,

            )

            # print(outputs.keys())
            # print(outputs.sequences.shape)
            # print(type(outputs))
            # print(type(outputs.attentions))
            # print(len(outputs.attentions))
            # print(type(outputs.attentions))
            # print(len(outputs.attentions[1]))
            
            # print("----------")
            
            # print(outputs.attentions[0][0].shape)
            # print(outputs.attentions[0][1].shape)
            # print(outputs.attentions[0][2].shape)

            # print("----------")

            # print(outputs.attentions[1][0].shape)
            # print(outputs.attentions[1][1].shape)
            # print(outputs.attentions[1][2].shape)

            # print("----------")

            # print(outputs.attentions[2][0].shape)
            # print(outputs.attentions[2][1].shape)
            # print(outputs.attentions[2][2].shape)

            # print("----------")
            # print(outputs.attentions[511][2].shape)
            # # print(outputs.attentions[0].shape)
            # # print(outputs)
            # assert False
        

        
        prompt_len = inputs.input_ids.shape[1]
        
        # Process each example in the batch
        for i, output_tensor in enumerate(outputs): # Changed 'output' to 'output_tensor' for clarity
            # Decode response
            # output_tensor_seq = output_tensor.sequences
            # print(output_tensor.shape)
            # assert False

            # prompt_length = len(prompts[i])
            response = tokenizer.decode(output_tensor[prompt_len:], skip_special_tokens=True)

            # npy_dic = {}

            # npy_dic['attn_prompt'] = torch.stack(outputs.attentions[0], dim=0).cpu().numpy() #! (layers, 128, 16, 164, 164)

            # for i in range(512-164):
            #     npy_dic[f'other_attn_{i}'] = torch.stack(outputs.attentions[i+164], dim=0).cpu().numpy()

            


            # Extract predicted answer
            pred_answer = extract_numeric_answer(response)
            
            # Get ground truth answer
            true_answer = int(batch_examples[i]['answer'].split('#### ')[-1].replace(',', ''))
            
            # Determine if correct
            is_correct = pred_answer == true_answer

            # Store per-sample details
            original_data_index = batch_indices[i]
            per_sample_results.append({
                "id": original_data_index,
                "prompt": prompts[i], # Adding prompt for better context
                "model_response": response, # Adding full model response
                "predicted_answer": pred_answer,
                "true_answer": true_answer,
                "is_correct": is_correct,
                "size" : output_tensor.shape[0]
            })
            
            # Print model output and comparison
            # print("\nExample", i + 1)
            # print("Prompt:", batch_examples[i]['prompt'])
            # print("Model Response:", response)
            # print("Predicted Answer:", pred_answer)
            # print("Ground Truth:", true_answer)
            # print("Correct:", pred_answer == true_answer)
            # print("-" * 50)
            
            # Compare answers
            if pred_answer == true_answer:
                correct += 1
    
    # Calculate metrics
    accuracy = correct / total
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'per_sample_results': per_sample_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Model name")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for processing examples")

    parser.add_argument("--json_name", type=str, default="sft_llama_256", help="Batch size for processing examples")
    
    args = parser.parse_args()
    
    print(f"Evaluating {args.model}")
    print(f"Using device_map='auto' to utilize all available GPUs")
    print(f"Batch size: {args.batch_size}")
    
    metrics = evaluate_model(args.model, args.samples, args.batch_size)
    
    print(f"Test Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")

    # Alternative: measure in tokens instead of characters
    response_lengths = [sample['size'] for sample in metrics['per_sample_results']]
    avg_response_length = sum(response_lengths) / len(response_lengths)
    print(f"Average Response Length: {avg_response_length:.1f} tokens")

    # You can now access metrics['per_sample_results']
    # For example, to print the first 5 sample results:
    # print("\nFirst 5 sample results:")
    # for i, sample_res in enumerate(metrics['per_sample_results'][:5]):
    #     print(f"Sample ID: {sample_res['id']}, Correct: {sample_res['is_correct']}, Predicted: {sample_res['predicted_answer']}, True: {sample_res['true_answer']}")
    # If you want to save these results to a file (e.g., JSON):
    import json
    print("len of metric saved: ",len(metrics['per_sample_results']))
    # save_name = "sft_llama_256_few_shot"
    save_name = args.json_name
    with open(f'per_sample/{save_name}.json', 'w') as f:
        json.dump(metrics['per_sample_results'], f, indent=4)
    print(f"Saved per-sample results to {save_name}.json")
