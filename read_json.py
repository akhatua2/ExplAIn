import json
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import numpy as np


grpo = json.load(open("per_sample/grpo.json"))
sft = json.load(open("per_sample/sft.json"))


# model_name = "Qwen3-1.7B-GSM8K-GRPO/checkpoint-932/"
# model_name = "Qwen3-1.7B-GSM8K-SFT-boxed/checkpoint-3736/"
model_name = "Qwen/Qwen3-1.7B"

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map="auto",  # Use all available GPUs
        torch_dtype=torch.float16,  # Use half precision for better memory efficiency
        trust_remote_code=True
    )

device = "cuda:0"
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model.eval()

grpo_over_sft = []
dic = {}

save_name = "pretrained_logits.npz"

for i in range(10):
    # print(sft[i]['prompt'])
    # print(grpo[i]['prompt'])

    # if sft[i]['is_correct']==False and grpo[i]['is_correct']==True:
    if True:

        # print(" ---- sft ----")
        # print(sft[i]['model_response'])
        # print(" ---- grpo ----")
        # print(grpo[i]['model_response'])
        # print("----------")
        # print(grpo[i]['prompt'])
        # print("----------")
        # grpo_over_sft.append(i)
        prompt_text = grpo[i]['prompt']

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate = True,
                output_scores = True,

                # output_attentions = True,

        )

        # print(len(outputs.scores))
        generated_token_ids = outputs.sequences[:, inputs.input_ids.shape[-1]:]

        # print(inputs.input_ids.shape)
        print(len(outputs.scores))
        # print(outputs.sequences.shape)
        # assert False
        scores = torch.concat(outputs.scores, dim=0)
        step_logprobs = F.log_softmax(scores, dim=-1) # Shape: (512, vocab_size)
        # print(generated_token_ids.shape)

        
        length_generated = generated_token_ids.shape[-1]
        first_arr = torch.arange(length_generated).squeeze()
        scores_sampled = step_logprobs[first_arr, generated_token_ids.squeeze()]

        dic[str(i)] = scores_sampled.cpu().numpy()

np.savez(save_name, **dic)
# print(len(grpo_over_sft))
# print(grpo_over_sft)
# print(len(a))

# print(a[0].keys())

