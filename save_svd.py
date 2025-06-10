import torch
from transformers import AutoModelForCausalLM
import numpy as np
from tqdm import tqdm


#Lets save all svd again
#1. unsloth/Llama-3.2-1B-Instruct
#2. Llama-3.2-1B-Instruct-no-KL-GSM8K-GRPO/checkpoint-932/
#3. Llama-3.2-1B-Instruct-GSM8K-GRPO/checkpoint-932/
#4. Llama-3.2-1B-Instruct-GSM8K-SFT-boxed-each-epoch/checkpoint-3740/

#1. Qwen/Qwen3-1.7B
#2. Qwen3-1.7B-no-KL-GSM8K-GRPO/checkpoint-932/
#3. Qwen3-1.7B-GSM8K-GRPO/checkpoint-932/
#4. Qwen3-1.7B-GSM8K-SFT-boxed-each-epoch/checkpoint-3740/

#1. qwen_test_map_pre_
#2. 

model_name = "Qwen3-1.7B-GSM8K-SFT-boxed-each-epoch/checkpoint-3740/"
save_name = "svd_rank_qwen_sft.npz"
# model = AutoModelForCausalLM.from_pretrained("ExplAIn/Qwen3-1.7B-GSM8K-SFT-boxed/checkpoint-3736/", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
model = model.to("cuda:1")

dic = {}
for name, param in tqdm(model.named_parameters()):

    param.requires_grad = False

    if "proj" in name:
        S = torch.linalg.svdvals(param)
        dic[name] = S.cpu().numpy()

np.savez(save_name, **dic)



    

