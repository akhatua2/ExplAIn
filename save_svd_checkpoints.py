import torch
from transformers import AutoModelForCausalLM
import numpy as np
from tqdm import tqdm


checkpoints = [500, 1000, 1500, 2000, 2500, 3000, 3500]

for checkpoint in checkpoints:

    model = AutoModelForCausalLM.from_pretrained(f"/matx/u/rahulsc/ExplAIn/Qwen3-1.7B-GSM8K-SFT/checkpoint-{checkpoint}/", trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    model = model.to("cuda:1")

    dic = {}
    for name, param in tqdm(model.named_parameters()):

        param.requires_grad = False

        if "proj" in name:
            S = torch.linalg.svdvals(param)
            dic[name] = S.cpu().numpy()

    np.savez(f"svd_rank_ft_{checkpoint}.npz", **dic)



        

