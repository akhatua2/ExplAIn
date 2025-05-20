import torch
from transformers import AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def info_k(arr, k):
    arr = np.square(arr)
    total = arr.sum()
    cumsum = np.cumsum(arr)
    return cumsum[k]/total

#Relative norm increase
def relative_norm_inc(arr_ft, arr_pre):
    arr_ft, arr_pre = arr_ft.detach().cpu().numpy(), arr_pre.detach().cpu().numpy()
    arr_pre_norm = np.linalg.norm(arr_pre, ord = 'fro')
    arr_ft_norm = np.linalg.norm(arr_ft, ord = 'fro')
    return (arr_ft_norm - arr_pre_norm)*100.0/arr_pre_norm

def top_k(arr, percentage=0.9):
    arr = np.square(arr)
    total = arr.sum()
    cumsum = np.cumsum(arr)
    return np.argmax(cumsum >= percentage * total) + 1

def get_layer_num(name):
    return int(name.split('.')[2])
    
def plot_bar(data, save_name = "save_fig.png",title="Bar Plot", xlabel="Index", ylabel="Value", figsize=(10, 6)):
    
    plt.figure(figsize=figsize)
    plt.bar(range(len(data)), data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    # plt.show()
    print("Saving ....")
    plt.savefig(f"{save_name}.png")

# npz_file_ft = np.load("svd_rank_rl.npz")
# npz_file_pretrained = np.load("svd_rank_pretrained.npz")

model_rl = AutoModelForCausalLM.from_pretrained("/matx/u/rahulsc/ExplAIn/Qwen3-1.7B-GSM8K-GRPO/checkpoint-932/", trust_remote_code=True)
model_ft = AutoModelForCausalLM.from_pretrained("/matx/u/rahulsc/ExplAIn/Qwen3-1.7B-GSM8K-SFT-boxed/checkpoint-3736/", trust_remote_code=True)

model_pre = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

avg = 0
bar = []

all_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# all_names = ["all_till_layer_3"]

for name_key in all_names:

    bar = []
    for (name_ft, param_ft), (name_pre, param_pre) in zip(model_rl.named_parameters(), model_pre.named_parameters()):

        if name_key in name_ft:
            print(name_ft, name_pre)
            print(param_ft.shape, param_pre.shape)
            # print(relative_norm_inc(param_ft, param_pre))
            bar.append(relative_norm_inc(param_ft, param_pre))

    bar = np.array(bar)
    plot_bar(bar, save_name=f"bar_plots/norm/{name_key}_rl.png", title=f"Norm Diff: {name_key}", xlabel="Layer", ylabel="% Norm Diff", figsize=(12, 6))

for name_key in all_names:

    bar = []
    for (name_ft, param_ft), (name_pre, param_pre) in zip(model_ft.named_parameters(), model_pre.named_parameters()):

        if name_key in name_ft:
            print(name_ft, name_pre)
            print(param_ft.shape, param_pre.shape)
            # print(relative_norm_inc(param_ft, param_pre))
            bar.append(relative_norm_inc(param_ft, param_pre))

    bar = np.array(bar)
    plot_bar(bar, save_name=f"bar_plots/norm/{name_key}_ft.png", title=f"Norm Diff: {name_key}", xlabel="Layer", ylabel="% Norm Diff", figsize=(12, 6))





    



# for named in all_names:
#     for name, param in model_rl.named_parameters():

        
    
    



# for name in all_names:
#     bar = []
#     for key in npz_file_ft.keys():
#         if name in key:
#             arr = npz_file_ft[key]
#             arr_pretrained = npz_file_pretrained[key]
#             rank_pre = top_k(arr, 0.99)
#             info_pre = info_k(arr_pretrained, rank_pre)
#             info_ft = info_k(arr, rank_pre)

#             info_inc = info_ft - info_pre
#             # rank_pretrained = top_k(arr_pretrained,0.99)
#             # print(rank_ft, rank_pretrained, arr.shape[0])
#             bar.append(info_inc/arr.shape[0])

#     bar = np.array(bar)
#     plot_bar(bar, save_name=f"bar_plots/{name}_info_inc_rl.png", title=f"SVD Rank across Layers: {name}", xlabel="Layer", ylabel="Rank", figsize=(12, 6))

    