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

def relative_L2(arr_ft, arr_pre):
    arr_ft, arr_pre = arr_ft.detach().cpu().numpy(), arr_pre.detach().cpu().numpy()
    diff_norm = np.linalg.norm(arr_pre - arr_ft, ord = 'fro')
    pre_norm = np.linalg.norm(arr_pre, ord = 'fro')
    # arr_ft_norm = np.linalg.norm(arr_ft, ord = 'fro')
    return (diff_norm)*100.0/pre_norm

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
def plot_line(data, save_name="save_fig.png", title="Line Plot", xlabel="Index", ylabel="Value", figsize=(10, 6), marker='o', linewidth=2, markersize=6):

    plt.figure(figsize=figsize)
    x_values = range(len(data))
    
    # Plot line with markers
    plt.plot(x_values, data, marker=marker, linewidth=linewidth, markersize=markersize, 
             linestyle='-', markerfacecolor='blue', markeredgecolor='darkblue', 
             color='blue', alpha=0.7)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("Saving ....")
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


# model_rl = AutoModelForCausalLM.from_pretrained("ExplAIn/Qwen3-1.7B-GSM8K-GRPO/checkpoint-932/", trust_remote_code=True)

# model_rl = AutoModelForCausalLM.from_pretrained("ExplAIn/Qwen3-1.7B-no-KL-GSM8K-GRPO/checkpoint-932/", trust_remote_code=True)

# model_ft = AutoModelForCausalLM.from_pretrained("ExplAIn/Qwen3-1.7B-GSM8K-SFT-boxed-each-epoch/checkpoint-3740/", trust_remote_code=True)

# model_ft = AutoModelForCausalLM.from_pretrained("ExplAIn/Qwen3-1.7B-GSM8K-GRPO/checkpoint-932/", trust_remote_code=True)

# model_pre = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)


# --- LLAMA -------

model_pre = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct", trust_remote_code=True)

# model_ft = AutoModelForCausalLM.from_pretrained("Llama-3.2-1B-Instruct-GSM8K-SFT-boxed-each-epoch/checkpoint-3740/", trust_remote_code=True)

model_ft = AutoModelForCausalLM.from_pretrained("Llama-3.2-1B-Instruct-no-KL-GSM8K-GRPO/checkpoint-932/", trust_remote_code=True)

model_rl = AutoModelForCausalLM.from_pretrained("Llama-3.2-1B-Instruct-GSM8K-GRPO/checkpoint-932/", trust_remote_code=True)
# ExplAIn/


avg = 0
bar = []
# all_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
save_name = "llama_rl_no_kl_vs_rl_with_kl"

first_name,second_name = "RL (no KL)","RL (KL)"

all_names = ["proj"]
not_names = ["o_proj","gate_proj","up_proj","down_proj"]
# not_names = ["q_proj","k_proj","v_proj"]
# all_names = ["all_till_layer_3"]

# Get parameter names containing "proj"
name_sorted = [x[0] for x in model_rl.named_parameters()]
name_sorted = [x for x in name_sorted if "proj" in x]

# Create pairs of (layer_number, name) and sort by layer number
sorted_pairs = sorted((get_layer_num(name), name) for name in name_sorted)

# Extract just the sorted names
name_sorted = [pair[1] for pair in sorted_pairs]

print(name_sorted)



# name_sorted = [x[0] for x in name_sorted]

# print(name_sorted[0])
# assert False

for name_key in all_names:
    bar_rl = []
    # Create dictionaries for quick parameter lookup
    rl_dict = dict(model_rl.named_parameters())
    pre_dict = dict(model_pre.named_parameters())
    
    # Use the sorted names to access parameters in order
    for name in name_sorted:
        if name_key in name and not any(not_name in name for not_name in not_names):
            print(name, name)
            param_ft = rl_dict[name]
            param_pre = pre_dict[name]
            print(param_ft.shape, param_pre.shape)
            bar_rl.append(relative_L2(param_ft, param_pre))

    bar_ft = []
    ft_dict = dict(model_ft.named_parameters())
    
    # Use the same sorted names for the fine-tuned model
    for name in name_sorted:
        if name_key in name and not any(not_name in name for not_name in not_names):
            print(name, name)
            param_ft = ft_dict[name]
            param_pre = pre_dict[name]
            print(param_ft.shape, param_pre.shape)
            bar_ft.append(relative_L2(param_ft, param_pre))

    not_names = ["o_proj","q_proj","k_proj","v_proj"]
    bar_rl_ffn = []
    bar_ft_ffn = []

    # Use sorted names for FFN parameters
    for name in name_sorted:
        if name_key in name and not any(not_name in name for not_name in not_names):
            print(name, name)
            param_ft = rl_dict[name]
            param_pre = pre_dict[name]
            print(param_ft.shape, param_pre.shape)
            bar_rl_ffn.append(relative_L2(param_ft, param_pre))

    for name in name_sorted:
        if name_key in name and not any(not_name in name for not_name in not_names):
            print(name, name)
            param_ft = ft_dict[name]
            param_pre = pre_dict[name]
            print(param_ft.shape, param_pre.shape)
            bar_ft_ffn.append(relative_L2(param_ft, param_pre))

    bar_rl = np.array(bar_rl)
    bar_ft = np.array(bar_ft)
    bar_rl_ffn = np.array(bar_rl_ffn)
    bar_ft_ffn = np.array(bar_ft_ffn)
    
    # Set larger font sizes for all plot elements
    plt.rcParams.update({'font.size': 14})  # Base font size
    plt.figure(figsize=(12, 6))
    x_values = range(len(bar_rl))
    
    plt.plot(x_values, bar_rl, marker='o', linewidth=2, markersize=6, 
             linestyle='-', color='red', alpha=0.7, label=f'{first_name} QKV')
    plt.plot(x_values, bar_ft, marker='s', linewidth=2, markersize=6, 
             linestyle='-', color='blue', alpha=0.7, label=f'{second_name} QKV')
    plt.plot(x_values, bar_rl_ffn, marker='o', linewidth=2, markersize=6, 
             linestyle='-', color='lightcoral', alpha=0.7, label=f'{first_name} FFN')
    plt.plot(x_values, bar_ft_ffn, marker='s', linewidth=2, markersize=6, 
             linestyle='-', color='skyblue', alpha=0.7, label=f'{second_name} FFN')
    
    plt.title(f"Relative Norm Difference", fontsize=20)
    plt.xlabel("Layer wise module index", fontsize=18)
    plt.ylabel("% Norm Diff", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    print("Saving ....")
    plt.savefig(f"bar_plots/L2_line/{save_name}_combined_noset_all.png", dpi=300, bbox_inches='tight')
    plt.close()
