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

npz_file_rl = np.load("svd_rank_qwen_kl.npz")
npz_file_ft = np.load("svd_rank_qwen_sft.npz")
npz_file_pretrained = np.load("svd_rank_qwen_pre.npz")

# QKV projections (exclude o_proj, gate_proj, up_proj, down_proj)
qkv_names = ["q_proj", "k_proj", "v_proj"]
# FFN projections (exclude o_proj, q_proj, k_proj, v_proj) 
ffn_names = ["gate_proj", "up_proj", "down_proj"]


save_name = "rank_qwen_kl_vs_sft"

first_name,second_name = "RL (KL)","SFT"

# Process RL QKV
bar_rl_qkv = []
for key in npz_file_rl.keys():
    if any(qkv_name in key for qkv_name in qkv_names):
        arr_rl = npz_file_rl[key]
        arr_pretrained = npz_file_pretrained[key]
        rank_rl = top_k(arr_rl, 0.99)
        rank_pre = top_k(arr_pretrained, 0.99)
        
        info_pre = info_k(arr_pretrained, rank_pre)
        info_rl = info_k(arr_rl, rank_pre)
        
        info_inc = (info_rl - info_pre)
        bar_rl_qkv.append(info_inc)

# Process FT QKV  
bar_ft_qkv = []
for key in npz_file_ft.keys():
    if any(qkv_name in key for qkv_name in qkv_names):
        arr_ft = npz_file_ft[key]
        arr_pretrained = npz_file_pretrained[key]
        rank_ft = top_k(arr_ft, 0.99)
        rank_pre = top_k(arr_pretrained, 0.99)
        
        info_pre = info_k(arr_pretrained, rank_pre)
        info_ft = info_k(arr_ft, rank_pre)
        
        info_inc = (info_ft - info_pre)
        bar_ft_qkv.append(info_inc)

# Process RL FFN
bar_rl_ffn = []
for key in npz_file_rl.keys():
    if any(ffn_name in key for ffn_name in ffn_names):
        arr_rl = npz_file_rl[key]
        arr_pretrained = npz_file_pretrained[key]
        rank_rl = top_k(arr_rl, 0.99)
        rank_pre = top_k(arr_pretrained, 0.99)
        
        info_pre = info_k(arr_pretrained, rank_pre)
        info_rl = info_k(arr_rl, rank_pre)
        
        info_inc = (info_rl - info_pre)
        bar_rl_ffn.append(info_inc)

# Process FT FFN
bar_ft_ffn = []
for key in npz_file_ft.keys():
    if any(ffn_name in key for ffn_name in ffn_names):
        arr_ft = npz_file_ft[key]
        arr_pretrained = npz_file_pretrained[key]
        rank_ft = top_k(arr_ft, 0.99)
        rank_pre = top_k(arr_pretrained, 0.99)
        
        info_pre = info_k(arr_pretrained, rank_pre)
        info_ft = info_k(arr_ft, rank_pre)
        
        info_inc = (info_ft - info_pre)
        bar_ft_ffn.append(info_inc)

# Convert to numpy arrays
bar_rl_qkv = np.array(bar_rl_qkv)
bar_ft_qkv = np.array(bar_ft_qkv)
bar_rl_ffn = np.array(bar_rl_ffn)
bar_ft_ffn = np.array(bar_ft_ffn)

# Plot all 4 arrays together
plt.figure(figsize=(12, 6))
x_values = range(len(bar_rl_qkv))

plt.rcParams.update({'font.size': 14})  # Increase base font size

plt.plot(x_values, bar_rl_qkv, marker='o', linewidth=2, markersize=6, 
         linestyle='-', color='red', alpha=0.7, label=f'{first_name} QKV')
plt.plot(x_values, bar_ft_qkv, marker='s', linewidth=2, markersize=6, 
         linestyle='-', color='blue', alpha=0.7, label=f'{second_name} QKV')
plt.plot(x_values, bar_rl_ffn, marker='o', linewidth=2, markersize=6, 
         linestyle='-', color='lightcoral', alpha=0.7, label=f'{first_name} FFN')
plt.plot(x_values, bar_ft_ffn, marker='s', linewidth=2, markersize=6, 
         linestyle='-', color='skyblue', alpha=0.7, label=f'{second_name} FFN')

plt.title(f"SVD Information Loss/Gain Across Layers", fontsize=20)
plt.xlabel("Layer wise module index", fontsize=18)
plt.ylabel("Information Difference", fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

print("Saving ....")
plt.savefig(f"bar_plots/rank_line/{save_name}.png", dpi=300, bbox_inches='tight')
plt.close()


