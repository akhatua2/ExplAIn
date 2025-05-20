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
# def relative_norm_inc(arr_ft, arr_pre):
#     arr_pre_norm = np.linalg.norm(arr_pre, axis=1)

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

npz_file_ft = np.load("svd_rank_rl.npz")
npz_file_pretrained = np.load("svd_rank_pretrained.npz")


avg = 0
# arr = []
bar = []

# all_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# all_names = ["all_till_layer_3"]


for key in npz_file_ft.keys():
    arr = npz_file_ft[key]
    arr_pretrained = npz_file_pretrained[key]
    rank_ft = top_k(arr, 0.99)
    rank_pre = top_k(arr_pretrained, 0.99)

    avg += rank_ft - rank_pre

    print(key, rank_pre, rank_ft, rank_ft - rank_pre)
    

#
# print(avg/len(npz_file_ft.keys()))

    # info_pre = info_k(arr_pretrained, rank_pre)
    # info_ft = info_k(arr, rank_pre)

    # info_inc = info_ft - info_pre
    # # rank_pretrained = top_k(arr_pretrained,0.99)
    # # print(rank_ft, rank_pretrained, arr.shape[0])
    # bar.append(info_inc/arr.shape[0])
        # if (get_layer_num(key) > 3):
        #     break
    

    # bar = np.array(bar)
    # plot_bar(bar, save_name=f"bar_plots/{name}_info_inc.png", title=f"SVD Rank across Layers: {name}", xlabel="Layer", ylabel="Rank", figsize=(12, 6))

    