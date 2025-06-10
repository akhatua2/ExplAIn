import json
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def save_attention_heatmap(attention_matrix, filename="attention_map.png"):
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Self-Attention Map')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

grpo = json.load(open("per_sample/qwen_rl.json"))
sft = json.load(open("per_sample/qwen_sft_full.json"))


print(len(grpo), len(sft))

# assert False

# model_name = "Qwen/Qwen3-1.7B"
# model_name = "Qwen3-1.7B-no-KL-GSM8K-GRPO/checkpoint-932/"
# model_name = "Qwen3-1.7B-GSM8K-GRPO/checkpoint-932/"
# model_name = "Qwen3-1.7B-GSM8K-SFT-boxed-each-epoch/checkpoint-3740/"

model_name = "unsloth/Llama-3.2-1B-Instruct"
model_name = "Llama-3.2-1B-Instruct-no-KL-GSM8K-GRPO/checkpoint-932/"
model_name = "Llama-3.2-1B-Instruct-GSM8K-GRPO/checkpoint-932/"
model_name = "Llama-3.2-1B-Instruct-GSM8K-SFT-boxed-each-epoch/checkpoint-3740/"

save_name = "llama_test_map_sft_"

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



remove_names = ["the","a",",","in","."]
remove_ids = [tokenizer.encode(i)[0] for i in remove_names]

print(remove_ids)
# assert False

stop_ids = [27275, 532,92]


words = []

min_len = 10000
min_index = -1

list_arr = [82, 134, 949]

dic_len = {}

for i in list_arr:
# for i in range(len(grpo)):
    

    if sft[i]['is_correct']==False and grpo[i]['is_correct']==True:

        prompt_text = grpo[i]['prompt']

        # print(prompt_text)
        # print("---")

        dic_len[i] = len(prompt_text)
        continue
        




        # prompt_text = sft[i]['prompt'] + sft[i]['model_response']
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    
        if inputs.input_ids.shape[1] < min_len:
            min_len = inputs.input_ids.shape[1]
            min_index = i

    
        # input_ids = inputs.input_ids.cpu().numpy().squeeze().tolist()

        # print(prompt_text)
        # for index, i in enumerate(input_ids):
        #     print(i, tokenizer.decode(i))
        
        
        outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate = True,
                # output_scores = True,
                output_attentions = True,
                # output_attentions = True,
        )

        print(len(outputs.attentions[0]))
        print(outputs.attentions[0][0].shape)

        arr = outputs.attentions[0][0].cpu().numpy().squeeze()

        arr = np.mean(arr, axis = 0) #(67, 67)

        # save_attention_heatmap(arr, "test_map_sft.png")

        np.savez(f"{save_name}{i}.npy", arr=arr)
        continue


        assert False

        attn_list = []

        for i in range(28):
            attn_tensor = outputs.attentions[0][i]
            attn_list.append(attn_tensor)
            
        all_layer_attn = torch.cat(attn_list, dim=0) #(28, 16, PROMPT_LEN, PROMPT_LEN)
        all_layer_attn_mean = torch.mean(all_layer_attn, dim=(0,1)).squeeze() #(PROMPT_LEN, PROMPT_LEN)

        all_layer_attn_mean_np = all_layer_attn_mean.cpu().numpy()

        print(all_layer_attn_mean_np.shape)

        
        attn_I_pay = all_layer_attn_mean_np[start_id:stop_id+1, :]
        # attn_I_pay = all_layer_attn_mean_np[:, start_id:stop_id+1]
        print(attn_I_pay.shape)
        attn_I_pay_np = np.mean(attn_I_pay, axis=0)

        attn_I_pay_np[0] = 0
        attn_I_pay_np[start_id:stop_id+1] = 0
        s = np.sum(attn_I_pay_np)
        attn_I_pay_np = attn_I_pay_np / s

        print(start_id, stop_id)

        print(attn_I_pay_np.shape)
        print(len(words))

        from circuitsvis.tokens import colored_tokens

        vis_html_object = colored_tokens(words, attn_I_pay_np.tolist())

        # Get the HTML string
        html_output = str(vis_html_object)
        # Alternatively, you could use:
        # html_output = vis_html_object.show_code()

        # Save it to a file
        with open("visualization_sft.html", "w") as f:
            f.write(html_output)

        break


# print(min_index)
# print(min_len)

# dic_len = dict(sorted(dic_len.items(), key=lambda item: item[1]))

# # print(len(dic_len))

print(dic_len)



#134, 949




# assert False