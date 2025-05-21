import json
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import numpy as np


grpo = json.load(open("per_sample/grpo.json"))
sft = json.load(open("per_sample/sft.json"))


# model_name = "Qwen3-1.7B-GSM8K-GRPO/checkpoint-932/"
model_name = "Qwen3-1.7B-GSM8K-SFT-boxed/checkpoint-3736/"
# model_name = "Qwen/Qwen3-1.7B"

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

remove_names = ["the","a",",","in","."]
remove_ids = [tokenizer.encode(i)[0] for i in remove_names]

print(remove_ids)
# assert False

stop_ids = [27275, 532,92]


words = []

for i in range(10):
    # print(sft[i]['prompt'])
    # print(grpo[i]['prompt'])

    find_id = 0
    stop_id = -1
    start_id = -1

    if sft[i]['is_correct']==False and grpo[i]['is_correct']==True:

        # prompt_text = grpo[i]['model_response']
        prompt_text = sft[i]['model_response']

        # print(prompt_text)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        input_ids = inputs.input_ids.cpu().numpy().squeeze().tolist()
        for index, i in enumerate(input_ids):
            print(i, tokenizer.decode(i))
            if i==79075:
                find_id+=1
                if find_id==2:
                    start_id = index
            
            if find_id==2 and i in stop_ids:
                stop_id = index
                break
        
        for i in input_ids:
            words.append(tokenizer.decode(i))

            



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

