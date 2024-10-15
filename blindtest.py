import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from transformers.generation import GenerateDecoderOnlyOutput

from mmm import MMM

df = pd.read_parquet("hf://datasets/XAI/vlmsareblind/data/valid-00000-of-00001.parquet")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default='chameleon-7b', help='Huggingface model name or path')
parser.add_argument("--device", type=str, default='cuda', help='cuda for single gpu | auto for multiple gpus')
parser.add_argument("--batch_size", type=int, default=1, help='Batch size')
parser.add_argument("--dtype", type=str, default='float32', help='float32 | bfloat16')
parser.add_argument("--viz", action='store_true', help='Visualizatino for the last layer')
parser.add_argument("--return_dict", action="store_true", help='Return dictionary outputs in model.generate(...)')
parser.add_argument("--access_token", type=str, default=None, help='Huggingface access token')
args = parser.parse_args()
    

def batch_inference(mmm):
    results = {}
    for df_idx in tqdm(range(0, len(df), mmm.batch_size), desc='batch'):
        batch_df = df.iloc[df_idx:df_idx+mmm.batch_size].to_dict('split')
        batch_data = list(map(mmm.prepare_input, batch_df['data']))
        
        tasks = []
        prompts = []
        input_prompts = []
        input_imgs = []
        gts = []
        for task, input_img, input_prompt, prompt, gt, md in batch_data:
            if not task in results.keys():
                results[task] = []
            tasks.append(task)
            input_imgs.append(input_img)
            input_prompts.append(input_prompt)
            prompts.append(prompt)
            gts.append(gt)

        if mmm.model_name_or_path in ['idefics-9b', 'idefics-9b-instruct']:
            inputs = mmm.processor(prompts=input_prompts, padding=True)
        else:
            inputs = mmm.processor(text=input_prompts, images=input_imgs, padding=True)

        if mmm.model_name_or_path in ['kosmos-2-patch14-224']:
            outputs = mmm.generate(pixel_values=inputs["pixel_values"],
                                   input_ids=inputs["input_ids"],
                                   attention_mask=inputs["attention_mask"],
                                   image_embeds_position_mask=inputs["image_embeds_position_mask"],)
        else:
            outputs = mmm.generate(**inputs)
        
        if isinstance(outputs, GenerateDecoderOnlyOutput):
            gen_ids = outputs.sequences
        else:
            gen_ids = outputs

        gen_outputs = mmm.batch_decode(gen_ids)
        
        for idx in range(len(gen_outputs)):
            answer = mmm.prepare_output([prompts[idx], gen_outputs[idx]])
            result = {'row_idx': df_idx+idx, 'prompt': prompts[idx], 'gt': gts[idx], 'output': gen_outputs[idx], 'answer': answer}
            results[tasks[idx]].append(result)

    with open(f"blindtest_{mmm.model_name_or_path}_{mmm.dtype.__str__().lstrip('torch.')}.json", 'w') as file:
        json.dump(results, file)
        

def save_fig(data):
    (folder_path, head_idx, input_prompt, input_img, head_q_att) = data
    
    resized_head_q_att = Image.fromarray(head_q_att).resize(input_img.size, resample=Image.BILINEAR)
    resized_head_q_att = np.array(resized_head_q_att)

    plt.figure(figsize=(10,10))
    plt.imshow(input_img)
    plt.imshow(resized_head_q_att, cmap='jet', alpha=0.5)
    plt.colorbar(label='Attention Score')
    plt.title(f'{input_prompt}')
    plt.axis('off')

    plt.savefig(folder_path + f'/{head_idx}.png', dpi=300)
    plt.close()

    return True

def inference(mmm):
    results = {}
    for row_idx, (task, img_info, prompt, gt, md) in tqdm(df.iterrows(), total=df.shape[0]):
        if not task in results.keys():
            results[task] = []

        task, input_img, input_prompt, prompt, gt, md = mmm.prepare_input([task, img_info, prompt, gt, md]) 

        if mmm.model_name_or_path in ['idefics-9b', 'idefics-9b-instruct']:
            inputs = mmm.processor(prompts=input_prompt)
        else:
            inputs = mmm.processor(text=input_prompt, images=input_img)

        if mmm.model_name_or_path in ['kosmos-2-patch14-224']:
            output = mmm.generate(pixel_values=inputs["pixel_values"],
                                  input_ids=inputs["input_ids"],
                                  attention_mask=inputs["attention_mask"],
                                  image_embeds_position_mask=inputs["image_embeds_position_mask"],)
        else:
            output = mmm.generate(**inputs)
        
        if isinstance(output, GenerateDecoderOnlyOutput):
            last_layer_head_atts = output.attentions[0][-1].cpu().numpy()

            head_num, tok_num = last_layer_head_atts.shape[1], last_layer_head_atts.shape[2]
            
            folder_path = f'viz_data/{task}/{row_idx}'
            Path(folder_path).mkdir(parents=True, exist_ok=True)

            head_q_att_list = []
            for head_idx in range(head_num):
                head_q_att = last_layer_head_atts[0,head_idx,-3,1:1025].reshape(32,32) # chameleon Query -3, num_img_tok 1024 = 32 x 32
                # head_q_att = last_layer_head_atts[0,head_idx,-7,5:581].reshape(24,24) # llava Query -7, num_img_tok 576 = 24 x 24
                head_q_att_list.append([folder_path, head_idx, input_prompt, input_img, head_q_att])
                
            _ = list(map(save_fig, head_q_att_list))
        else:
            gen_ids = output[0]

            gen_output = mmm.decode(gen_ids)
            answer = mmm.prepare_output([prompt, gen_output])

            result = {'row_idx': row_idx, 'prompt': prompt, 'gt': gt, 'output': gen_output, 'answer': answer}
            results[task].append(result)
    
    if not mmm.viz:
        with open(f"blindtest_{mmm.model_name_or_path}_{mmm.dtype.__str__().lstrip('torch.')}.json", 'w') as file:
            json.dump(results, file)


def main():
    mmm = MMM(args)

    if mmm.batch_size == 1:
        inference(mmm)
    else:
        batch_inference(mmm)

if __name__ == '__main__':
    main()