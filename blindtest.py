import os
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
parser.add_argument("--viz_ptype", type=str, default='base', help='[base | common | task-general] prompt for visualization')
parser.add_argument("--viz_row_idx", type=int, default=0, help="Data index in the blindtest dataset")
parser.add_argument("--viz_q_indice", type=int, default=-1, help="Query index in attention sequences")
parser.add_argument("--access_token", type=str, default=None, help='Huggingface access token')
args = parser.parse_args()
    

def save_json(mmm, data, folder_path=f'blindtest_data_{len(df)/1000:.1f}K'):
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    base_f_name = f"blindtest_{mmm.model_name_or_path}_{mmm.dtype.__str__().lstrip('torch.')}"
    extension = '.json'

    num_exs_files = len([f for f in os.listdir(folder_path) if f.startswith(base_f_name) and f.endswith(extension)])
    if num_exs_files == 0:
        new_f_name = base_f_name + extension
    else:
        new_f_name = f"{folder_path}/{base_f_name}_({num_exs_files})" + extension

    with open(new_f_name, 'w') as file:
        json.dump(data, file)

def viz_inference(mmm):
    if mmm.viz_row_idx is None:
        raise Exception('Assign a row idx')
    if mmm.model_name_or_path in ('idefics-9b', 'idefics-9b-instruct'):
        raise Exception('Not supported model')
    
    task, img_info, prompt, gt, md = df.iloc[mmm.viz_row_idx]
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
        output = mmm.generate(**inputs, return_legacy_cache=True)

    if isinstance(output, GenerateDecoderOnlyOutput):
        folder_path = f'viz_data/{mmm.model_name_or_path}/{task}/{mmm.viz_row_idx}'
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        last_layer_head_atts = output.attentions[0][-1].cpu().numpy()
        num_head, num_tok = last_layer_head_atts.shape[1], last_layer_head_atts.shape[2]

        fig, axes = plt.subplots(5, 8, figsize=(20,15))
        for i in range(num_head):
            ax = axes[i // 8, i % 8]
            ax.imshow(last_layer_head_atts[0, i, :, :], cmap='jet', interpolation='nearest')
            ax.set_title(f'Head {i}')
            ax.axis('off')

        for fn_idx, (fn, fn_name) in enumerate(zip([np.sum, np.mean, np.max], ['Sum', 'Mean', 'Max'])):
            ax = axes[4, fn_idx]
            ax.imshow(fn(last_layer_head_atts[0], axis=0), cmap='jet', interpolation='nearest')
            ax.set_title(fn_name)
            ax.axis('off')

        for i in [3, 4, 5, 6, 7]:
            plt.delaxes(axes[4,i])

        fig.suptitle(input_prompt, fontsize=8)
        plt.tight_layout()
        plt.savefig(folder_path + f'/{mmm.viz_ptype}_att.png', dpi=300)
        plt.close()

        if mmm.do_q_att:
            for i in range(num_head):
                head_q_att = last_layer_head_atts[0, i, mmm.viz_q_indice, mmm.img_tok_slice].reshape(mmm.img_tok_size, mmm.img_tok_size)

                resized_head_q_att = Image.fromarray(head_q_att).resize(input_img.size, resample=Image.BILINEAR)
                resized_head_q_att = np.array(resized_head_q_att)

                plt.figure(figsize=(10,10))
                plt.imshow(input_img)
                plt.imshow(resized_head_q_att, cmap='jet', alpha=0.5)
                plt.colorbar(label='Att Score')
                plt.title(input_prompt, fontsize=8)
                plt.axis('off')

                plt.savefig(folder_path + f'/{mmm.viz_ptype}_q_att_{i}.png')
                plt.close()

        with open(folder_path + f'/{mmm.viz_ptype}_tokens.txt', 'w') as file:
            input_prompt_tokens = mmm.convert_ids_to_tokens(inputs['input_ids'][0])
            file.write(str(input_prompt_tokens))
    else:
        raise Exception('Activate an agrument for viz')


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

    save_json(mmm, results)
        

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
            gen_ids = output.sequences[0]
        else:
            gen_ids = output[0]

            gen_output = mmm.decode(gen_ids)
            answer = mmm.prepare_output([prompt, gen_output])

            result = {'row_idx': row_idx, 'prompt': prompt, 'gt': gt, 'output': gen_output, 'answer': answer}
            results[task].append(result)

    save_json(mmm, results)

def main():
    mmm = MMM(args)
    if args.viz:
        viz_inference(mmm)
    elif mmm.batch_size == 1:
        inference(mmm)
    else:
        batch_inference(mmm)

if __name__ == '__main__':
    main()