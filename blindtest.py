import os
import io
import json
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

import torch
from transformers.generation import GenerateDecoderOnlyOutput
from transformers import (ChameleonProcessor, ChameleonForConditionalGeneration,
                          LlavaProcessor, LlavaForConditionalGeneration, 
                          IdeficsProcessor, IdeficsForVisionText2Text,
                          Kosmos2Processor, Kosmos2ForConditionalGeneration)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

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

class MMM:
    def __init__(self, args) -> None:
        if args.model_name_or_path not in ('chameleon-7b', 'chameleon-30b', 'llava-1.5-7b-hf', 
                                           'idefics-9b', 'idefics-9b-instruct', 'kosmos-2-patch14-224'):
            raise Exception('model_name_or_path')
        if args.device not in ('cuda', 'auto', 'cuda:1', 'cuda:2', 'cuda:3'):
            raise Exception('device_map')
        if args.batch_size < 1:
            raise Exception('batch size')
        if args.dtype not in ('float32', 'bfloat16'):
            raise Exception('dtype')
        if args.access_token is None:
            raise Exception('access token')

        self.model_name_or_path = args.model_name_or_path
        self.device_map = args.device
        self.batch_size = args.batch_size
        self.max_new_tokens = 1
        self.dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
        self.viz = args.viz
        self.return_dict = True if args.viz else False
        self.output_attentions = True if self.viz else False
        self.access_token = args.access_token

        self.__prepare_model()

    def __prepare_model(self) -> None:
        model_kwargs = {'return_dict_in_generate': self.return_dict,
                        'output_attentions': self.output_attentions,
                        'max_new_tokens': self.max_new_tokens}

        if self.model_name_or_path in ['chameleon-7b', 'chameleon-30b']:
            preprocessing = self.__chameleon_preprocessing_fn
            postprocessing = self.__chameleon_postprocessing_fn
            processor = ChameleonProcessor.from_pretrained(f"facebook/{self.model_name_or_path}", token=self.access_token)
            model = ChameleonForConditionalGeneration.from_pretrained(f"facebook/{self.model_name_or_path}", pad_token_id=processor.tokenizer.eos_token_id,
                                                                      torch_dtype=self.dtype, device_map=self.device_map, token=self.access_token)
        elif self.model_name_or_path in ['llava-1.5-7b-hf']:
            preprocessing = self.__llava_preprocessing_fn
            postprocessing = self.__llava_postprocessing_fn
            processor = LlavaProcessor.from_pretrained(f"llava-hf/{self.model_name_or_path}", patch_size=14, vision_feature_select_strategy='default')
            model = LlavaForConditionalGeneration.from_pretrained(f"llava-hf/{self.model_name_or_path}", torch_dtype=self.dtype, device_map=self.device_map)
        elif self.model_name_or_path in ['idefics-9b', 'idefics-9b-instruct']:
            preprocessing = self.__idefics_preprocessing_fn
            postprocessing = self.__idefics_postprocessing_fn
            processor = IdeficsProcessor.from_pretrained(f"HuggingFaceM4/{self.model_name_or_path}")
            model = IdeficsForVisionText2Text.from_pretrained(f"HuggingFaceM4/{self.model_name_or_path}", torch_dtype=self.dtype, device_map=self.device_map)
        elif self.model_name_or_path in ['kosmos-2-patch14-224']:
            preprocessing = self.__kosmos_preprocessing_fn
            postprocessing = self.__kosmos_postprocessing_fn
            processor = Kosmos2Processor.from_pretrained(f"microsoft/{self.model_name_or_path}", padding_side='left')
            model = Kosmos2ForConditionalGeneration.from_pretrained(f"microsoft/{self.model_name_or_path}", torch_dtype=self.dtype, device_map=self.device_map)
            model_kwargs.update({'image_embeds': None, 'use_cache': True})

        setattr(self, 'preprocessing_fn', preprocessing)
        setattr(self, 'postprocessing_fn', postprocessing)
        setattr(self, 'model', model)
        setattr(self, 'processor', processor)
        setattr(self, 'model_kwargs', model_kwargs)

    def prepare_input(self, data):
        return self.preprocessing_fn(data)
    
    def prepare_output(self, data):
        return self.postprocessing_fn(data)
    
    def __chameleon_preprocessing_fn(self, data):
        task, img_info, prompt, gt, md = data

        if self.viz:
            input_prompt = "<image> Describe the scene in the image."
        else:
            input_prompt = prompt + "<image>"

        img_bytes = img_info['bytes']
        img_bytes = io.BytesIO(img_bytes)
        input_img = Image.open(img_bytes)

        return (task, input_img, input_prompt, prompt, gt, md)
    
    def __chameleon_postprocessing_fn(self, data):
        prompt, gen_output = data
        return gen_output[len(prompt):]
    
    def __llava_preprocessing_fn(self, data):
        task, img_info, prompt, gt, md = data

        if self.viz:
            input_prompt = "USER: <image>\nDescribe the scene in the image. ASSISTANT:"
        else:
            input_prompt = f"USER: <image>\n{prompt} ASSISTANT:"

        img_bytes = img_info['bytes']
        img_bytes = io.BytesIO(img_bytes)
        input_img = Image.open(img_bytes)

        return (task, input_img, input_prompt, prompt, gt, md)

    def __llava_postprocessing_fn(self, data):
        _, gen_output = data
        return gen_output.split('ASSISTANT: ')[-1]
    
    def __idefics_preprocessing_fn(self, data):
        task, img_info, prompt, gt, md = data

        img_bytes = img_info['bytes']
        img_bytes = io.BytesIO(img_bytes)
        input_img = Image.open(img_bytes)

        input_prompt = [
            "User:",
            input_img,
            f"{prompt}\nAssistant:",
        ]

        return (task, input_img, input_prompt, prompt, gt, md)

    def __idefics_postprocessing_fn(self, data):
        _, gen_output = data
        return gen_output.split('Assistant:')[-1]
    
    def __kosmos_preprocessing_fn(self, data):
        task, img_info, prompt, gt, md = data

        input_prompt = f"<grounding> {prompt}"

        img_bytes = img_info['bytes']
        img_bytes = io.BytesIO(img_bytes)
        input_img = Image.open(img_bytes)

        return (task, input_img, input_prompt, prompt, gt, md)

    def __kosmos_postprocessing_fn(self, data):
        prompt, gen_output = data
        caption, entities = self.processor.post_process_generation(gen_output)
        return caption[len(prompt):]
    

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
            inputs = mmm.processor(prompts=input_prompts, padding=True, return_tensors="pt").to(mmm.model.device, dtype=mmm.dtype)
        else:
            inputs = mmm.processor(text=input_prompts, images=input_imgs, padding=True, return_tensors="pt").to(mmm.model.device, dtype=mmm.dtype)

        if mmm.model_name_or_path in ['kosmos-2-patch14-224']:
            outputs = mmm.model.generate(pixel_values=inputs["pixel_values"],
                                        input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"],
                                        image_embeds_position_mask=inputs["image_embeds_position_mask"],
                                        **mmm.model_kwargs)
        else:
            outputs = mmm.model.generate(**inputs, **mmm.model_kwargs)
        
        if isinstance(outputs, GenerateDecoderOnlyOutput):
            gen_ids = outputs.sequences
        else:
            gen_ids = outputs

        gen_outputs = mmm.processor.batch_decode(gen_ids, skip_special_tokens=True)
        
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
        if not task == 'Touching Circles':
            continue

        if not task in results.keys():
            results[task] = []

        task, input_img, input_prompt, prompt, gt, md = mmm.prepare_input([task, img_info, prompt, gt, md]) 

        if mmm.model_name_or_path in ['idefics-9b', 'idefics-9b-instruct']:
            inputs = mmm.processor(prompts=[input_prompt], return_tensors="pt").to(mmm.model.device, dtype=mmm.dtype)
        else:
            inputs = mmm.processor(text=input_prompt, images=input_img, return_tensors="pt").to(mmm.model.device, dtype=mmm.dtype)

        if mmm.model_name_or_path in ['kosmos-2-patch14-224']:
            output = mmm.model.generate(pixel_values=inputs["pixel_values"],
                                        input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"],
                                        image_embeds_position_mask=inputs["image_embeds_position_mask"],
                                        **mmm.model_kwargs)
        else:
            output = mmm.model.generate(**inputs, **mmm.model_kwargs)
        
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

            gen_output = mmm.processor.decode(gen_ids, skip_special_tokens=True)
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