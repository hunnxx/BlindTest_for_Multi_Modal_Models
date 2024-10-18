import io

from PIL import Image

import torch
from transformers import (ChameleonProcessor, ChameleonForConditionalGeneration,
                          LlavaProcessor, LlavaForConditionalGeneration, 
                          IdeficsProcessor, IdeficsForVisionText2Text,
                          Kosmos2Processor, Kosmos2ForConditionalGeneration,
                          MllamaProcessor, MllamaForConditionalGeneration)


class VizConfig:
    prompts = {
        'common': 'Describe the scene in the image.',
        'Subway Connections': 'How many characters are there?',
        'Nested Squares': 'What shape is in the image?',
        'Line Plot Intersections': 'How many colors are there?',
        'Touching Circles': 'What shape is in the image?',
        'Counting Grid - Word Grids': 'Do you see any lines?',
        'Counting Grid - Blank Grids': 'Do you see any lines?',
        'Olympic Counting - Circles': 'What shape is in the image?',
        'Olympic Counting - Pentagons': 'What shape is in the image?',
        'Circled Letter': 'What word is shown in the image?'
    }
    chameleon = {
        'do_q_att': True,
        'img_tok_slice': slice(1,1025),
        'img_tok_size': 32
    }
    llama = {
        'do_q_att': False,
        'img_tok_slice': None,
        'img_tok_size': None
    }
    llava = {
        'do_q_att': True,
        'img_tok_slice': slice(5,581),
        'img_tok_size': 24
    }
    idefics = None
    kosmos = {
        'do_q_att': True,
        'img_tok_slice': slice(2,66),
        'img_tok_size': 8
    }


class MMM:
    def __init__(self, args) -> None:
        if args.model_name_or_path not in ('chameleon-7b', 'chameleon-30b', 'Llama-3.2-11B-Vision', 'Llama-3.2-90B-Vision',
                                           'Llama-3.2-11B-Vision-Instruct', 'Llama-3.2-90B-Vision-Instruct', 'Llama-Guard-3-11B-Vision',
                                           'llava-1.5-7b-hf', 'idefics-9b', 'idefics-9b-instruct', 'kosmos-2-patch14-224'):
            raise Exception('model_name_or_path')
        if args.device not in ('cuda', 'auto', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'):
            raise Exception('device_map')
        if args.batch_size < 1:
            raise Exception('batch size')
        if args.dtype not in ('float32', 'bfloat16'):
            raise Exception('dtype')
        if args.viz_ptype not in ('base', 'common', 'task-general'):
            raise Exception('viz_type: [base, common, task-general] or None')
        if args.access_token is None:
            raise Exception('access token')

        self.model_name_or_path = args.model_name_or_path
        self.device_map = args.device
        self.batch_size = args.batch_size
        self.max_new_tokens = 50
        self.dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
        self.viz = args.viz
        self.viz_ptype = args.viz_ptype
        self.viz_row_idx = args.viz_row_idx
        self.viz_q_indice = args.viz_q_indice
        self.return_dict = True if args.viz else False
        self.output_attentions = True if self.viz else False
        self.access_token = args.access_token

        self.__prepare_model()

    def __prepare_model(self) -> None:
        __model_kwargs = {'return_dict_in_generate': self.return_dict,
                          'output_attentions': self.output_attentions,
                          'max_new_tokens': self.max_new_tokens}
        __processor_kwargs = {'return_tensors': 'pt'}

        if self.model_name_or_path in ['chameleon-7b', 'chameleon-30b']:
            __preprocessing = self.__chameleon_preprocessing_fn
            __postprocessing = self.__chameleon_postprocessing_fn
            __processor = ChameleonProcessor.from_pretrained(f"facebook/{self.model_name_or_path}", token=self.access_token)
            __model = ChameleonForConditionalGeneration.from_pretrained(f"facebook/{self.model_name_or_path}", pad_token_id=__processor.tokenizer.eos_token_id,
                                                                      torch_dtype=self.dtype, device_map=self.device_map, token=self.access_token)
            __viz_config_name = 'chameleon'
        elif self.model_name_or_path in ['Llama-3.2-11B-Vision', 'Llama-3.2-90B-Vision']:
            __preprocessing = self.__mllama_preprocessing_fn
            __postprocessing = self.__mllama_postprocessing_fn
            __processor = MllamaProcessor.from_pretrained(f"meta-llama/{self.model_name_or_path}", padding_side='left', token=self.access_token)
            __model = MllamaForConditionalGeneration.from_pretrained(f"meta-llama/{self.model_name_or_path}",
                                                                     torch_dtype=self.dtype, device_map=self.device_map, token=self.access_token)
            __viz_config_name = 'llama'
        elif self.model_name_or_path in ['Llama-3.2-11B-Vision-Instruct', 'Llama-3.2-90B-Vision-Instruct', 'Llama-Guard-3-11B-Vision']:
            __preprocessing = self.__mllama_i_preprocessing_fn
            __postprocessing = self.__mllama_i_postprocessing_fn
            __processor = MllamaProcessor.from_pretrained(f"meta-llama/{self.model_name_or_path}", padding_side='left', token=self.access_token)
            __model = MllamaForConditionalGeneration.from_pretrained(f"meta-llama/{self.model_name_or_path}",
                                                                     torch_dtype=self.dtype, device_map=self.device_map, token=self.access_token)
            __model_kwargs.update({'max_new_tokens': 200})
            __processor_kwargs.update({'add_special_tokens': False})
            __viz_config_name = 'llama'
        elif self.model_name_or_path in ['llava-1.5-7b-hf']:
            __preprocessing = self.__llava_preprocessing_fn
            __postprocessing = self.__llava_postprocessing_fn
            __processor = LlavaProcessor.from_pretrained(f"llava-hf/{self.model_name_or_path}", patch_size=14, vision_feature_select_strategy='default')
            __model = LlavaForConditionalGeneration.from_pretrained(f"llava-hf/{self.model_name_or_path}", torch_dtype=self.dtype, device_map=self.device_map)
            __viz_config_name = 'llava'
        elif self.model_name_or_path in ['idefics-9b', 'idefics-9b-instruct']:
            __preprocessing = self.__idefics_preprocessing_fn
            __postprocessing = self.__idefics_postprocessing_fn
            __processor = IdeficsProcessor.from_pretrained(f"HuggingFaceM4/{self.model_name_or_path}")
            __model = IdeficsForVisionText2Text.from_pretrained(f"HuggingFaceM4/{self.model_name_or_path}", torch_dtype=self.dtype, device_map=self.device_map)
            __viz_config_name = 'idefics'
        elif self.model_name_or_path in ['kosmos-2-patch14-224']:
            __preprocessing = self.__kosmos_preprocessing_fn
            __postprocessing = self.__kosmos_postprocessing_fn
            __processor = Kosmos2Processor.from_pretrained(f"microsoft/{self.model_name_or_path}", padding_side='left')
            __model = Kosmos2ForConditionalGeneration.from_pretrained(f"microsoft/{self.model_name_or_path}", torch_dtype=self.dtype, device_map=self.device_map)
            __model_kwargs.update({'image_embeds': None, 'use_cache': True})
            __viz_config_name = 'kosmos'

        self.__preprocessing_fn = __preprocessing
        self.__postprocessing_fn = __postprocessing
        self.__processor = __processor
        self.__model = __model
        self.model_kwargs = __model_kwargs
        self.processor_kwargs = __processor_kwargs

        if self.viz:
            __viz_prompts = getattr(VizConfig, 'prompts')
            __viz_config = getattr(VizConfig, __viz_config_name)
            self.viz_prompts = __viz_prompts
            self.do_q_att = __viz_config['do_q_att']
            self.img_tok_slice = __viz_config['img_tok_slice']
            self.img_tok_size = __viz_config['img_tok_size']

            self.max_new_tokens = 1
            self.model_kwargs['max_new_tokens'] = 1
    
    def __select_prompt(self, prompt, task):
        if self.viz and self.viz_ptype != 'base':
            if self.viz_ptype == 'common':
                return self.viz_prompts['common']
            elif self.viz_ptype == 'task-general':
                return self.viz_prompts[task]
        else:
            return self.__revised_text(prompt)
    
    def __revised_text(self, text):
        last_word = text[-1]
        if last_word == '.' or last_word == '?':
            return text
        else:
            return text + '.'

    def __bytes2img(self, img_bytes):
        img_bytes = io.BytesIO(img_bytes)
        return Image.open(img_bytes)
    
    # ---------------------------------------------------------------------------------
    # MMM Utils
    # ---------------------------------------------------------------------------------
    def processor(self, **kwargs):
        return self.__processor(**kwargs, **self.processor_kwargs).to(self.__model.device, dtype=self.dtype)

    def tokenization(self, text, **text_kwargs):
        return self.__processor.tokenizer(text, **text_kwargs).encodings[0]
    
    def convert_ids_to_tokens(self, ids):
        return self.__processor.tokenizer.convert_ids_to_tokens(ids)
    
    def decode(self, ids):
        return self.__processor.decode(ids, skip_special_tokens=True)
    
    def batch_decode(self, ids):
        return self.__processor.batch_decode(ids, skip_special_tokens=True)
    
    def generate(self, **inputs):
        return self.__model.generate(**inputs, **self.model_kwargs)
    
    def prepare_input(self, data):
        return self.__preprocessing_fn(data)
    
    def prepare_output(self, data):
        return self.__postprocessing_fn(data)
    
    # ---------------------------------------------------------------------------------
    # Chameleon
    # ---------------------------------------------------------------------------------
    def __chameleon_preprocessing_fn(self, data):
        task, img_info, prompt, gt, md = data

        prompt = self.__select_prompt(prompt, task)
        input_prompt = "<image>" + prompt

        input_img = self.__bytes2img(img_info['bytes'])

        return (task, input_img, input_prompt, prompt, gt, md)
    
    def __chameleon_postprocessing_fn(self, data):
        prompt, gen_output = data
        return gen_output[len(prompt):]
    
    # ---------------------------------------------------------------------------------
    # Llama-3.2
    # ---------------------------------------------------------------------------------
    def __mllama_preprocessing_fn(self, data):
        task, img_info, prompt, gt, md = data

        prompt = self.__select_prompt(prompt, task)
        input_prompt = f"<|image|>{prompt}"

        input_img = self.__bytes2img(img_info['bytes'])

        return (task, input_img, input_prompt, prompt, gt, md)

    def __mllama_postprocessing_fn(self, data):
        prompt, gen_output = data
        return gen_output[len(prompt):]
    
    def __mllama_i_preprocessing_fn(self, data):
        task, img_info, prompt, gt, md = data

        prompt = self.__select_prompt(prompt, task)
        input_prompt = [
            [
                {"role": "user", 
                 "content": [
                     {"type": "image"},
                     {"type": "text", "text": prompt}
                     ]
                }
            ],
        ]
        input_prompt = self.__processor.apply_chat_template(input_prompt, add_generation_prompt=True)
        if self.batch_size > 1:
            input_prompt = input_prompt[0]

        input_img = self.__bytes2img(img_info['bytes'])

        return (task, input_img, input_prompt, prompt, gt, md)

    def __mllama_i_postprocessing_fn(self, data):
        _, gen_output = data
        return gen_output.split('assistant\n\n')[-1]

    # ---------------------------------------------------------------------------------
    # Llava
    # ---------------------------------------------------------------------------------
    def __llava_preprocessing_fn(self, data):
        task, img_info, prompt, gt, md = data

        prompt = self.__select_prompt(prompt, task)
        input_prompt = f"USER: <image>\n{prompt} ASSISTANT:"

        input_img = self.__bytes2img(img_info['bytes'])

        return (task, input_img, input_prompt, prompt, gt, md)

    def __llava_postprocessing_fn(self, data):
        _, gen_output = data
        return gen_output.split('ASSISTANT: ')[-1]
    
    # ---------------------------------------------------------------------------------
    # IDEFICS
    # ---------------------------------------------------------------------------------
    def __idefics_preprocessing_fn(self, data):
        task, img_info, prompt, gt, md = data

        input_img = self.__bytes2img(img_info['bytes'])

        prompt = self.__select_prompt(prompt, task)
        input_prompt = [
            "User:",
            input_img,
            f"{prompt}\nAssistant:",
        ]

        if self.batch_size > 1:
            input_prompt = [input_prompt]

        return (task, input_img, input_prompt, prompt, gt, md)

    def __idefics_postprocessing_fn(self, data):
        _, gen_output = data
        return gen_output.split('Assistant:')[-1]
    
    # ---------------------------------------------------------------------------------
    # KOSMOS-2
    # ---------------------------------------------------------------------------------
    def __kosmos_preprocessing_fn(self, data):
        task, img_info, prompt, gt, md = data

        prompt = self.__select_prompt(prompt, task)
        input_prompt = f"<grounding> {prompt}"

        input_img = self.__bytes2img(img_info['bytes'])

        return (task, input_img, input_prompt, prompt, gt, md)

    def __kosmos_postprocessing_fn(self, data):
        prompt, gen_output = data
        caption, entities = self.processor.post_process_generation(gen_output)
        return caption[len(prompt):]