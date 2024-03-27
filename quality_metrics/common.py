import os
import re
import pathlib

import torch
import random
import json
from transformers import (
    CodeLlamaTokenizer,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)

import numpy as np
from openai import OpenAI
from typing import Optional, List, Any, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from datasets import Dataset

from utils_test import prompt_train, return_prompt_format  # TODO move these someplace else?
from quality_metrics.utils.p3 import get_puzzle_sol


### problem utils

@dataclass
class Problem:
    idx: str
    instruction: str
    completion: str
    quality: Optional[Dict[str, Any]] = field(default_factory=dict)
    description: Optional[str] = ''
    origin: Optional[str] = 'unknown'

    @classmethod
    def from_dict(self, dic):
        p = Problem(
            idx=dic['idx'],
            instruction=dic['instruction'],
            completion=dic['completion']
        )
        if 'quality' in dic:
            p.quality = dic['quality']
        if 'description' in dic:
            p.description = dic['description']
        if 'origin' in dic:
            p.origin = dic['origin']
        
        return p
    
    @classmethod
    def from_p3(self, dic, origin=None):
        # handle both the program_str case and the original p3 format
        puzzle, solution = get_puzzle_sol(dic)
        if 'name' in dic:
            idx = dic['name']
        else:
            idx = f'random_{np.random.choice(1000000)}'
        
        if origin is not None:
            p_origin = origin
        elif 'origin' in dic:
            p_origin = dic['origin']
        else:
            p_origin = 'unknown'
        
        if 'description' in dic:
            p_description = dic['description']
        else:
            p_description = ''

        p = Problem(
            idx=idx,
            instruction=puzzle,
            completion=solution,
            description=p_description,
            origin=p_origin,
        )
        return p
    
    def get_token_counts(self, tokenizer):
        return len(tokenizer(self.instruction).input_ids) + len(tokenizer(self.completion).input_ids)
    
    def get_problem(self):
        """return the problem as a string"""
        return "```python+\n"+self.instruction + "\n" + self.completion + "\n```"


def dataset_from_p3(dataset: List) -> List[Problem]:
    problem_dataset = []
    pb_idx = 0
    for p in dataset:
        if 'sol_bodies' in p and not p['sol_bodies']:  # p3 probem without solutions
            continue
        if 'name' in p:
            problem_dataset.append(Problem.from_p3(p))
        else:
            given_name = f'unnamed_puzzle_{pb_idx}'
            p['name'] = given_name
            problem_dataset.append(Problem.from_p3(p))
            pb_idx += 1

    return problem_dataset


def dict_from_dataset(dataset: List[Problem]) -> List[Dict]:
    json_dataset = []
    for p in dataset:
        el = {}
        el['instruction'] = p.instruction
        el['completion'] = p.completion
        el['idx'] = p.idx
        # print(p.quality)
        el['quality'] = p.quality
        el['description'] = p.description
        el['origin'] = p.origin
        json_dataset.append(el)
    
    return json_dataset


def save_dataset(dataset: List[Problem], path: str):
    json_dataset = dict_from_dataset(dataset)

    if not os.path.exists(str(pathlib.Path(path).parent)):
        os.makedirs(str(pathlib.Path(path).parent))

    with open(path, 'w') as f:
        json.dump(json_dataset, f)


def load_dataset(path: str):
    dataset = []
    print(os.getcwd())
    with open(path, 'r') as f:
        json_dataset = json.load(f)
    for p in json_dataset:
        dataset.append(Problem(**p))
    return dataset


def p3_insert_description(description, instruction):
    # think this works in all cases
    return '\n'.join([instruction.split('\n')[0], description] + instruction.split('\n')[1:])


def process_description(description):
    if isinstance(description, str):
        # process string so it matches the expected format
        pattern = '    """.+"""'
        if not re.match(pattern, description, re.DOTALL):
            description = f'    """{description}"""'
        return description
    else:
        return process_description(description[0])


def formatting_prompts_func(example):
    output_texts = []
    # print(len(example['program_str']))
    for i in range(len(example['program_str'])):

        puzzle = example['program_str'][i]
        try:
            prompt_f=puzzle.split("def g(")[0]
            prompt_g= "def g(" + puzzle.split("def g(")[1]
            full_prompt = prompt_train.format(pb=prompt_f, g=prompt_g)
            output_texts.append(full_prompt)
        except:
            print("error in formatting_prompts_func idx",i)
            print(example['program_str'][i])
            print("======================")
            print(puzzle)
    return output_texts


def get_hf_dataset(dataset, quality_key=None, seed=0, use_description=True):
    def process_fn(el):
        if quality_key is not None:  # in this case we raise an error if we don't have quality
            try:
                el['quality'] = np.mean(el['quality'][quality_key])
            except Exception as e:
                print(el)
                raise e
            
        instruction = p3_insert_description(process_description(el['description']), el['instruction'])
        el['program_str'] = instruction + "\n\n" + el['completion'] + "\n\nassert f(g()) is True"
        
        keys_to_rm = []
        for key, val in el.items():
            if isinstance(val, list):
                keys_to_rm.append(key)
        for key in keys_to_rm:
            del el[key]

        return el

    processed_ds = []
    if quality_key is not None:
        for el in dataset:
            if 'quality' in el:
                if quality_key in el['quality']:
                    processed_ds.append(process_fn(el))
    else:
        processed_ds = [process_fn(el) for el in dataset]

    dataset = Dataset.from_list(processed_ds)
    dataset = dataset.shuffle(seed=seed)
    return dataset


def get_tokenized_hf_dataset(dataset, tokenizer, use_description=True, max_legth=2048, labels=False,
                             use_chat_format=True):
    if use_chat_format:
        dummy_chat = [{'role': 'user', 'content': '{instruction}'}]
        complete_prompt = tokenizer.apply_chat_template(dummy_chat, tokenize=False, add_generation_prompt=True)
    else:
        complete_prompt = '{instruction}'

    def process_fn(el):
        instruction = p3_insert_description(process_description(el['description']), el['instruction'])
        puzzle = instruction + "\n\n" + el['completion'] + "\n\nassert f(g()) is True"
        prompt_f = puzzle.split("def g(")[0]
        prompt_g = "def g(" + puzzle.split("def g(")[1]
        instruction = prompt_train.format(pb=prompt_f, g=prompt_g)
        return complete_prompt.format(instruction=instruction)

    texts = [process_fn(el) for el in dataset]
    tokenizer_outs = tokenizer(texts, max_length=max_legth)
    dataset = [{'input_ids': tokenizer_outs.input_ids[i], 'text': texts[i]} for i in range(len(texts))]
    if labels:
        for el in dataset:
            el['label'] = el['input_ids']
    dataset = Dataset.from_list(dataset)
    return dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


### quality

class QualityMetric(ABC):
    @abstractmethod
    def __call__(self, problem: Problem, return_list=True):
        raise NotImplementedError


def create_model_and_tokenizer(model_id, compile=True, dtype=torch.bfloat16, flash_attn=True,exllama2=False):
    if 'codellama' in model_id:
        tokenizer = CodeLlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
    elif 'llama' in model_id:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    gptq_config = None
    if exllama2:
        from transformers import GPTQConfig
        gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
    # todo: simplify
    if flash_attn:
        try:
            import flash_attn
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                quantization_config=gptq_config,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                quantization_config=gptq_config,
                device_map="auto",
                # local_files_only=True,
                trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            quantization_config=gptq_config,
            device_map="auto",
            # local_files_only=True,
            trust_remote_code=True
        )
    # model.cuda()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True
    if compile:
        model = torch.compile(model)

    return model, tokenizer


# OpenAI inference

from concurrent.futures import ThreadPoolExecutor
def get_completion(client, prompt :str, cfg_generation :dict, system_prompt :str = None, temperature=None)->str:
    """Get completion from OpenAI API
    cfg_generation: kwarg of client.chat.completions.create (model,temperature, max_tokens, top_p, logbprobs,...)
    """
    kwargs={}
    kwargs.update(cfg_generation)
    if temperature is not None:
        kwargs["temperature"]= temperature
    if system_prompt == None:
        sys_message = "You are a helpful assistant."
    else:
        sys_message = system_prompt
    try :
        completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": sys_message},
            {"role": "user", "content": prompt}
        ],**kwargs
        )
    except Exception as e:
        print("completion problem: ",e)
        return None 
    if "logprobs" in cfg_generation and cfg_generation["logprobs"]:
        out = completion.choices[0]
    else:
        out = completion.choices[0].message.content
    return out


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_multiple_completions(client, batch_prompt: list[str], cfg_generation: dict, max_workers=10, temperature=None) -> list[str]:
    """
    Get batch completions from OpenAI API
    #TODO:  need to integrate batch tools in the loop 
    """
    # check that batch_prompt is list[str]
    if isinstance(batch_prompt, str):
        batch_prompt = [batch_prompt]
    completions = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sub_batch in chunks(batch_prompt, max_workers):
            for _, message_list in enumerate(sub_batch):
                kwargs = {"client": client, "prompt": message_list}
                kwargs["cfg_generation"] = cfg_generation
                if temperature is not None:
                    kwargs["temperature"] = temperature
                future = executor.submit(
                    get_completion, **kwargs
                )
                completions.append(future)
    # Retrieve the results from the futures
    results = [future.result() for future in completions]
    return results


if __name__ == '__main__':
    # try to load dataset
    data_path = 'data/dataset.json'
    ds = json.load(open(data_path, 'r'))
    pb_ds = dataset_from_p3(ds)
    save_dataset(pb_ds, 'data/saved.json')
    print('done')


def add_padding_to_tokenizer(tokenizer):
    """ add the padding tokens in the tokenizer """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    