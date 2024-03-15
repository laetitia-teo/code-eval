import os
import pathlib

import torch
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


def save_dataset(dataset: List[Problem], path: str):
    json_dataset = []
    for p in dataset:
        el = {}
        el['instruction'] = p.instruction
        el['completion'] = p.completion
        el['idx'] = p.idx
        el['quality'] = p.quality
        el['description'] = p.description
        el['origin'] = p.origin
        json_dataset.append(el)

    if not os.path.exists(str(pathlib.Path(path).parent)):
        os.makedirs(str(pathlib.Path(path).parent))

    with open(path, 'w') as f:
        json.dump(json_dataset, f)


def load_dataset(path: str):
    dataset = []
    with open(path, 'r') as f:
        json_dataset = json.load(f)
    for p in json_dataset:
        dataset.append(Problem(**p))
    return dataset


### quality

class QualityMetric(ABC):
    @abstractmethod
    def __call__(self, problem: Problem):
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