import torch
from transformers import (
    CodeLlamaTokenizer,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from typing import Optional, List, Any, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from quality_metrics.utils.p3 import get_puzzle_sol


### problem utils

@dataclass
class Problem:
    idx: str
    instruction: str
    completion: str
    quality: Optional[float] = 0.
    description: Optional[str] = ''
    origin: Optional[str] = ''

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
    def from_p3(self, dic):
        # handle both the program_str case and the original p3 format
        puzzle, solution = get_puzzle_sol(dic)
        if 'name' in dic:
            idx = dic['name']
        else:
            raise RuntimeError("No name found in puzzle")  # find a fallback when this happens

        p = Problem(
            idx=idx,
            instruction=puzzle,
            completion=solution
        )
        return p
    
    def get_token_counts(self, tokenizer):
        return len(tokenizer(self.instruction).input_ids) + len(tokenizer(self.completion).input_ids)


def dataset_from_p3(dataset):
    problem_dataset = []
    for p in dataset:
        problem_dataset.append(Problem.from_p3(p))
    return problem_dataset


### quality

class QualityMetric(ABC):
    @abstractmethod
    def __call__(self, problem: Problem):
        raise NotImplementedError


### model utils

def create_model_and_tokenizer(model_id, compile=True, dtype=torch.bfloat16, flash_attn=True):
    if 'codellama' in model_id:
        tokenizer = CodeLlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
    elif 'llama' in model_id:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # todo: simplify
    if flash_attn:
        try:
            import flash_attn
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                # quantization_config=quantization_config,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                # quantization_config=quantization_config,
                device_map="auto",
                # local_files_only=True,
                trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            # quantization_config=quantization_config,
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
