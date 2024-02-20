import os
import json
import pathlib
from tqdm import tqdm

import torch
import numpy as np

# TODO add llamatokenizer where needed
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, LlamaTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

from utils_test import (
    prompt_train,
    Prompt_Intstruction,
)

def train(config, run_id):
    
    print("\n=============\n ")
    print("Path of training data ", config.train_path)
    print("\n=============\n ")
    print("Save path ", f'outputs/{run_id}')
    print("\n=============\n ")

    if config.slurm.cluster == 'jz':
        os.environ["WANDB_DISABLED"] = "True"
        os.environ['TRANSFORMERS_OFFLINE'] = "1"
        os.environ['WANDB_MODE'] = "offline"
    
    os.environ["WANDB_PROJECT"] = 'measure_pp'
    os.environ['WANDB_CACHE_DIR'] = 'wandb_cache'
    os.environ['TOKENIZERS_PARALLELISM'] = "True"

    if config.slurm.gpu == "v100":
        bf16 = False
        fp16 = True
    else:
        bf16 = True
        fp16 = False

    run_name_wandb = f'{config.model_id}_{run_id}'
    model_save_dir = f'outputs/{run_id}/save_models'
    name_json_save_all = f'outputs/{run_id}/save_results/passk.json'
    model_name = config.model_id.split('/')[-1]
    print(f'run_name_wandb {run_name_wandb}')

    print(os.getcwd())
    # hf way to load json dataset
    with open(config.train_path, encoding="utf-8") as f:
        dataset = json.load(f)
    to_remove = [
        "emb",
        "target_skills",
        "puzzle_history",
        "quality",
        "description",
        "is_valid",
        "is_valid_explanation"
    ]
    for i in dataset:
        for j in to_remove:
            if j in i:
                del i[j]

    dataset = Dataset.from_list(dataset)
    dataset = dataset.shuffle(seed=config.seed)

    save_all_dir = str(pathlib.Path(name_json_save_all).parent)
    if not os.path.exists(save_all_dir):
        os.makedirs(save_all_dir)

    if not os.path.exists(name_json_save_all):
        # Create a new JSON file with some sample data
        sample_data = {}
        with open(name_json_save_all, 'w') as file:
            json.dump(sample_data, file, indent=4)

    if not os.path.exists('save_results'):
        os.makedirs('save_results')
    if not os.path.exists('save_sol'):
        os.makedirs('save_sol')

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map="auto",
    )
    
    def formatting_prompts_func(example):
        output_texts = []
        # print(len(example['program_str']))
        for i in range(len(example['program_str'])):

            puzzle= example['program_str'][i]
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

    lr_scheduler_type= "cosine"

    warmup_ratio=0.1

    if config.sol==True:
        response_template= "Solution 1:"
    else:
        response_template = "Problem 1:"

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer,mlm=False)

    learning_rate=1e-5

    training_arguments=TrainingArguments(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.accum_step,
        run_name=run_name_wandb,
        save_strategy="no",
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        num_train_epochs=config.num_epochs,
        learning_rate=learning_rate,
        bf16=bf16, 
        fp16=fp16,
        gradient_checkpointing=False,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_torch",
        max_grad_norm=0.3,
    )

    # TODO add validation dataset maybe
    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=1024,
        args=training_arguments
    )
    trainer.train()

    output_dir = os.path.join(model_save_dir, f'{model_name}_{run_id}')
    trainer.save_model(output_dir)
    del model
    del tokenizer

    return trainer
