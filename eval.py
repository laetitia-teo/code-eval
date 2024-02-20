import os
import json
import copy
from tqdm import tqdm

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM


from utils_test import (
    return_full_prompt,
    prompt_train,
    pass_at_k,
    Prompt_Intstruction,
    judge_parallel,
)


def eval(config, run_id, trainer=None):  # TODO how to get the same run_id than the train if done separately

    if config.slurm.gpu == "v100":
        type_use = torch.float16
    else:
        type_use = torch.bfloat16

    model_save_dir = f'outputs/{run_id}/save_models'
    name_json_save_all = f'outputs/{run_id}/save_results/passk.json'
    model_name = config.model_id.split('/')[-1]
    name_json_sol = f'outputs/{run_id}/save_sol/sols.json'
    output_dir = os.path.join(model_save_dir, f'{model_name}_{run_id}')

    if trainer is not None:
        trainer.accelerator.clear()

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            obj.cpu()            
    gc.collect()
    torch.cuda.empty_cache()

    # testing
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=type_use,

        # quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True
    )
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True
    if config.compile:
        model = torch.compile(model)

    with open(config.test_path) as f:
        testset = json.load(f)
    curr_idx = 0
    correct_puzz = 0

    num_return_sequences = config.k #n_try
    list_all_passk=[[] for i in range(num_return_sequences)]
    list_passk=[]

    list_puzzle=[]
    list_all_puzzle=[]

        
    list_testset= [x["program_str"] for x in testset]
    list_puzzle_correct=[]

    with torch.inference_mode():
        
        for idx in tqdm(range(curr_idx, len(list_testset), config.eval_batch_size)):
            print(f"\n\n============ idx {idx} ==================\n")
            
            list_prompt = []
            list_prompt_f = []
            subset_test = list_testset[idx:idx+config.eval_batch_size]
            
            for idx_puz in range(len(subset_test)):
                prompt_f = subset_test[idx_puz].split("def g(")[0]
                list_prompt_f.append(prompt_f)
                prompt = return_full_prompt(model_id=model_name, pb=prompt_f)
                list_prompt.append(prompt)

            inputs = tokenizer(list_prompt, return_tensors="pt", padding=True).to("cuda")

            len_prompt = inputs["input_ids"].shape[1]
            list_puzzle_gen=[[] for _ in range(len(list_prompt))]

            for idx_gen in range(num_return_sequences):
                outputs = model.generate(**inputs,max_new_tokens=512,do_sample=True, temperature=0.7)
                generated_texts = tokenizer.batch_decode(outputs[:,len_prompt:], skip_special_tokens=True)
                for idx_out_gen in range(len(outputs)):
                    list_puzzle_gen[idx_out_gen].append(generated_texts[idx_out_gen])

            list_generated_text = copy.deepcopy(list_puzzle_gen)

            for i in range(len(list_puzzle_gen)): # along the bs
                dic_save = {}
                list_raw_puzzle = []
                list_proc_puzzle = []
                for j in range(len(list_puzzle_gen[i])):
                    prompt_f = list_prompt_f[i]
                    try:
                        list_puzzle_gen[i][j] = list_puzzle_gen[i][j].replace("```python","```")
                        list_puzzle_gen[i][j] = list_puzzle_gen[i][j].replace("```Python","```")

                        if "```" in list_puzzle_gen[i][j]:
                            extract_g=list_puzzle_gen[i][j].split("```")[1].split("assert")[0]
                        else:
                            if "assert" in list_puzzle_gen[i][j]:
                                extract_g=list_puzzle_gen[i][j].split("assert")[0]
                            else:    
                                extract_g=list_puzzle_gen[i][j]

                    except:
                        print("error extract g")
                        print(list_puzzle_gen[i][j])

                    extract_g = extract_g + "\nassert f(g()) == True\n"
                    test_fg= prompt_f + extract_g 
                    list_puzzle_gen[i][j] = test_fg
                    list_puzzle.append(test_fg)
                    list_proc_puzzle.append(test_fg)
                    list_raw_puzzle.append(prompt_f + list_puzzle_gen[i][j])

                dic_save["raw_puzzle"] = list_raw_puzzle
                dic_save["process_puzzle"] = list_proc_puzzle
                
                list_valid_puzzles = judge_parallel(list_puzzle_gen[i])
                dic_save["list_valid"] = list_valid_puzzles                 
                list_all_puzzle.append(dic_save)    

                cor_puz= np.sum(list_valid_puzzles)

                n_sample, n_correct = num_return_sequences,cor_puz
                pass_k = pass_at_k(n_sample, n_correct, k=num_return_sequences)
                list_passk.append(pass_k)

                for idx_passk in range(num_return_sequences):
                    pass2add = pass_at_k(n_sample, n_correct, k=idx_passk+1)
                    list_all_passk[idx_passk].append(pass2add)
                    testset[idx + i][f'pass_{idx_passk+1}'] = pass2add

                proba_solved = n_correct / n_sample
                testset[idx + i]['proba_solved'] = float(proba_solved)
                testset[idx + i]['n_sample'] = int(n_sample)
                testset[idx + i]['n_correct'] = int(n_correct)
                testset[idx + i]['generated_text'] = list_generated_text[i]
                testset[idx + i]['parsed_puzzles'] = list_puzzle_gen[i]
                
            print(f"correct puzzles: {int(np.sum(list_passk))}/{len(list_passk)}")

        for idx_passk in range(num_return_sequences):
            print(f"pass {idx_passk+1}: {np.sum(list_all_passk[idx_passk])}/{len(list_all_passk[idx_passk])}")
        dic_passk={}
        for idx_passk in range(num_return_sequences):
            dic_passk[f"pass_{idx_passk+1}"]=float(np.sum(list_all_passk[idx_passk]))

        with open(name_json_save_all, "r") as outfile:
            json_content = json.load(outfile)
        json_content[run_id] = dic_passk 
        with open(name_json_save_all, "w") as outfile:
            json.dump(json_content, outfile, indent=4)

        with open(name_json_sol, "w") as outfile:
            json.dump(testset, outfile, indent=4)
