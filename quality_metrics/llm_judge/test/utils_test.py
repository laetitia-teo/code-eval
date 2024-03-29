import json
import numpy as np
import os
test_directory = os.path.dirname(os.path.realpath(__file__))
# path_prompt = os.path.abspath("src/examplefile.txt") which one is "better"?
path_prompt = os.path.join(test_directory,'puzzles_test.json')

def load_archive(n_puzzle=30, path_archive=path_prompt):
    with open(path_archive, "r") as file:
        data = json.load(file)
    np.random.seed(0)
    if n_puzzle == -1:
        n_puzzle = len(data)
    idx = np.random.choice(len(data), n_puzzle, replace=False)
    puzzle_aces = {f"puzz_{i}": format_puzzle(data[i]) for i in idx}
    return puzzle_aces

def format_puzzle(data_point):
    puzzle = data_point["program_str"]
    description = data_point["description"]
    for _ in range(3): 
        if isinstance(description, list):
            description = description[0]
        if isinstance(description, str):
            break
    format = description +"\n```python\n"+puzzle+"\n```"
    return format


def save_quality(path_archive,grades,name_quality="yes_finetuning_deepseek-1.3b"):
    with open(path_archive, 'r') as f:
        puzzle_dict_raw = json.load(f)
    len(set(grades.keys()))==len(puzzle_dict_raw)
    for key,values in grades.items():
        idx=int(key.split("_")[1])
        values_mean=float(np.mean(values))
        puzzle_dict_raw[idx][name_quality]=values_mean
    with open(path_archive, 'w') as f:
        json.dump(puzzle_dict_raw, f, indent=4)