from quality_metrics.llm_judge.test.utils_test import load_archive
from quality_metrics.llm_judge.judge_hf import (
    Auto_j_Rank,
    )
archive=load_archive(n_puzzle=5)

# HF version of autoj model (can also use unquantized model)
path_model = "/home/flowers/work/hf/autoj-13b-GPTQ-4bits"
prompt_instruction="Please rank the following puzzles according to their quality. If two puzzles are of equal quality, please assign the same rank to both. If you are unsure, please assign the same rank to both puzzles. "
# pairwise
autoj = Auto_j_Rank(archive, prompt_instruction=prompt_instruction, model_id=path_model, mode_rank="pairwise",bs=2,exllama2=True)
ranked_keys, grades = autoj.absolute_ranking()
ranked_puzzles, win_record = autoj.computing_ranking()
print("Pairwise mode (win record +1 won, +0.5 tie, +0 lost): ")
print(win_record)

# aboslute
prompt_instruction="Please grade the following puzzles according to its quality."
autoj.prompt_instruction = prompt_instruction
autoj.mode_rank = "absolute"
# autoj.puzzle_dict = puzzle_dict
ranked_keys, grades = autoj.computing_ranking()
sorted_grades=sorted(grades, key=grades.get, reverse=True)
print("Absolute mode (grades): ")
print(grades)