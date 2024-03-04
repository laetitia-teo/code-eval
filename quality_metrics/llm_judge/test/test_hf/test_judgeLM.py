from quality_metrics.llm_judge.test.utils_test import load_archive
from quality_metrics.llm_judge.judge_hf import (
    JudgeLM,
    )
from quality_metrics.llm_judge.prompt_judge import prompt_instruction_finetuning
archive=load_archive(n_puzzle=5)

# HF version of autoj model (can also use unquantized model)
path_model = "/home/flowers/work/hf/JudgeLM-7B-v1.0"
prompt_instruction = prompt_instruction_finetuning
# pairwise
judgeLM = JudgeLM(archive, prompt_instruction=prompt_instruction,fast_eval=True, model_id=path_model, mode_rank="pairwise",bs=1,exllama2=False)
ranked_keys, grades = judgeLM.absolute_ranking()
ranked_puzzles, win_record = judgeLM.computing_ranking()
print("Pairwise mode (win record +1 won, +0.5 tie, +0 lost): ")
print(win_record)

# aboslute
judgeLM.prompt_instruction = prompt_instruction
judgeLM.mode_rank = "absolute"
# autoj.puzzle_dict = puzzle_dict
ranked_keys, grades = judgeLM.computing_ranking()
sorted_grades=sorted(grades, key=grades.get, reverse=True)
print("Absolute mode (grades): ")
print(grades)