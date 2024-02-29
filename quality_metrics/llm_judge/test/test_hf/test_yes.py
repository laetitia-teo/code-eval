from quality_metrics.llm_judge.test.utils_test import load_archive
from quality_metrics.llm_judge.judge_hf import (
    Yes_model,
    )
archive=load_archive()

# HF version of yes model
path_model ="/home/flowers/work/hf/deepseek-coder-1.3b-instruct"

yes = Yes_model(puzzle_dict=archive,model_id=path_model,yes_mode="education",exllama2=False)
ranked_keys, grades = yes.absolute_ranking()
sorted_grades=sorted(grades, key=grades.get, reverse=True)
print(grades)
