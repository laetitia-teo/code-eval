from quality_metrics.llm_judge.test.utils_test import load_archive
from quality_metrics.llm_judge.judge_openai import (
    Yes_model,
    )
archive=load_archive()
from quality_metrics.key import cfg_client_azure


yes = Yes_model(puzzle_dict=archive, yes_mode="finetuning",cfg_openai_client=cfg_client_azure,temperature=0)
ranked_keys, grades = yes.absolute_ranking()
sorted_grades=sorted(grades, key=grades.get, reverse=True)
print(grades)