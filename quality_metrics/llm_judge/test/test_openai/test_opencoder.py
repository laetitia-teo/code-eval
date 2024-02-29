from quality_metrics.llm_judge.test.utils_test import load_archive
from quality_metrics.llm_judge.judge_openai import (
    OpenCodeInterpreter,
    )
archive=load_archive()
from quality_metrics.key import cfg_client_azure


opencodeinterp = OpenCodeInterpreter(puzzle_dict=archive, Opencode_mode="1",cfg_openai_client=cfg_client_azure,temperature=0)
ranked_keys, grades = opencodeinterp.absolute_ranking()
sorted_grades=sorted(grades, key=grades.get, reverse=True)
print(grades)