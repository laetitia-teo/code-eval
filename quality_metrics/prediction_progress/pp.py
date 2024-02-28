import os
import torch

from typing import Union, Dict, List, Optional

from quality_metrics.common import (
    QualityMetric,
    Problem,
    create_model_and_tokenizer,
)

from quality_metrics.prompts.prompts import pp_prompt
from quality_metrics.prediction_progress import losses


class PredictionProgressCE(QualityMetric):
    def __init__(
            self,
            model_id_or_path: str,
            archive_path_or_list: Union[str, List[Problem]],
            reference_problem: Problem,
            prompt: Optional[str] = None,
            solution_mask: bool = False,
            batch_size: int = 1,
        ):
        """
        In-context Prediction Progress using CrossEntropy.
        We define it here as loss(ref problem) - loss(problem), so that higher is better.
        """
        # load model
        # create model and tokenizer
        model, tokenizer = create_model_and_tokenizer(model_id_or_path, compile=False)
        self.model = model
        self.tokenizer = tokenizer
    
        # load archive
        if isinstance(archive_path_or_list, str):
            self.problem_archive = Problem.load_dataset(archive_path_or_list)
        else:
            self.problem_archive = archive_path_or_list

        self.reference_problem = reference_problem
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = pp_prompt

        # check what wappens with non-chat models here
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful chatbot who solves python programming problems.",
            },
        ]

        # TODO find a way to mask some tokens of the solution, maybe with a pattern
        self.solution_mask = solution_mask
        self._filter_problems()
        self.original_losses = self._get_original_losses()
    
    def _filter_problems(self):
        # TODO use the model's context size and the prompt to compute which problems 
        #  are too long
        pass

    def _get_losses(self, problem: Problem):
        # format prompts with archive and ref puzzles
        archive_puzzle_sols = [
            self.prompt_text.format(
                puzzle=problem.instruction,
                solution=problem.completion,
                archive_puzzle=archive_problem.instruction,
                archive_solution=archive_problem.completion)
            for archive_problem in self.problem_archive]

        # apply chat template to all the prompts
        prompts = []
        for archive_puz_sol in archive_puzzle_sols:
            messages = self.messages + [{'user': archive_puz_sol}]
            prompts.append(self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            ))
        archive_tokenized_puzzles = self.tokenizer(prompts, return_tensors='pt',
                                                   padding=True)

        # get solution mask
        if self.solution_mask:
            raise NotImplementedError

        return losses.get_solution_losses(archive_tokenized_puzzles, self.model,
                                          batch_size=self.batch_size)

    def __call__(self, problem: Problem):
        final_losses = self._get_losses(problem)
        return (self.original_losses - final_losses).mean().item()
        

