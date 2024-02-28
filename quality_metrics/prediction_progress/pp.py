import re
import torch

from typing import Union, Dict, List, Optional

from quality_metrics.common import (
    QualityMetric,
    Problem,
    create_model_and_tokenizer,
)

from quality_metrics.prompts.prompts import pp_prompt_user_1ex, pp_prompt_assistant
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
            solution_exclude_pattern: Optional[str] = 'def g\(.*\).*:'
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
        self.batch_size = batch_size
        self.solution_exclude_pattern = solution_exclude_pattern
    
        # load archive
        if isinstance(archive_path_or_list, str):
            self.problem_archive = Problem.load_dataset(archive_path_or_list)
        else:
            self.problem_archive = archive_path_or_list

        self.reference_problem = reference_problem
        self.user_prompt = pp_prompt_user_1ex
        self.assistant_prompt = pp_prompt_assistant

        # check what wappens with non-chat models here
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful chatbot who solves python programming problems.",
            },
        ]

        # TODO find a way to mask some tokens of the solution, maybe with a pattern
        self.solution_mask = solution_mask
        self._filter_archive_problems()
        self.original_losses = self._get_original_losses()
    
    def _get_original_losses(self):
        return self._get_losses(self.reference_problem)

    def _filter_archive_problems(self):
        # TODO use the model's context size and the prompt to compute which problems 
        #  are too long
        pass

    def _is_problem_too_long(self, problem: Problem):
        return False

    def _get_losses(self, problem: Problem):
        # format prompts with archive and ref puzzles
        completed_user_prompts = [
            self.user_prompt.format(
                puzzle=problem.instruction,
                solution=problem.completion,
                archive_puzzle=archive_problem.instruction)
                # archive_solution=archive_problem.completion)
            for archive_problem in self.problem_archive
        ]

        completed_assistant_prompts = [
            self.assistant_prompt.format(
                archive_solution=archive_problem.completion)
            for archive_problem in self.problem_archive
        ]

        # apply chat template to all the prompts
        prompts = []
        for user_prompt, assistant_prompt in zip(completed_user_prompts, completed_assistant_prompts):
            # TODO wrong use of the chat template, fix this
            messages = self.messages + [
                {'role': 'user', 'content': user_prompt},
                {'role': 'assistant', 'content': assistant_prompt}
            ]
            prompts.append(self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            ))
        archive_tokenized_puzzles = self.tokenizer(prompts, return_tensors='pt',
                                                   padding=True)

        # get solution mask
        if self.solution_mask:
            before_solutions = []
            solutions_to_predict = []
            for archive_problem, prompt in zip(self.problem_archive, prompts):
                if self.solution_exclude_pattern is None:
                    # simply mask all tokens that are not the solution
                    before_solutions.append(''.join(prompt.split(archive_problem.completion)[:-1]))
                    solutions_to_predict.append(archive_problem.completion)
                else:
                    matched_exclude = re.findall(
                        self.solution_exclude_pattern, archive_problem.completion)[0]
                    completion = archive_problem.completion.replace(matched_exclude, '')
                    solutions_to_predict.append(completion)
                    before_solutions.append(''.join(prompt.split(completion)[:-1]))
            
            num_tokens_before = [len(t) for t in self.tokenizer(before_solutions).input_ids]
            num_tokens_solution = [len(s) for s in solutions_to_predict]
            masks = torch.zeros_like(archive_tokenized_puzzles.attention_mask)
            offsets = [l.tolist().index(1) for l in archive_tokenized_puzzles.attention_mask]
            for i, (t, num_solution_tokens, o) in enumerate(
                    zip(num_tokens_before, num_tokens_solution, offsets)):
                masks[i, o+t:o+t+num_solution_tokens] = 1.

            archive_tokenized_puzzles.loss_attention_mask = masks

        return losses.get_solution_losses(archive_tokenized_puzzles, self.model,
                                          batch_size=self.batch_size)

    def differences(self, problem: Problem):
        final_losses = self._get_losses(problem)
        return (self.original_losses - final_losses)

    def __call__(self, problem: Problem):
        return self.differences(problem).mean().item()


