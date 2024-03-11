import re
import json
import torch

from typing import Union, Dict, List, Optional

from quality_metrics.common import (
    QualityMetric,
    Problem,
    create_model_and_tokenizer,
)

from quality_metrics.prompts.prompts import pp_prompt_user_1ex, pp_prompt_assistant
from quality_metrics.prediction_progress import losses
from quality_metrics.common import dataset_from_p3


class PredictionProgressCE(QualityMetric):
    def __init__(
            self,
            model_id_or_path: str,
            archive_path_or_list: Union[str, List[Problem]],
            reference_problem: Optional[Problem] = None,
            prompt: Optional[str] = None,
            solution_mask: bool = False,
            batch_size: int = 1,
            solution_exclude_pattern: Optional[str] = 'def g\(.*\).*:',
            use_doc: bool = False,  # TODO update this
            max_len: Optional[int] = None,  # if not set, use the model's max position embeddings
            **kwargs,
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

        if max_len is None:
            self.max_len = self.model.config.max_position_embeddings
        else:
            self.max_len = max_len
    
        # load archive
        if isinstance(archive_path_or_list, str):
            dataset = json.load(open(archive_path_or_list, 'r'))
            self.problem_archive = dataset_from_p3(dataset)
        else:
            self.problem_archive = archive_path_or_list

        if reference_problem is not None:
            self.reference_problem = reference_problem
        else:
            from quality_metrics.utils.p3 import REF_PUZZLE_NODOC, REF_SOL
            self.reference_problem = Problem(idx='reference', instruction=REF_PUZZLE_NODOC, 
                                             completion=REF_SOL)
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
        print('Filtering problem archive')
        self._filter_archive_problems()
        print('done')
        print('Computing original losses')
        self.original_losses = self._get_original_losses()
        print('done')
    
    def _get_original_losses(self):
        return self._get_losses(self.reference_problem)

    def _filter_archive_problems(self):
        prompts = self._get_prompts(self.reference_problem)
        new_archive_ids = []
        for idx, prompt in enumerate(prompts):
            if len(self.tokenizer(prompt).input_ids) < self.max_len:
                new_archive_ids.append(idx)
        self.problem_archive = [self.problem_archive[i] for i in new_archive_ids]

    def _is_problem_too_long(self, problem: Problem):
        return False

    def _get_prompts(self, problem: Problem):
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
        return prompts

    def _get_losses(self, problem: Problem):
        # format prompts with archive and ref puzzles
        prompts = self._get_prompts(problem)
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

    def differences(self, problem: Problem, return_list=False):
        final_losses = self._get_losses(problem)
        diff = (self.original_losses - final_losses)
        if return_list:
            diff = diff.tolist()
        return diff

    def __call__(self, problem: Problem):
        return self.differences(problem).mean().item()


class PredictionProgressCEDiff(PredictionProgressCE):
    """
    Version where we get the differences in losses instead of the average over reference puzzles.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, problem: Problem, return_list=True):
        return super().differences(problem, return_list)


class NormalizedPredictionProgressCE(PredictionProgressCE):
    """
    Normalize the PP by the original loss of the solution (to get a percentage of improvement).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eps = 1e-8
    
    def differences(self, problem: Problem, return_list=False):
        final_losses = self._get_losses(problem)
        diff = (self.original_losses - final_losses)
        diff = diff / (final_losses + 1e-8)
        if return_list:
            diff = diff.tolist()
        return diff
    

class NormalizedPredictionProgressCEDiff(NormalizedPredictionProgressCE):
    """
    Normalized x Diff: get normalized PP differences.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, problem: Problem, return_list=True):
        return super().differences(problem, return_list)


if __name__ == '__main__':
    # some tests
    ds = dataset_from_p3(json.load(open('data/puzzles_test_1.json')))
    # load config
    from omegaconf import OmegaConf
    conf = OmegaConf.load('experiments/conf/quality_p3_train.yaml')
    metric = PredictionProgressCE(**conf.metric)
    print(metric(ds[0]))
    metric = NormalizedPredictionProgressCE(**conf.metric)
    print(metric(ds[0]))
    metric = PredictionProgressCEDiff(**conf.metric)
    print(metric(ds[0]))
    metric = NormalizedPredictionProgressCEDiff(**conf.metric)
    print(metric(ds[0]))
    print('passed')