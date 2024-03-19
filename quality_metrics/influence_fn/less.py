# Using the LESS influence function described in: https://arxiv.org/abs/2402.04333 
import sys
import os
import json
import logging

import hydra
from omegaconf import OmegaConf
from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist

import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer)
import datasets
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

import less
from quality_metrics.common import (
    QualityMetric,
    Problem,
    create_model_and_tokenizer,
    dataset_from_p3
)

from less_utils import (collect_grads, collect_reps, get_loss)
# from less.data_selection.collect_grad_reps import (collect_grads, collect_reps,
#                                                    get_loss)
# from less_utils import get_training_dataset
# from less.data_selection.get_training_dataset import get_training_dataset
from less_utils import get_dataloader
# from less.data_selection.get_validation_dataset import (get_dataloader,
#                                                         get_dataset)
from less_utils import get_data_statistics
# from less.train.data_arguments import DataArguments, get_data_statistics
from less_utils import add_padding_to_tokenizer
# from less.train.model_arguments import ModelArguments, add_padding_to_tokenizer


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# TODO maybe move to utils
def load_model(model_name_or_path: str,
               torch_dtype: Any = torch.bfloat16) -> Any:
    """
    Load a model from a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        torch_dtype (Any, optional): The torch data type. Defaults to torch.bfloat16.

    Returns:
        Any: The loaded model.
    """

    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto")
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="auto")

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model


def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores


class LESS(QualityMetric):
    """
    LESS influence function. Expensive to compute, since we have to perform a training run with the provided 
    example to compute gradients at each time step. We implement this metric for experimental purposes and 
    only support computing influence on a set of train examples, those we use for training.

    - Warmup the model for a few epochs on a subset of the data (to init the Adam moments);
    - Compute the gradients over the training set and store them, accessible by the ID of the puzzle.
    - Get the validation gradients (we'll use the p3 train set as validation set, for now);
    
    After these steps, when a problem is given as input, match the puzzle id and get the gradient, and 
    compute the influence function.
    """
    def __init__(
            self,
            training_args: OmegaConf,
            model_args: OmegaConf,
            data_args: OmegaConf,
            grad_args: OmegaConf,
            influence_args: OmegaConf,
            **kwargs,
        ):

        self.lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.grad_args = grad_args
        self.influence_args = influence_args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.influences = {}
    
        self.dtype = torch.float16 if self.grad_args.torch_dtype == "float16" else torch.bfloat16

        # self._load_data(archive_path_or_list, dataset_path)
        self._warmup()
        self._compute_train_grads()
        self._compute_archive_grads()
        self._get_influence()

    def _load_data(self, archive_path_or_list, dataset_path):
        # load archive
        if isinstance(archive_path_or_list, str):
            dataset = json.load(open(archive_path_or_list, 'r'))
            self.problem_archive = dataset_from_p3(dataset)
        else:
            self.problem_archive = archive_path_or_list

        # load dataset
        # TODO apply chat template, and use transformer Datasets?
        dataset = json.load(open(dataset_path, 'r'))
        self.dataset = dataset_from_p3(dataset)  # some work needed to plug this guy in the dataloader for less

    def _warmup(self):
        # TODO add option for loading model and optim state at init
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        transformers.utils.logging.set_verbosity_info()
        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        logger.info(f"Training parameters {self.training_args}")
        logger.info(f"Model parameters {self.model_args}")
        logger.info(f"Dataset parameters {self.data_args}")

        # Set seed before initializing model.
        less.set_seed(self.training_args.seed)  # TODO do this with our own fns

        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        # Load training dataset
        train_dataset = self.dataset  # TODO some work needed here

        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path, torch_dtype=self.model_args.torch_dtype)
        add_padding_to_tokenizer(tokenizer)  # TODO do this with our own fns

        # resize embeddings if needed (e.g. for LlamaTokenizer)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
            # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
            if isinstance(model, PeftModel):
                model.get_input_embeddings().weight.requires_grad = False
                model.get_output_embeddings().weight.requires_grad = False
        
        # peft the model
        if not isinstance(model, PeftModel) and self.model_args.lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.model_args.lora_r,
                lora_alpha=self.model_args.lora_alpha,
                lora_dropout=self.model_args.lora_dropout,
                target_modules=self.model_args.lora_target_modules,
            )
            model = get_peft_model(model, lora_config)
            logger.info(
                f"Applied LoRA to model."
            )
            model.print_trainable_parameters()

            # for checkpointing
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # TODO some work here
        get_data_statistics(train_dataset)

        if "dataset" in train_dataset.features:
            train_dataset = train_dataset.remove_columns(
                ["dataset", "id", "messages"])
                
        for index in np.random.randint(len(train_dataset), 1):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

        model_params = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
        logger.info(f"trainable model_params: {model_params}")

        analysis_dataset = None
        if self.training_args.analysis_mode:
            from less.data_selection.get_validation_dataset import get_dataset
            analysis_dataset = get_dataset(self.training_args.analysis_dataset,
                                        data_dir=self.data_args.data_dir,
                                        tokenizer=tokenizer,
                                        max_length=self.data_args.max_seq_length)

        if dist.is_initialized() and dist.get_rank() == 0:
            print(model)
        elif not dist.is_initialized():
            print(model)

        # train model
        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=analysis_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(  # TODO check this works
                tokenizer=tokenizer, model=model, padding="longest")
        )

        # Training
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # remove the full model in the end to save space, only adapter is needed
        if isinstance(model, PeftModel):
            pytorch_model_path = os.path.join(
                self.training_args.output_dir, "pytorch_model_fsdp.bin")
            os.remove(pytorch_model_path) if os.path.exists(
                pytorch_model_path) else None
        
    def _compute_grads(self, dataset):  # TODO simplify
        # tokenizer = AutoTokenizer.from_pretrained(self.grad_args.model_path)
        model = load_model(self.grad_args.model_path, self.dtype)
        # model = self.model  # TODO beware of stateful modifs

        # pad token is not added by default for pretrained models
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # resize embeddings if needed (e.g. for LlamaTokenizer)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            model.resize_token_embeddings(len(self.tokenizer))

        if self.grad_args.initialize_lora:
            assert not isinstance(model, PeftModel)
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.grad_args.lora_r,
                lora_alpha=self.grad_args.lora_alpha,
                lora_dropout=self.grad_args.lora_dropout,
                target_modules=self.grad_args.lora_target_modules,
            )
            model = get_peft_model(model, lora_config)

        if isinstance(model, PeftModel):
            model.print_trainable_parameters()

        adam_optimizer_state = None
        if self.grad_args.info_type == "grads" and self.grad_args.gradient_type == "adam":
            optimizer_path = os.path.join(self.grad_args.model_path, "optimizer.bin")
            adam_optimizer_state = torch.load(
                optimizer_path, map_location="cpu")["state"]

        # if self.grad_args.task is not None:  # modification here
        #     dataset = get_dataset(self.grad_args.task,
        #                         data_dir=self.grad_args.data_dir,
        #                         tokenizer=self.tokenizer,
        #                         chat_format=self.grad_args.chat_format,
        #                         use_chat_format=self.grad_args.use_chat_format,
        #                         max_length=self.grad_args.max_length,
        #                         zh=self.grad_args.zh)
        #     dataloader = get_dataloader(dataset, tokenizer=self.tokenizer)
        # else:
        #     assert self.grad_args.train_file is not None
        #     dataset = get_training_dataset(
        #         self.grad_args.train_file, self.tokenizer, self.grad_args.max_length, sample_percentage=1.0)
        #     columns = deepcopy(dataset.column_names)
        #     columns.remove("input_ids")
        #     columns.remove("labels")
        #     columns.remove("attention_mask")
        #     dataset = dataset.remove_columns(columns)
        dataloader = get_dataloader(dataset, tokenizer=self.tokenizer)

        # this saves the grads to disk
        if self.grad_args.info_type == "reps":
            collect_reps(dataloader, model, self.grad_args.output_path,
                        max_samples=self.grad_args.max_samples)
        elif self.grad_args.info_type == "grads":
            collect_grads(dataloader,
                        model,
                        self.grad_args.output_path,
                        proj_dim=self.grad_args.gradient_projection_dimension,
                        gradient_type=self.grad_args.gradient_type,
                        adam_optimizer_state=adam_optimizer_state,
                        max_samples=self.grad_args.max_samples)
        elif self.grad_args.info_type == "loss":
            get_loss(dataloader, model, self.grad_args.output_path)


    def _compute_archive_grads(self):
        # load archive dataset
        # TODO change arguments so that we do sgd instead of adam
        dataset = ...
        self._compute_grads(dataset)
    
    def _compute_train_grads(self):
        dataset = ...
        self._compute_grads(dataset)
    
    def _get_influence(self):
        if sum(self.influence_args.checkpoint_weights) != 1:
            s = sum(self.influence_args.checkpoint_weights)
            self.influence_args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

        for target_task_name in self.influence_args.target_task_names:
            for train_file_name in self.influence_args.train_file_names:
                influence_score = 0
                for i, ckpt in enumerate(self.influence_args.ckpts):
                    # validation_path = self.influence_args.validation_gradient_path.format(
                    # target_task_name, ckpt)
                    validation_path = self.influence_args.validation_gradient_path.format(
                        ckpt, target_task_name)
                    validation_info = torch.load(validation_path)

                    if not torch.is_tensor(validation_info):
                        validation_info = torch.tensor(validation_info)
                    validation_info = validation_info.to(self.device).float()
                    # gradient_path = self.influence_args.gradient_path.format(train_file_name, ckpt)
                    gradient_path = self.influence_args.gradient_path.format(ckpt, train_file_name)
                    training_info = torch.load(gradient_path)

                    if not torch.is_tensor(training_info):
                        training_info = torch.tensor(training_info)
                    training_info = training_info.to(self.device).float()

                    influence_score += self.influence_args.checkpoint_weights[i] * \
                        calculate_influence_score(
                            training_info=training_info, validation_info=validation_info)
                influence_score = influence_score.mean(-1)[0]
                output_dir = os.path.join(self.influence_args.output_path, target_task_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_file = os.path.join(
                    self.influence_args.output_path, target_task_name, f"{train_file_name}_influence_score.pt")
                torch.save(influence_score, output_file)
                print("Saved influence score to {}".format(output_file))

    
    def __call__(self, p: Problem):
        return self.influences[p.idx]


class InstantLESS(QualityMetric):  # Would be interesting to cosine sim these one with the previous ones
    """
    Attempt at a one-step version of LESS (only compute the cosine sim for data and val gradients).
    We keep the warmup period for now.
    """
    pass


@hydra.main(config_path='../../conf', config_name='less_dev')
def main(args):
    less_metric = LESS(
        args.training, 
        args.model,
        args.data,
        args.grad,
        args.influence,
    )
    print('Done!')


if __name__ == "__main__":
    main()