# Using the LESS influence function described in: https://arxiv.org/abs/2402.04333 
import sys
import os
import json
import logging
from omegaconf import OmegaConf

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

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
            model_id_or_path: str,
            archive_path_or_list: str,
            dataset_path: str,
            save_path: str,
            training_args: OmegaConf,
            model_args: OmegaConf,
            data_args: OmegaConf,
        ):

        self.lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        self.model_id_or_path = model_id_or_path
        self.save_path = save_path
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.store = []  # not sure how this will work

        self._load_data(archive_path_or_list, dataset_path)
        self._warmup()
        self._compute_train_grads()
        self._compute_archive_grads()

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
        less.set_seed(self.training_args.seed)

        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        # Load training dataset
        train_dataset = self.dataset  # TODO some work needed here

        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path, torch_dtype=self.model_args.torch_dtype)
        less.train.model_arguments.add_padding_to_tokenizer(tokenizer)

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
        less.train.self.data_argsments.get_data_statistics(train_dataset)

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
        
    def _compute_grads(self):
        ...

    def _compute_archive_grads(self):
        ...
    
    def _compute_train_grads(self):
        ...
    


class InstantLESS(QualityMetric):  # Would be interesting to cosine sim these one with the previous ones
    """
    Attempt at a one-step version of LESS (only compute the cosine sim for data and val gradients).
    We keep the warmup period for now.
    """
    pass


if __name__ == "__main__":
    # test goes here
    NotImplemented