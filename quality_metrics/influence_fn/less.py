# Using the LESS influence function described in: https://arxiv.org/abs/2402.04333 
import sys
import os
import re
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
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
import datasets
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from quality_metrics.common import (
    QualityMetric,
    Problem,
    dataset_from_p3,
    dict_from_dataset,
    set_seed,
    get_tokenized_hf_dataset,
    add_padding_to_tokenizer
)

from quality_metrics.influence_fn.less_utils import (collect_grads, collect_reps, get_loss)
from quality_metrics.influence_fn.less_utils import get_dataloader


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
            dataset_path: str,
            model_name_or_id: str,
            archive_path: str,
            training_args: OmegaConf,
            model_args: OmegaConf,
            data_args: OmegaConf,
            grad_args: OmegaConf,
            influence_args: OmegaConf,
            **kwargs,
        ):
        self.dataset_path = dataset_path
        self.model_name_or_id = model_name_or_id
        self.archive_path = archive_path
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.grad_args = grad_args
        self.influence_args = influence_args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.influences = {}
    
        self.dtype = torch.float16 if self.model_args.torch_dtype == "float16" else torch.bfloat16

        self._warmup()
        self._compute_train_grads()
        self._compute_archive_grads()
        self._get_influence()
        print('Initilization complete')

    def _warmup(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        transformers.utils.logging.set_verbosity_info()
        log_level = self.training_args.log_level
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, dtype: {self.model_args.torch_dtype}"
        )
        logger.info(f"Training parameters {self.training_args}")
        logger.info(f"Model parameters {self.model_args}")
        logger.info(f"Dataset parameters {self.data_args}")

        # Set seed before initializing model.
        set_seed(self.training_args.seed)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_id)
        # Load training dataset
        train_dataset = json.load(open(self.dataset_path, encoding="utf-8"))
        train_dataset = dict_from_dataset(dataset_from_p3(train_dataset))
        if self.training_args.dev:
            train_dataset = train_dataset[:20]

        if self.data_args.percentage != 1.:
            samples_to_keep = int(self.data_args.percentage * len(train_dataset))
            train_dataset = np.random.choice(train_dataset, samples_to_keep, replace=False).tolist()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_id, torch_dtype=self.dtype)
        add_padding_to_tokenizer(tokenizer)

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
                target_modules=list(self.model_args.lora_target_modules),
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

        train_dataset = get_tokenized_hf_dataset(train_dataset, tokenizer,
                                                 max_length=self.data_args.max_seq_length,
                                                 use_chat_format=self.data_args.use_chat_format)
        
        for index in np.random.randint(0, len(train_dataset), 2):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[int(index)]}.")

        model_params = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
        logger.info(f"trainable model_params: {model_params}")

        if dist.is_initialized() and dist.get_rank() == 0:
            print(model)
        elif not dist.is_initialized():
            print(model)

        training_args = deepcopy(OmegaConf.to_container(self.training_args))
        del training_args['log_level']
        del training_args['local_rank']  # not used for now
        del training_args['device']  # not used for now
        del training_args['n_gpu']  # not used for now
        del training_args['dev']

        training_arguments = TrainingArguments(
            **training_args
        )

        # train model
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False,
            )
        )

        # Training
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer._save_optimizer_and_scheduler(trainer.args.output_dir)

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
        
    def _compute_grads(self, dataset, model_path, output_path, optimizer='adam', suffix='train'):
        tokenizer = AutoTokenizer.from_pretrained(self.grad_args.model_path)
        dataset = get_tokenized_hf_dataset(dataset, tokenizer)
        dataset = dataset.remove_columns('text')
        dataset = dataset.add_column('labels', dataset['input_ids'])
        model = load_model(model_path, self.dtype)

        # pad token is not added by default for pretrained models
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # resize embeddings if needed (e.g. for LlamaTokenizer)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

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
        if self.grad_args.info_type == "grads" and optimizer == "adam":
            optimizer_path = os.path.join(self.grad_args.model_path, "optimizer.pt")
            adam_optimizer_state = torch.load(
                optimizer_path, map_location="cpu")["state"]

        dataloader = get_dataloader(dataset, tokenizer=tokenizer)

        # computes grads and saves to disk
        if self.grad_args.info_type == "reps":
            collect_reps(dataloader, model, self.grad_args.output_path,
                        max_samples=self.grad_args.max_samples)
        elif self.grad_args.info_type == "grads":
            collect_grads(dataloader,
                          model,
                          output_path,
                          proj_dim=self.grad_args.gradient_projection_dimension,
                          gradient_type=optimizer,
                          adam_optimizer_state=adam_optimizer_state,
                          max_samples=self.grad_args.max_samples)
        elif self.grad_args.info_type == "loss":
            get_loss(dataloader, model, self.grad_args.output_path)

    def _get_checkpoints(self):
        checkpoint_pattern = 'checkpoint\-([0-9]+)'
        files = os.listdir(self.training_args.output_dir)
        checkpoint_files = [re.match(checkpoint_pattern, f)[0] for f in files if re.match(checkpoint_pattern, f)]
        checkpoints = [int(re.match(checkpoint_pattern, f)[1]) for f in files if re.match(checkpoint_pattern, f)]
        return checkpoints, checkpoint_files

    def _compute_archive_grads(self):
        dataset = json.load(open(self.archive_path, encoding="utf-8"))
        dataset = dict_from_dataset(dataset_from_p3(dataset))
        # TODO: remove
        if self.training_args.dev:
            dataset = dataset[6:10]
        checkpoints, checkpoint_files = self._get_checkpoints()
        for checkpoint, checkpoint_f in zip(checkpoints, checkpoint_files):
            print(f'(archive) Computing gradients for {checkpoint_f}')
            model_path = os.path.join(self.training_args.output_dir, checkpoint_f)
            output_path = os.path.join(self.training_args.output_dir, 'grads_archive', checkpoint_f)
            self._compute_grads(dataset, model_path, output_path, optimizer='sgd', suffix='archive')
    
    def _compute_train_grads(self):
        dataset = json.load(open(self.dataset_path, encoding="utf-8"))
        dataset = dict_from_dataset(dataset_from_p3(dataset))
        self.train_problem_idx = [p['idx'] for p in dataset]
        # TODO: remove
        if self.training_args.dev:
            dataset = dataset[:6]
        checkpoints, checkpoint_files = self._get_checkpoints()
        for checkpoint, checkpoint_f in zip(checkpoints, checkpoint_files):
            print(f'(train) Computing gradients for {checkpoint_f}')
            model_path = os.path.join(self.training_args.output_dir, checkpoint_f)
            output_path = os.path.join(self.training_args.output_dir, 'grads_train', checkpoint_f)
            self._compute_grads(dataset, model_path, output_path, optimizer='adam', suffix='train')
    
    def _get_influence(self):
        # TODO add checkpoint weights: average lr of the epoch/steps between saves
        # for target_task_name in self.influence_args.target_task_names:  # single task
        checkpoints, checkpoint_files = self._get_checkpoints()

        influence_score = 0
        for i, checkpoint in enumerate(checkpoints):
            output_path = os.path.join(self.training_args.output_dir, 'grads_archive', f'checkpoint-{checkpoint}')
            validation_path = output_path + '/dim8192/all_orig.pt'
            validation_info = torch.load(validation_path)

            if not torch.is_tensor(validation_info):
                validation_info = torch.tensor(validation_info)
            validation_info = validation_info.to(self.device).float()
            output_path = os.path.join(self.training_args.output_dir, 'grads_train', f'checkpoint-{checkpoint}')
            training_path = output_path + '/dim8192/all_orig.pt'
            training_info = torch.load(training_path)

            if not torch.is_tensor(training_info):
                training_info = torch.tensor(training_info)
            training_info = training_info.to(self.device).float()

            influence_score += 1 * \
                calculate_influence_score(
                    training_info=training_info, validation_info=validation_info)
        
        self.influences = {k: v for k, v in zip(self.train_problem_idx, influence_score.cpu().tolist())}
        influence_score = influence_score.mean(-1)
        output_dir = os.path.join(self.training_args.output_dir, 'influence')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(
            output_dir, f"{self.dataset_path.split('/')[-1]}_influence_score.pt")
        torch.save(influence_score, output_file)
        print("Saved influence score to {}".format(output_file))

    def __call__(self, p: Problem, return_list=True):
        # TODO refactoring: actually compute influence scores for all our checkpoints
        return self.influences[p.idx]


@hydra.main(config_path='../../conf', config_name='less_dev', version_base='1.2')
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
