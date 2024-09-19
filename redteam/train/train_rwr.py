# Borrowed from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import os

# Add this here for wandb project name
os.environ["WANDB_ENTITY"] = "mt_redteam"
os.environ["WANDB_PROJECT"] = "redteaming"

from typing import Any

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List
import transformers

import math
import pathlib
import numpy as np
from redteam.train.datasets import MultiturnSFTDataset, MultiturnRWRDataset
from redteam.train.dataset_utils import get_conversations
from redteam.train.common import (
    get_tokenizer_separators,
    safe_save_model_for_hf_trainer,
    set_seed_everywhere,
    TokenizerSeparators,
)
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

from transformers import Trainer
from redteam.train.rwr import RWRTrainer
from redteam.train.dataset_utils import RWRDatasetHelper, RWRDatasetValueFunctionHelper
import torch, time

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# Args/ Arg parsers
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    agent_type: str = field(
        default="attacker",
        metadata={"help": "Type of agent to train on. Available options: attacker, defender"},
    )
    gamma: float = field(default=0.9, metadata={"help": "Discount factor for the rewards."})
    dataset_type: str = field(default="naive_balance")
    length_key: str = field(default="")
    max_length: int = field(default=-1)
    value_function_type: str = (field(default=""),)
    model_name: str = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    value_function_experiment: str = field(default=None)
    dataset_type_weights: List[float] = field(
        default_factory=lambda: [0.5, 0.25, 0.25],
        metadata={"help": "Weights for different dataset types"},
    )
    num_samples: int = field(
        default=7814, metadata={"help": "Number of samples to draw from the dataset."}
    )  # 7814 is the number of samples from naive balance


@dataclass
class RWRArguments:
    rwr_temperature: float = field(
        default=1.0,
        metadata={"help": "RWR temperature (β) . Higher -> | Lower -> "},
    )
    rwr_type: str = (
        field(default="exp", metadata={"help": "RWR term type. Available options: exp"}),
    )

    rwr_value_function_token_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the value function tokens in the RWR term."},
    )
    r_max: float = field(
        default=2.71,  # 3 turns ; \gamma = 0.9; r_max = \gamma^2 + \gamma + 1
        metadata={"help": "Maximum reward value for the value function tokens."},
    )
    r_min: float = field(
        default=0.0,
        metadata={"help": "Minimum reward value for the value function tokens."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default="/data/tir/projects/tir6/bisk/athankar/projects/.cache"
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    torch_empty_cache_steps: int = field(
        default=1, metadata={"help": "Number of steps to call torch.cuda.empty_cache()"}
    )
    save_only_model: bool = field(
        default=True, metadata={"help": "Save only the model and not the trainer."}
    )
    # Adding args for data, model and rwr
    data_args: Any = field(default=None)
    model_args: Any = field(default=None)
    rwr_args: Any = field(default=None)
    exp_desc: str = field(default="")


# Debugging utils
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# TODO: Remove
def rank0_debug(interval=9999999):
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        breakpoint()
    else:
        time.sleep(interval)


def get_dataset(
    data_args: dataclass,
    tokenizer: transformers.AutoTokenizer,
    tokenizer_separator: TokenizerSeparators,
) -> Tuple[Dataset, Dataset]:
    """Get train test dataset for multiturn sft"""
    # dataset_helper = RWRDatasetHelper(
    #     data_args.data_path,
    #     data_args.agent_type,
    #     dataset_type=data_args.dataset_type,
    #     length_key=data_args.length_key,
    #     max_length=data_args.max_length,
    # )
    dataset_helper = RWRDatasetValueFunctionHelper(
        data_args.data_path,
        data_args.agent_type,
        dataset_type=data_args.dataset_type,
        length_key=data_args.length_key,
        max_length=data_args.max_length,
        dataset_type_weights=data_args.dataset_type_weights,
    )

    conversation_reward_dict = dataset_helper.get_conversations(
        num_samples=data_args.num_samples
    )

    train_dataset = MultiturnRWRDataset(
        conversation_reward_dict["conversations"],
        tokenizer,
        tokenizer_separator,
        IGNORE_TOKEN_ID,
        conversation_reward_dict["rewards"],
        gamma=data_args.gamma,
        value_function_type=data_args.value_function_type,
        model_name=data_args.model_name,
        value_function_experiment=data_args.value_function_experiment,
    )
    # You never use the eval dataset, so ignore this for now
    eval_conversation_reward_dict = RWRDatasetValueFunctionHelper(
        data_args.eval_data_path,
        data_args.agent_type,
        dataset_type=data_args.dataset_type,
        length_key=data_args.length_key,
        max_length=data_args.max_length,
        dataset_type_weights=data_args.dataset_type_weights,
    ).get_conversations(num_samples=10)

    eval_dataset = MultiturnRWRDataset(
        eval_conversation_reward_dict["conversations"],
        tokenizer,
        tokenizer_separator,
        IGNORE_TOKEN_ID,
        eval_conversation_reward_dict["rewards"],
        gamma=data_args.gamma,
        value_function_type=data_args.value_function_type,
        model_name=data_args.model_name,
        value_function_experiment=data_args.value_function_experiment,
    )
    return train_dataset, eval_dataset


def train():
    global local_rank
    # Parse args; All configs sent to the model
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, RWRArguments, TrainingArguments)
    )
    model_args, data_args, rwr_args, training_args = parser.parse_args_into_dataclasses()

    # Set rwr args
    rwr_args.r_max = data_args.gamma**2 + data_args.gamma + 1
    rwr_args.r_min = 0.0

    training_args.data_args = data_args
    training_args.model_args = model_args
    training_args.rwr_args = rwr_args

    assert (
        training_args.model_max_length >= data_args.max_length
    ), "Model max length should be greater than data max length"

    # Seed everything
    set_seed_everywhere(training_args.seed)

    # Mostly for debugging
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )

    # Set RoPE scaling factor - useful when you want to train with a larger context window
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    # Tie the weights - Commonly used for memory efficient training?
    model.tie_weights()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # Add padding token to the tokenizer + Add masking values. tokenizer separator will be used for masking in dataloader
    tokenizer, tokenizer_separator = get_tokenizer_separators(tokenizer)
    # NOTE: if the token_id exceed the vocab_size will cause failing in training process! we need add special config and resize the embedding size!
    model.resize_token_embeddings(len(tokenizer))

    train_dataset, eval_dataset = get_dataset(
        data_args=data_args,
        tokenizer=tokenizer,
        tokenizer_separator=tokenizer_separator,
    )

    trainer = RWRTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":

    train()
