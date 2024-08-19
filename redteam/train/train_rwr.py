# Borrowed from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import os

# Add this here for wandb project name
os.environ["WANDB_PROJECT"] = "redteam"

from dataclasses import dataclass, field
from typing import Optional, Tuple
from typing import Optional, Tuple
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
from redteam.train.dataset_utils import RWRDatasetHelper
import torch, time

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# Args/ Arg parsers
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.1")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    agent_type: str = field(
        default="attacker",
        metadata={"help": "Type of agent to train on. Available options: attacker, defender"},
        # choices=["attacker", "defender", "llama_attacker"],
    )
    train_ratio: float = field(
        default=0.99, metadata={"help": "Ratio of data to use for training."}
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
    data_dir: str,
    agent_type: str,
    train_ratio: float,
    tokenizer: transformers.AutoTokenizer,
    tokenizer_separator: TokenizerSeparators,
) -> Tuple[Dataset, Dataset]:
    """Get train test dataset for multiturn sft"""
    dataset_helper = RWRDatasetHelper(data_dir, agent_type, dataset_type="naive_balance")

    conversation_reward_dict = dataset_helper.get_conversations()
    train_dataset = MultiturnRWRDataset(
        conversation_reward_dict["conversations"],
        tokenizer,
        tokenizer_separator,
        IGNORE_TOKEN_ID,
        conversation_reward_dict["rewards"],
    )
    eval_conversation_reward_dict = RWRDatasetHelper(
        "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_eval_data_llama_rewards_flat.json",
        agent_type,
        dataset_type="naive_balance",
    ).get_conversations()

    eval_dataset = MultiturnRWRDataset(
        eval_conversation_reward_dict["conversations"],
        tokenizer,
        tokenizer_separator,
        IGNORE_TOKEN_ID,
        eval_conversation_reward_dict["rewards"],
    )
    return train_dataset, eval_dataset


def train():
    global local_rank
    # Parse args; All configs sent to the model
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
        data_args.data_path,
        data_args.agent_type,
        data_args.train_ratio,
        tokenizer,
        tokenizer_separator,
    )

    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    # )
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
