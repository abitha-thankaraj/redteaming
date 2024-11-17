import transformers
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from transformers import Trainer
from redteam.train.dpo_trainer import DPOTrainer
from redteam.train.common import (
    set_seed_everywhere,
    safe_save_model_for_hf_trainer,
    TokenizerSeparators,
    get_tokenizer_separators,
)
from redteam.train.datasets import *
from redteam.train.dataset_utils import *

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# Args/ Arg parsers
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(
        default="", metadata={"help": "Path to the evaluation data."}
    )
    agent_type: str = field(
        default="attacker",
        metadata={"help": "Type of agent to train on. Available options: attacker, defender"},
    )

    gamma: float = field(default=0.9, metadata={"help": "Discount factor for the rewards."})

    dataset_type: str = field(
        default="naive_balance", metadata={"help": "How are the data points sampled?"}
    )
    length_key: str = field(
        default="", metadata={"help": "Key to use for length of the tokenized sequence"}
    )
    max_length: int = field(
        default=-1, metadata={"help": "Maximum length of the tokenized sequence"}
    )
    model_name: str = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    value_function_type: str = (field(default=""),)
    value_function_experiment: str = field(default="")
    dataset_type_weights: List[float] = field(
        default_factory=lambda: [0.5, 0.25, 0.25],
        metadata={"help": "Weights for different dataset types"},
    )
    num_samples: int = field(
        default=7814, metadata={"help": "Number of samples to draw from the dataset."}
    )  # 7814 is the number of samples from naive balance

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    algo: str = field(default="sft")
    cache_dir: str = field(
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
    # Adding args for data, model
    data_args: Any = field(default=DataArguments())
    model_args: Any = field(default=ModelArguments())
    exp_desc: str = field(default="")
    # dataset_stats: DatasetStats = field(default=None)


def get_dataset(algo, data_args, tokenizer, tokenizer_separator):

    fn = {
        "sft": get_sft_dataset,
        "dpo": get_dpo_dataset,
        "sft_precomputed": get_precomputed_sft_dataset,
        "sft_precomputed_all_data": get_precomputed_all_sft_dataset,  # This should just be a flag.
    }
    return fn[algo](data_args, tokenizer, tokenizer_separator)


def get_trainer(
    model, tokenizer, train_dataset, eval_dataset, training_args: TrainingArguments
):
    fn = {
        "sft": Trainer,
        "sft_precomputed": Trainer,
        "sft_precomputed_all_data": Trainer,
        "dpo": DPOTrainer,
    }

    return fn[training_args.algo](
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def get_sft_dataset(
    data_args: dataclass,
    tokenizer: transformers.AutoTokenizer,
    tokenizer_separator: TokenizerSeparators,
) -> Tuple[Dataset, Dataset]:
    """Get train test dataset for multiturn sft"""
    dataset_helper = SFTDatasetHelper(
        data_args.data_path,
        data_args.agent_type,
        dataset_type=None,
        length_key=data_args.length_key,
        max_length=data_args.max_length,
    )

    conversation_reward_dict = dataset_helper.get_conversations(
        num_samples=data_args.num_samples
    )

    train_dataset = MultiturnSFTDataset(
        conversation_reward_dict["conversations"],
        tokenizer,
        tokenizer_separator,
        ignore_token_id=IGNORE_TOKEN_ID,
    )

    return train_dataset, Dataset()


def get_precomputed_sft_dataset(
    data_args: dataclass,
    tokenizer: transformers.AutoTokenizer,
    tokenizer_separator: TokenizerSeparators,
) -> Tuple[Dataset, Dataset]:
    """Get train test dataset for multiturn sft"""

    train_dataset = MultiturnSFTDatasetFromTensors(
        dataset_path=data_args.data_path,
    )
    return train_dataset, Dataset()


def get_precomputed_all_sft_dataset(
    data_args: dataclass,
    tokenizer: transformers.AutoTokenizer,
    tokenizer_separator: TokenizerSeparators,
) -> Tuple[Dataset, Dataset]:
    """Get train test dataset for multiturn sft"""

    train_dataset = MultiturnSFTDatasetFromTensors(
        dataset_path=data_args.data_path,
        add_all_data=True,
    )
    return train_dataset, Dataset()    


def get_dpo_dataset(
    data_args: dataclass,
    tokenizer: transformers.AutoTokenizer,
    tokenizer_separator: TokenizerSeparators,
) -> Tuple[Dataset, Dataset]:
    train_dataset = MultiturnDPODatasetFromTensors(dataset_path=data_args.data_path)
    eval_dataset = Dataset()
    return train_dataset, eval_dataset
