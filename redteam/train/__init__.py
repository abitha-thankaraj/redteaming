import transformers
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from transformers import Trainer
from redteam.train.rwr_trainer import RWRTrainer
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


@dataclass
class DatasetStats:
    train_dataset_length: int
    rtg_dict: Dict[float, int]

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
    
    dataset_type: str = field(default="naive_balance", metadata={"help": "How are the data points sampled?"})
    length_key: str = field(default="", metadata={"help": "Key to use for length of the tokenized sequence"})
    max_length: int = field(default=-1, metadata={"help": "Maximum length of the tokenized sequence"})
    model_name: str = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    value_function_type: str = (field(default=""),)
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
    rwr_type: str = field(
        default="exp",             
        metadata={"help": "RWR term type. Available options: exp, raw_rewards."})

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
    algo: str = field(default="rwr")
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

    dataset_stats: DatasetStats = field(default=None)



def get_dataset(algo, data_args, tokenizer, tokenizer_separator):

    fn = {
        "rwr": get_rwr_dataset,
        "sft": get_sft_dataset,
        "dpo": get_dpo_dataset,
    }
    return fn[algo](data_args, tokenizer, tokenizer_separator)
    

def get_trainer(model, tokenizer, train_dataset, eval_dataset, training_args:TrainingArguments):
    fn = {
        "rwr": RWRTrainer,
        "sft": Trainer,
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

    conversation_reward_dict = dataset_helper.get_conversations(num_samples=data_args.num_samples)

    train_dataset = MultiturnSFTDataset(
        conversation_reward_dict["conversations"],
        tokenizer,
        tokenizer_separator,
        ignore_token_id=IGNORE_TOKEN_ID,
    )

    eval_dataset_helper = SFTDatasetHelper(
        data_args.eval_data_path,
        data_args.agent_type,
        dataset_type=None,
        length_key=data_args.length_key,
        max_length=data_args.max_length,
    )

    eval_conversation_reward_dict = eval_dataset_helper.get_conversations(num_samples=data_args.num_samples)

    eval_dataset = MultiturnSFTDataset(
        eval_conversation_reward_dict["conversations"],
        tokenizer,
        tokenizer_separator,
        ignore_token_id=IGNORE_TOKEN_ID,
    )

    return train_dataset, eval_dataset

def get_rwr_dataset(
    data_args: dataclass,
    tokenizer: transformers.AutoTokenizer,
    tokenizer_separator: TokenizerSeparators,
) -> Tuple[Dataset, Dataset]:
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

def get_dpo_dataset(
    data_args: dataclass,
    tokenizer: transformers.AutoTokenizer,
    tokenizer_separator: TokenizerSeparators,
) -> Tuple[Dataset, Dataset]:
    train_dataset = MultiturnDPODatasetFromTensors(dataset_path=data_args.data_path)
    eval_dataset = Dataset()
    return train_dataset, eval_dataset
