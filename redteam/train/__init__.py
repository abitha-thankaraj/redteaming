import transformers
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


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
    max_length: int = field(default=-1, metadata="Maximum length of the tokenized sequence")
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



def get_dataset():
    pass


def get_trainer():
    pass







