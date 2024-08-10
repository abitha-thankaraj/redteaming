import torch, random
import numpy as np
from typing import Tuple, List
from transformers import AutoTokenizer
from dataclasses import dataclass
import transformers


@dataclass
class TokenizerSeparators:
    assistant_prefix: str = ("",)
    assistant_suffix: str = ("",)
    prefix_offset: int = (0,)  # Offset to add for masking; Ugly single space with mistral.
    assistant_prefix_ids: List[int] = None

    def set_assistant_prefix_ids(self, tokenizer: AutoTokenizer):
        self.assistant_prefix_ids = tokenizer.encode(
            self.assistant_prefix, add_special_tokens=False
        )


def get_tokenizer_separators(
    tokenizer: AutoTokenizer,
) -> Tuple[AutoTokenizer, TokenizerSeparators]:
    """
    Set padding tokens and unk for standard tokenizer. [modified in place; should be used for training]
    Set the tokenizer separators for the assistant tokens."""
    if tokenizer.name_or_path == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        tokenizer.unk_token_id = 128004  # Use the finetune right padding token
        tokenizer.pad_token_id = 128004  # Use the finetune right padding token
        tokenizer_separator = TokenizerSeparators(
            assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
            assistant_suffix="<|eot_id|>",
            prefix_offset=0,
        )
        tokenizer_separator.set_assistant_prefix_ids(tokenizer)
        return tokenizer, tokenizer_separator
    elif tokenizer.name_or_path == "mistralai/Mistral-7B-Instruct-v0.1":
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer_separator = TokenizerSeparators(
            assistant_prefix=" [/INST] ", assistant_suffix="</s>", prefix_offset=-1
        )
        tokenizer_separator.set_assistant_prefix_ids(tokenizer)
        return tokenizer, tokenizer_separator
    else:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path} not supported")


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
