import torch, random
import numpy as np
from typing import Tuple, List
from transformers import AutoTokenizer
from dataclasses import dataclass
import transformers

import logging

logger = logging.getLogger(__name__)


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
    elif tokenizer.name_or_path == "mistralai/Mistral-7B-Instruct-v0.3":
        # TODO : Test tokenizer_separator
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer_separator = TokenizerSeparators(
            assistant_prefix=" [/INST] ", assistant_suffix="</s>", prefix_offset=-1
        )
        tokenizer_separator.set_assistant_prefix_ids(tokenizer)
        return tokenizer, tokenizer_separator
    elif tokenizer.name_or_path == "meta-llama/Llama-2-7b-hf":
        # Only for tests.
        return tokenizer, None
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


# def update_control_tokens_llama(tokenizer: AutoTokenizer,
#                                 new_tokens: List[str],
#                                 make_special: bool = False):

#     """Add new tokens to the tokenizer."""

#     # if not make_special:
#     #     tokenizer.add_tokens(new_tokens)
#     #     logger.warn(f"Added {len(new_tokens)} new tokens to the tokenizer. Resize model embeddings")
#     # else:
#     #     raise NotImplementedError("Special tokens not implemented")

# https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py


@torch.inference_mode
def mean_of_trained_tokens(model, eps=1e-16):
    """
    Llama-3 for eg has untrained vectors in the base model.
    These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
    We reset them to the mean of the rest of the tokens
    """
    embedding_matrix = model.get_input_embeddings().weight.clone()
    lm_head_matrix = model.get_output_embeddings().weight.clone()

    # Get untrained tokens
    indicator_untrained = torch.amax(embedding_matrix, axis=1) <= eps
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained
    # if n_untrained != 0:
    #     print(
    #         f"Unsloth: Not an error, but your model has {n_untrained} untrained tokens.\n"\
    #         "We shall set them to the mean of the other trained tokens."
    #     )
    # pass

    # Get sum of all items
    sum_embedding = torch.sum(embedding_matrix, dtype=torch.float32, axis=0)
    sum_lm_head = torch.sum(lm_head_matrix, dtype=torch.float32, axis=0)

    # Remove bad tokens
    sum_embedding -= torch.sum(embedding_matrix[where_untrained], dtype=torch.float32, axis=0)
    sum_lm_head -= torch.sum(lm_head_matrix[where_untrained], dtype=torch.float32, axis=0)

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = sum_embedding / n_trained
    mean_lm_head = sum_lm_head / n_trained

    return mean_embedding, mean_lm_head
