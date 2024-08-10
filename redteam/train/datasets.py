import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Dict, Any, List
from transformers import AutoTokenizer
from redteam.train.common import TokenizerSeparators


class MultiturnSFTDataset(
    Dataset
):  # TODO: Add docstring+ key differences b/w single turn and multiturn
    def __init__(
        self,
        conversations: List[Dict],
        tokenizer: Any,
        tokenizer_separator: Any,
        ignore_token_id: int,
    ):

        self.input_ids, self.labels, self.attention_mask = [], [], []
        for conversation in tqdm(conversations):
            data_dict = mask_non_assistant_tokens(
                tokenizer, conversation, tokenizer_separator, ignore_token_id
            )
            self.input_ids.append(data_dict["input_ids"])
            self.labels.append(data_dict["labels"])
            self.attention_mask.append(data_dict["attention_mask"])

        self.input_ids = torch.stack(self.input_ids, dim=0)
        self.labels = torch.stack(self.labels, dim=0)
        self.attention_mask = torch.stack(self.attention_mask, dim=0)

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx],
        }


def mask_non_assistant_tokens(
    tokenizer: AutoTokenizer,
    conversation: List[Dict[str, str]],
    tokenizer_separator: TokenizerSeparators,
    ignore_token_id: int,
):
    """Mask all tokens that are not part of the assistant's outputs.
    Note: We expect the end of assistant's output to be marked by the assistant_suffix token.
    """
    # TODO Add example

    tokens = tokenizer.apply_chat_template(
        conversation, tokenize=True, padding="max_length", truncation=True
    )

    masked_tokens = [ignore_token_id] * len(tokens)  # Start with all tokens masked
    assert tokenizer_separator.assistant_prefix_ids is not None, "Assistant prefix ids not set"
    assistant_prefix_ids = tokenizer_separator.assistant_prefix_ids

    assistant_contents = [
        tokenizer_separator.assistant_prefix
        + msg["content"]
        + tokenizer_separator.assistant_suffix
        for msg in conversation
        if msg["role"] == "assistant"
    ]

    start_index = 0
    for content in assistant_contents:
        content_tokens = tokenizer.encode(content, add_special_tokens=False)
        while True:
            try:  # Note: This will be slow, one time loading doesnt matter.
                start_index = tokens.index(content_tokens[0], start_index)
                if tokens[start_index : start_index + len(content_tokens)] == content_tokens:
                    masked_tokens[
                        start_index
                        + len(assistant_prefix_ids)
                        + tokenizer_separator.prefix_offset : start_index
                        + len(content_tokens)
                    ] = tokens[
                        start_index
                        + len(assistant_prefix_ids)
                        + tokenizer_separator.prefix_offset : start_index
                        + len(content_tokens)
                    ]
                    break
                start_index += 1
            except ValueError:
                break

    input_ids = torch.tensor(tokens)
    labels = torch.tensor(masked_tokens)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
