import torch
from typing import Any, Tuple, Dict, List
from transformers import AutoTokenizer
from dataclasses import dataclass
import transformers
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class TokenizerSeparators:
    assistant_prefix: str = "",
    assistant_suffix: str = "",
    prefix_offset: int = 0,  # Offset to add for masking; Ugly single space with mistral.
    assistant_prefix_ids: List[int] = None

    def set_assistant_prefix_ids(self, tokenizer: AutoTokenizer):
        self.assistant_prefix_ids = tokenizer.encode(
            self.assistant_prefix, add_special_tokens=False
        )


def get_tokenizer_separators(
    tokenizer: AutoTokenizer,
) -> Tuple[AutoTokenizer, TokenizerSeparators]:
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
            assistant_prefix=" [/INST] ", 
            assistant_suffix="</s>", 
            prefix_offset=-1
        )
        tokenizer_separator.set_assistant_prefix_ids(tokenizer)
        return tokenizer, tokenizer_separator
    else:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path} not supported")


def mask_non_assistant_tokens(
    tokenizer: AutoTokenizer,
    conversation: List[Dict[str, str]],
    tokenizer_separator: TokenizerSeparators,
):

    tokens = tokenizer.apply_chat_template(
        conversation, tokenize=True, padding="max_length", truncation=True
    )
    # masked_tokens = [tokenizer.pad_token_id] * len(tokens)  # Start with all tokens masked
    masked_tokens = [IGNORE_TOKEN_ID] * len(tokens)  # Start with all tokens masked
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
            try:
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


if __name__ == "__main__":

    EXAMPLE_FNAME = "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_train_data.json"
    from redteam.train.datasets import get_conversations
    attacker_raw_messages = get_conversations(EXAMPLE_FNAME, "attacker")
    llama_attacker_raw_messages = get_conversations(EXAMPLE_FNAME, "llama_attacker")
    from IPython import embed; embed()
    print(len(attacker_raw_messages))
    c = attacker_raw_messages[0]
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {
            "role": "assistant",
            "content": "I'm doing well, thank you! How can I assist you today?",
        },
        {"role": "user", "content": "What's the weather like today?"},
        {"role": "assistant", "content": "How can I assist you tomorrow?"},
    ]

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    config = transformers.AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        # model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )
    tokenizer, tokenizer_separator = get_tokenizer_separators(tokenizer)
    from IPython import embed; embed()

    masked_tokens = mask_non_assistant_tokens(tokenizer, conversation, tokenizer_separator)["labels"]
    print("Meta-Llama")
    print(tokenizer.decode(torch.where(masked_tokens == IGNORE_TOKEN_ID, tokenizer.pad_token_id, masked_tokens)).replace(tokenizer.pad_token, ""))
    
    mistral_config = transformers.AutoConfig.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        trust_remote_code=True,
    )
    mistral_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        config=mistral_config,
        trust_remote_code=True,
        model_max_length=64,
        padding_side="right",
        use_fast=False,
    )

    mistral_tokenizer, mistral_tokenizer_separator = get_tokenizer_separators(
        mistral_tokenizer
    )
    mistral_masked_tokens = mask_non_assistant_tokens(
        mistral_tokenizer, conversation, mistral_tokenizer_separator
    )["labels"]
    print("Mistral")
    print(mistral_tokenizer.decode(torch.where(mistral_masked_tokens == IGNORE_TOKEN_ID, mistral_tokenizer.pad_token_id, mistral_masked_tokens)).replace(mistral_tokenizer.pad_token, ""))

    # print(
    #     mistral_tokenizer.decode(mistral_masked_tokens).replace(
    #         mistral_tokenizer.pad_token, ""
    #     )
    # )
