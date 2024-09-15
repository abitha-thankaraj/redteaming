import torch
import transformers
from redteam.train.multiturn_sft import (
    apply_prompt_template,
    preprocess,
    tokenize_conversations,
    mask_targets,
    SupervisedDataset,
)
from fastchat.model.model_adapter import get_conversation_template

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


TEST_SOURCES = [
    {
        "conversations": [
            # {
            #     "role": "system",
            #     "content": "System says hello."
            # },
            {"role": "user", "content": "A says marco"},
            {"role": "assistant", "content": "A says polo"},
            # {
            #     "role": "user",
            #     "content": "U says gamma"
            # },
            # {
            #     "role": "assistant",
            #     "content": "A says delta"
            # }
        ]
    },
    {
        "conversations": [
            # {
            #     "role": "system",
            #     "content": "System says hello2."
            # },
            {"role": "user", "content": "User says marco2"},
            {"role": "assistant", "content": "Assistant says polo2"},
            {"role": "user", "content": "User says gamma2"},
            {"role": "assistant", "content": "Assistant says delta2"},
        ]
    },
]


def multiturn_mask(conversation, tokenizer):
    pass


if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    # model_name = "meta-llama/Llama-2-7b-hf"

    training_cache_dir = "/data/tir/projects/tir6/bisk/athankar/projects/.cache"
    training_model_max_length = 1024

    config = transformers.AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=training_cache_dir,
    )
    config.use_cache = False
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     config=config,
    #     trust_remote_code=True,
    #     cache_dir=training_cache_dir,
    # )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        cache_dir=training_cache_dir,
        model_max_length=training_model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    print(f"tokens len: {len(tokenizer)}")
    # model.resize_token_embeddings(len(tokenizer))

    template_id = model_name

    from redteam.train.datasets import get_conversations

    EXAMPLE_FNAME = "/data/tir/projects/tir7/user_data/athankar/redteaming/data/gen_judge_multiturn_conversation/gpt-4_judge_generated_multiturn_conversations_22-00-1722564014.json"
    attacker_raw_messages = get_conversations(EXAMPLE_FNAME, "attacker")

    dataset = SupervisedDataset(attacker_raw_messages, tokenizer, template_id)
    test_data_point = dataset[0]
    print("Decoding inputs")
    print(tokenizer.decode(test_data_point["input_ids"]))
    print("Masked labels")
    print(
        tokenizer.decode(
            torch.where(
                test_data_point["labels"] == IGNORE_TOKEN_ID,
                tokenizer.unk_token_id,
                test_data_point["labels"],
            )
        )
    )

    from IPython import embed

    embed()

    raw_data = TEST_SOURCES

    systems = ""
    # [example["conversations"].get("system", "") for example in raw_data]
    sources = [example["conversations"] for example in raw_data]

    # conversations = [tokenizer.apply_chat_template(conversation, tokenize=False, padding="max_length", return_tensors="pt") for conversation in sources]
    # conv = get_conversation_template(template_id)
    conversations, conv = apply_prompt_template(sources, template_id, systems)
    # tokenizer.apply_chat_template(sources, tokenize=False, padding="max_length")
    input_ids, targets = tokenize_conversations(conversations, tokenizer)

    print(tokenizer.decode(input_ids[0]))
    print(tokenizer.decode(input_ids[1]))

    targets = mask_targets(conversations, targets, tokenizer, conv)

    # data_dict = preprocess(sources, tokenizer, template_id, systems=systems)
    # masked_targets = [target + torch.where(target == -100, 100, 0) for target in targets]
    # masked_inputs = [input_id + torch.where(input_id == -100, 100, 0) for input_id in input_ids]
    processed = dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
    #
    z = torch.where(targets == IGNORE_TOKEN_ID, tokenizer.unk_token_id, targets)
    print(tokenizer.decode(z[0]))
    print(tokenizer.decode(z[1]))

    from IPython import embed

    embed()
