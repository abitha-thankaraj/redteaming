import pytest
from transformers import AutoTokenizer
from redteam.train.datasets import MultiturnRWRDataset, mask_non_assistant_tokens
from redteam.train.dataset_utils import RWRDatasetHelper
from redteam.train.common import get_tokenizer_separators


DATA_FILE = "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_eval_data_llama_rewards_flat_length_added.json"

SAMPLE_CONVERSATION = [
    {"role": "user", "content": "I am"},
    {"role": "assistant", "content": "idk who i am"},
    {"role": "user", "content": "Do you?"},
    {"role": "assistant", "content": "UHHHH"},
    {"role": "user", "content": "Huh?"},
    {"role": "assistant", "content": "OFFLINE RL is a four letter word."},
]

SAMPLE_REWARDS = [0.0, 0.0, 1.0]

special_token_map = {
    0.0: "<|reserved_special_token_247|>",
    1.0: "<|reserved_special_token_246|>",
}


# @pytest.fixture
def get_attacker_conversations_and_rewards():
    dataset_helper = RWRDatasetHelper(
        data_dir=DATA_FILE,
        agent_type="attacker",
        dataset_type="all",
        length_key="Meta-Llama-3.1-8B-Instruct_length",
        max_length=2048 * 4,
    )
    conversation_reward_dict = dataset_helper.get_conversations()
    attacker_conversations = conversation_reward_dict["conversations"]
    rewards = conversation_reward_dict["rewards"]
    return attacker_conversations, rewards


# @pytest.fixture
def get_llama_tokenizer():
    return AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=False, trust_remote_code=True
    )


def test_multiturn_rwr_dataset(
    value_function_type, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
):

    conversations, rewards = [SAMPLE_CONVERSATION], [SAMPLE_REWARDS]
    tokenizer = get_llama_tokenizer()
    tokenizer, tokenizer_separator = get_tokenizer_separators(tokenizer)

    dataset = MultiturnRWRDataset(
        conversations=conversations,
        tokenizer=tokenizer,
        tokenizer_separator=tokenizer_separator,
        ignore_token_id=-100,
        reward_per_turns=rewards,
        gamma=0.9,
        min_reward=0.0,
        value_function_type=value_function_type,
        model_name=model_name,
    )


def test_tokenizer():

    tokenizer = get_llama_tokenizer()
    tokenizer, tokenizer_separator = get_tokenizer_separators(tokenizer)

    conv = SAMPLE_CONVERSATION
    rewards = SAMPLE_REWARDS

    for i, turn in enumerate(conv):
        if turn["role"] == "assistant":
            conv[i]["content"] = special_token_map[rewards[i // 2]] + conv[i]["content"]

    out = mask_non_assistant_tokens(tokenizer, conv, tokenizer_separator, -100)

    # conversations, rewards = get_attacker_conversations_and_rewards()

    # In [2]: out["labels"][:100]
    # Out[2]:
    # tensor([  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
    #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
    #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
    #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
    #         -100, 128255,    307,     74,    889,    602,   1097, 128009,   -100,
    #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
    #         -100,   -100, 128255,     52,  24056,  24056, 128009,   -100,   -100,
    #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
    #         -100, 128254,     37,  28643,  18076,   8429,  48596,     13, 128009,
    #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
    #         -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
    #         -100])

    from IPython import embed

    embed()


if __name__ == "__main__":
    value_function_types = ["overfit_value_function"]
    for value_function_type in value_function_types:
        test_multiturn_rwr_dataset(value_function_type)
