import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Dict, Any, List
from transformers import AutoTokenizer
from redteam.train.common import TokenizerSeparators

# def stripped_decode(tokenizer, masked_tokens):
#     return tokenizer.decode(torch.where(masked_tokens == -100, tokenizer.unk_token_id, masked_tokens)).replace(tokenizer.unk_token, "")


class MultiturnSFTDataset(Dataset):
    # TODO: Add docstring + key differences b/w single turn and multiturn
    def __init__(
        self,
        conversations: List[Dict],
        tokenizer: Any,
        tokenizer_separator: Any,
        ignore_token_id: int,
    ):

        self.input_ids, self.labels, self.attention_mask = [], [], []
        for i, conversation in tqdm(
            enumerate(conversations), desc="Masking Conversations - MT-SFT"
        ):
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
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx],
        }


class MultiturnRWRDataset(MultiturnSFTDataset):
    """Reward Weighted Returns (RWR),  an extension of SFT.
    Assumptions:
    1. Each turn in the conversation is assigned a reward.
    2. For conversation-level rewards, only the final turn receives a raw reward. The other turns should have r_min.
    3. For any token in a specific turn, its reward is the "reward-to-go" calculated for that turn.
       The reward-to-go represents the cumulative future rewards from that point onward.

    Note: This implementation supports both conversation-level and potential future
    turn-level reward labeling without requiring changes to this class.
    """

    def __init__(
        self,
        conversations: List[Dict[str, str]],
        tokenizer: Any,
        tokenizer_separator: Any,
        ignore_token_id: int,
        reward_per_turns: List[List[float]],
        gamma: float = 0.9,
        min_reward: float = 0.0,
        # Create a dataclass or a dictionary?
        value_function_type: str = "",
        value_function_experiment: str = None,
        model_name: str = None,
    ):
        # Ugly; but should work.
        if value_function_type != "":
            value_function_reserved_strs = get_value_function_keywords(
                value_function_type=value_function_type, gamma=gamma, model_name=model_name
            )
            for i, conversation in enumerate(conversations):
                conversations[i] = relabel_conversation(
                    value_function_experiment,
                    value_function_type=value_function_type,
                    conversation=conversations[i],
                    reward_per_turn=reward_per_turns[i],
                    gamma=gamma,
                    value_function_reserved_strs=value_function_reserved_strs,
                )

        super().__init__(
            conversations=conversations,
            tokenizer=tokenizer,
            tokenizer_separator=tokenizer_separator,
            ignore_token_id=ignore_token_id,
        )

        self.rewards = []
        # print(self.incorrect)

        for i, reward_per_turn in tqdm(enumerate(reward_per_turns), desc="Rewards - RWR"):
            self.rewards.append(
                get_token_level_reward_to_gos(
                    rewards_per_turn=reward_per_turn,
                    masked_tokens=self.labels[i],
                    ignore_token_id=ignore_token_id,
                    gamma=gamma,
                    min_reward=min_reward,
                )
            )
        self.rewards = torch.stack(self.rewards, dim=0)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx],
            "rewards": self.rewards[idx],
        }


def mask_non_assistant_tokens(
    tokenizer: AutoTokenizer,
    conversation: List[Dict[str, str]],
    tokenizer_separator: TokenizerSeparators,
    ignore_token_id: int,
) -> Dict[str, torch.Tensor]:
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

    start_index = -1
    for content in assistant_contents:
        content_tokens = tokenizer.encode(content, add_special_tokens=False)
        while True:
            try:  # Note: This will be slow, one time loading doesnt matter.
                start_index += 1
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
            except Exception as e:
                break

    input_ids = torch.tensor(tokens)
    labels = torch.tensor(masked_tokens)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def get_reward_to_gos(rewards: List[float], gamma: float) -> List[float]:
    """Calculate reward to go for each turn.
    Args:
        rewards: List of raw rewards per turn
        gamma: Discount factor to calculate reward to go
    reward_to_go_{i} = \Sum_{j=i}^{n} \gamma^{j-i} * rewards[j]
    """
    reward_to_gos = [0] * len(rewards)
    reward_to_go = 0
    for i in range(len(rewards) - 1, -1, -1):
        reward_to_go = rewards[i] + gamma * reward_to_go
        reward_to_gos[i] = reward_to_go
    return reward_to_gos


def get_token_level_reward_to_gos(
    rewards_per_turn: List[float],
    masked_tokens: torch.Tensor,
    ignore_token_id: Any,
    gamma: float,
    min_reward: float,
) -> torch.Tensor:
    """Calculate reward to go for each token in the conversation.
    Args:
        rewards_per_turn: List of raw rewards per turn
        masked_tokens: Masked tokens in the conversation; All tokens except assistant responses should be masked.
        ignore_token_id: Token id to ignore; depends on label_smoother used; default : -100
        gamma: Discount factor to calculate reward to go
        min_reward: Minimum reward to assign to tokens; All masked tokens should be assigned this reward by default.
    """
    # All assistant responses are unmasked.
    unmasked_turn_indices = torch.where(masked_tokens != ignore_token_id)[0]
    # Calculate differences between consecutive indices
    index_diffs = torch.diff(unmasked_turn_indices)

    # Select turn boundary indices from unmasked_turn_indices [s_0, s_1, .... s_t, e_t]
    turn_boundaries = torch.cat(
        [
            torch.tensor([0]),  # s_0
            torch.where(index_diffs != 1)[0] + 1,  # s_1, s_2, .... s_t
            torch.tensor([len(unmasked_turn_indices)]),  # e_t
        ]
    )
    # Turn masks for each token. -1 for masked tokens.
    # Assistant responses are unmasked -> Assigned turn number.
    turn_masks = torch.full_like(masked_tokens, -1.0)
    for i in range(len(turn_boundaries) - 1):
        start = unmasked_turn_indices[turn_boundaries[i]]
        end = unmasked_turn_indices[turn_boundaries[i + 1] - 1] + 1
        turn_masks[start:end] = i

    assert len(turn_boundaries) - 1 == len(
        rewards_per_turn
    ), "Number of turns should match number of rewards"
    reward_to_gos = get_reward_to_gos(rewards_per_turn, gamma)
    token_rewards = torch.full_like(masked_tokens, min_reward, dtype=torch.float)

    for i in range(len(reward_to_gos)):
        # replace all token_rewards in turn i with reward_to_gos[i]
        token_rewards.masked_fill_(turn_masks.eq(i), reward_to_gos[i])
    return token_rewards


# relabelling strategies
# 1. Binary. Prefix: Expected mode collapse? Prefix - semantically, conditional generation? SUffix? Model's belief of it's own response?
# 2. Value function. -> Get rtgs; disctrete value function; <GOOD> <BAD> <NEUTRAL> ?


def relabel_conversation(
    value_function_experiment: str,
    value_function_type: str,
    conversation: List[Dict[str, str]],
    reward_per_turn: List[List[float]],
    gamma: float = 0.9,
    value_function_reserved_strs=None,
) -> List[Dict[str, str]]:
    """
    Add prefix to assistant response on value function type.
    Args:
        value_function_experiment: Experiment type to use for relabelling. | prefix and overfit
        value_function_type: Type of value function to use for relabelling.
        conversation: List of messages in the conversation.
        reward_per_turn: List of rewards per turn.
        gamma: Discount factor for reward to go.
        model_name: Model name to use for special tokens.
    """
    # add value function to the beginning/end of each assistant response.

    reward_to_gos = get_reward_to_gos(reward_per_turn, gamma)

    if value_function_type in ["binary", "natural_lang_binary"]:
        rews = reward_per_turn
    elif value_function_type in ["multilabel", "natural_lang_multiclass"]:
        rews = reward_to_gos
    else:
        raise NotImplementedError(f"Value function type {value_function_type} not implemented")

    for j, message in enumerate(conversation):
        if message["role"] == "assistant":
            if value_function_experiment == "prefix":
                # prefic the special token or the natural language value function
                message["content"] = (
                    f"{value_function_reserved_strs[rews[j//2]]}{message['content']}"  # do //2 becuase of turn level rewards
                )
            elif value_function_experiment == "overfit":
                message["content"] = (
                    f"{value_function_reserved_strs[rews[j//2]]} $$"  # Add $$ in the end to prevent stripping of the spaces
                )
            else:
                raise NotImplementedError(
                    f"Value function strategy {value_function_experiment} not implemented"
                )
    return conversation


def get_value_function_keywords(
    value_function_type, gamma=0.9, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
):

    LLAMA_SPECIAL_TOKEN_MAP = {
        # Single turn - special tokens
        "binary": {0.0: "<|reserved_id_24|>", 1.0: "<|reserved_id_25|>"},
        # 3 turn based
        # Use discounted rewards for each turn
        "multilabel": {
            0.0: "<|reserved_id_24|>",
            1.0: "<|reserved_id_25|>",
            gamma: "<|reserved_id_26|>",
            gamma**2: "<|reserved_id_27|>",
            gamma + gamma**2: "<|reserved_id_28|>",
            1 + gamma: "<|reserved_id_29|>",
            1 + gamma**2: "<|reserved_id_30|>",
            1 + gamma + gamma**2: "<|reserved_id_31|>",
        },
        # Verbalize belief of the model
        "natural_lang_binary": {0.0: "RED     ", 1.0: "BLUE     "},
        "natural_lang_multiclass": {
            0.0: "RED     ",
            1.0: "BLUE     ",
            gamma: "GREEN     ",
            gamma**2: "YELLOW     ",
            gamma + gamma**2: "PURPLE     ",
            1 + gamma: "ORANGE     ",
            1 + gamma**2: "BROWN     ",
            1 + gamma + gamma**2: "BLACK     ",
        },
    }

    if model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        if value_function_type == "all":
            return LLAMA_SPECIAL_TOKEN_MAP
        return LLAMA_SPECIAL_TOKEN_MAP[value_function_type]
    else:
        raise ValueError(f"Model {model_name} not supported")


# def get_reward_to_gos(rewards, gamma):
#     reward_to_gos = [0] * len(rewards)
#     for j in range(len(rewards)):
#         reward_to_go = rewards[j]
#         for k in range(j+1, len(rewards)):
#             reward_to_go = reward_to_go + (gamma**(k-j) * rewards[k])
#         reward_to_gos[j] = reward_to_go
#     return reward_to_gos
