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
        value_function_type=None,
    ):
        self.input_ids, self.labels, self.attention_mask, self.value_function_token_idxs = (
            [],
            [],
            [],
            [],
        )
        for i, conversation in tqdm(
            enumerate(conversations), desc="Masking Conversations - MT-SFT"
        ):
            data_dict = mask_non_assistant_tokens(
                tokenizer,
                conversation,
                tokenizer_separator,
                ignore_token_id,
                value_function_type,
            )
            self.input_ids.append(data_dict["input_ids"])
            self.labels.append(data_dict["labels"])
            self.attention_mask.append(data_dict["attention_mask"])
            self.value_function_token_idxs.append(data_dict["value_function_token_idxs"])

        self.input_ids = torch.stack(self.input_ids, dim=0)
        self.labels = torch.stack(self.labels, dim=0)
        self.attention_mask = torch.stack(self.attention_mask, dim=0)
        self.value_function_token_idxs = torch.stack(self.value_function_token_idxs, dim=0)

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
        if value_function_type not in ["", "multilabel", "binary"]:
            raise NotImplementedError(
                f"Value function type {value_function_type} not implemented"
            )

        super().__init__(
            conversations=conversations,
            tokenizer=tokenizer,
            tokenizer_separator=tokenizer_separator,
            ignore_token_id=ignore_token_id,
            value_function_type=value_function_type,
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
            "value_function_token_idxs": self.value_function_token_idxs[idx],
        }


def mask_non_assistant_tokens(
    tokenizer: AutoTokenizer,
    conversation: List[Dict[str, str]],
    tokenizer_separator: TokenizerSeparators,
    ignore_token_id: int,
    value_function_type: str = None,
) -> Dict[str, torch.Tensor]:
    """Mask all tokens that are not part of the assistant's outputs.
    Note: We expect the end of assistant's output to be marked by the assistant_suffix token.
    """
    # TODO Add example
    tokens = tokenizer.apply_chat_template(
        conversation, tokenize=True, padding="max_length", truncation=True
    )

    masked_tokens = [ignore_token_id] * len(tokens)  # Start with all tokens masked
    value_function_masked_tokens = [ignore_token_id] * len(tokens)
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
                    if value_function_type in ["binary", "multilabel"]:
                        # single token
                        value_function_masked_tokens[
                            start_index
                            + len(assistant_prefix_ids)
                            + tokenizer_separator.prefix_offset
                        ] = tokens[
                            start_index
                            + len(assistant_prefix_ids)
                            + tokenizer_separator.prefix_offset
                        ]
                    break
            except Exception as e:
                break

    input_ids = torch.tensor(tokens)
    labels = torch.tensor(masked_tokens)
    # #TODO: Fix this.
    # value_function_token_idxs = torch.tensor(value_function_masked_tokens)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        value_function_token_idxs=torch.tensor(value_function_masked_tokens).ne(
            ignore_token_id
        ),
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
                # prefix the special token or the natural language value function
                message[
                    "content"
                ] = f"{value_function_reserved_strs[rews[j//2]]}{message['content']}"  # do //2 becuase of turn level rewards
            elif value_function_experiment == "overfit":
                message[
                    "content"
                ] = f"{value_function_reserved_strs[rews[j//2]]} $$"  # Add $$ in the end to prevent stripping of the spaces
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
        "binary": {0.0: "<|reserved_special_token_24|>", 1.0: "<|reserved_special_token_25|>"},
        # 3 turn based
        # Use discounted rewards for each turn
        "multilabel": {
            0.0: "<|reserved_special_token_24|>",
            1.0: "<|reserved_special_token_25|>",
            gamma: "<|reserved_special_token_26|>",
            gamma**2: "<|reserved_special_token_27|>",
            gamma + gamma**2: "<|reserved_special_token_28|>",
            1 + gamma: "<|reserved_special_token_29|>",
            1 + gamma**2: "<|reserved_special_token_30|>",
            1 + gamma + gamma**2: "<|reserved_special_token_31|>",
        },
        # Verbalize belief of the model
        # "natural_lang_binary": {0.0: "RED     ", 1.0: "BLUE     "},
        # "natural_lang_multiclass": {
        #     0.0: "RED     ",
        #     1.0: "BLUE     ",
        #     gamma: "GREEN     ",
        #     gamma**2: "YELLOW     ",
        #     gamma + gamma**2: "PURPLE     ",
        #     1 + gamma: "ORANGE     ",
        #     1 + gamma**2: "BROWN     ",
        #     1 + gamma + gamma**2: "BLACK     ",
        # },
        "natural_lang_binary": {0.0: "<UNSAFE>", 1.0: "<SAFE>"},
        # Anything <1 is moderate. 
        "natural_lang_multiclass": {
            0.0: "<UNSAFE>",
            1.0: "<SAFE>",
            gamma: "<MODERATE>",
            gamma**2: "<MODERATE>",
            gamma + gamma**2: "<MODERATE>",
            1 + gamma: "<SAFE>",
            1 + gamma**2: "<SAFE>",
            1 + gamma + gamma**2: "<SAFE>",
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


class MultiturnDPODataset(Dataset):
    """
    Dataset class for DPO, in the multi-turn setting

    Each datapoint contains two elements:
        (chosen conversation, rejected_conversation)

    NOTE: This variation of the DPO class works from conversation in
    string/dict format
    """

    def __init__(
        self,
        chosen_conversations: List[List[Dict[str, str]]],
        rejected_conversations: List[List[Dict[str, str]]],
        tokenizer: AutoTokenizer,
        tokenizer_separator: TokenizerSeparators,
        ignore_token_id: int,
    ):
        """
        Input:
            chosen_conversations (List[List[Dict[str, str]]]):
                chosen_conversations[i] = the i-th datapoint's chosen conversation

                chosen_conversations looks like:
                [
                    [
                        {
                            "role": "user",
                            "content": user_prompt,
                        } ....
                    ], # 1st conversation
                    [
                        {
                            "role": "user",
                            "content": user_prompt,
                        } ....
                    ], # 2nd conversation
                    .... (More conversations like this)
                ]

            rejected_conversations (List[List[Dict[str, str]]]):
                rejected_conversations[i] = the i-th datapoint's rejected conversation

                Has the same format as chosen_conversations

            tokenizer (Tokenizer):
                tokenizer for the model to be trained.

            tokenizer_separator (TokenizerSeparators):
                The tokenizer separator for assistant/user special tokens.

            ignore_token_id (int):
                The token that should be ignored from loss calculations.
        """
        self.chosen_dataset = MultiturnSFTDataset(
            conversations=chosen_conversations,
            tokenizer=tokenizer,
            tokenizer_separator=tokenizer_separator,
            ignore_token_id=ignore_token_id,
        )

        self.rejected_dataset = MultiturnSFTDataset(
            conversations=rejected_conversations,
            tokenizer=tokenizer,
            tokenizer_separator=tokenizer_separator,
            ignore_token_id=ignore_token_id,
        )

        assert len(self.chosen_dataset) == len(self.rejected_dataset)
        self.length = len(self.chosen_dataset)

    def __len__(self) -> int:
        """
        Returns the number of datapoints in the dataset.

        Input:
            None

        Output:
            length: Number of datapoints in the dataset
        """
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns the i-th datapoint in a particular format.

        Input:
            idx (int):
                The index of the datapoint that needs to be returned.

        Output:
            datapoint_dict (Dict[str, torch.Tensor]):
                The i-th datapoint in the following dictionary format:
                    {
                        "chosen_input_ids": input_id (tensor),
                        "chosen_labels": label (tensor),
                        "chosen_attention_mask": attention_mask (tensor),
                        "rejected_input_ids": input_id (tensor),
                        "rejected_labels": label (tensor),
                        "rejected_attention_mask": attention_mask (tensor),
                    }

                    i.e., the input_ids, labels and attention_mask for
                    the i-th chosen conversation, and the i-th rejected
                    conversation
        """
        chosen_datapoint = self.chosen_dataset[idx]
        rejected_datapoint = self.rejected_dataset[idx]

        return {
            "chosen_input_ids": chosen_datapoint["input_ids"],
            "chosen_labels": chosen_datapoint["labels"],
            "chosen_attention_mask": chosen_datapoint["attention_mask"],
            "rejected_input_ids": rejected_datapoint["input_ids"],
            "rejected_labels": rejected_datapoint["labels"],
            "rejected_attention_mask": rejected_datapoint["attention_mask"],
        }


class MultiturnDPODatasetFromTensors(Dataset):
    """
    A variation of the DPO dataset, that uses precomputed tensors,
    specially precomputed log_probs, saved in ".pt" format.

    This is unlike MultiturnDPODataset, that uses conversations in
    dict/str format.
    """

    def __init__(
        self,
        dataset_path: str,
    ):
        """
        Input:
            dataset_path (str):
                The path to the dataset (in ".pt" format)
                that contains the dataset
        """
        assert dataset_path.endswith(".pt")
        data = torch.load(dataset_path)

        self.chosen_input_ids = data["chosen_input_ids"]
        self.chosen_labels = data["chosen_labels"]
        self.chosen_attention_mask = data["chosen_attention_mask"]
        self.chosen_ref_log_probs = data["chosen_log_probs"]

        self.rejected_input_ids = data["rejected_input_ids"]
        self.rejected_labels = data["rejected_labels"]
        self.rejected_attention_mask = data["rejected_attention_mask"]
        self.rejected_ref_log_probs = data["rejected_log_probs"]

        self._validate_tensor_shapes()
        self._print_dataset_information()

    def _validate_tensor_shapes(self) -> None:
        """
        Performs various checks on the loaded tensors,
        to make sure they have the correct shapes.

        Input:
            None

        Output:
            None
        """
        num_data = self.chosen_input_ids.shape[0]
        assert (
            self.chosen_labels.shape[0] == num_data
            and self.chosen_attention_mask.shape[0] == num_data
            and self.chosen_ref_log_probs.shape[0] == num_data
            and self.rejected_input_ids.shape[0] == num_data
            and self.rejected_labels.shape[0] == num_data
            and self.rejected_ref_log_probs.shape[0] == num_data
        )

        chosen_sequence_length = self.chosen_input_ids.shape[1]
        assert (
            self.chosen_attention_mask.shape[1] == chosen_sequence_length
            and self.chosen_labels.shape[1] == chosen_sequence_length
            and self.chosen_ref_log_probs.shape[1] == chosen_sequence_length - 1
        )

        rejected_sequence_length = self.rejected_input_ids.shape[1]
        assert (
            self.rejected_attention_mask.shape[1] == rejected_sequence_length
            and self.rejected_labels.shape[1] == rejected_sequence_length
            and self.rejected_ref_log_probs.shape[1] == rejected_sequence_length - 1
        )

    def _print_dataset_information(self) -> None:
        """
        Prints the number of datapoints, sequence length of chosen trajectories
        and rejected trajectories.

        Input:
            None

        Output:
            None
        """
        num_data = self.chosen_input_ids.shape[0]
        chosen_sequence_length = self.chosen_input_ids.shape[1]
        rejected_sequence_length = self.rejected_input_ids.shape[1]

        print("\n Dataset information:")
        print("Num data: ", num_data)
        print("Chosen sequence length: ", chosen_sequence_length)
        print("Rejected sequence length: ", rejected_sequence_length, "\n")

    def __len__(self) -> int:
        """
        Returns the number of datapoints in the dataset.

        Input:
            None

        Output:
            length: Number of datapoints in the dataset
        """
        return self.chosen_input_ids.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns the i-th datapoint in a particular format.

        Input:
            idx (int):
                The index of the datapoint that needs to be returned.

        Output:
            datapoint_dict (Dict[str, torch.Tensor]):
                The i-th datapoint in the following dictionary format:
                    {
                        "input_ids": chosen_input_id (tensor),
                        "labels": chosen_label (tensor),
                        "attention_mask": chosen_attention_mask (tensor),
                        "ref_log_probs": chosen_ref_log_probs (tensor),
                        "rejected_input_ids": rejected_input_id (tensor),
                        "rejected_labels": rejected_label (tensor),
                        "rejected_attention_mask": rejected_attention_mask (tensor),
                        "rejected_ref_log_probs": rejected_ref_log_probs (tensor),
                    }

                    i.e., the input_ids, labels, attention_mask and precomputed ref log probs
                    for the i-th chosen conversation, and the i-th rejected
                    conversation
        """
        return {
            "input_ids": self.chosen_input_ids[idx],
            "labels": self.chosen_labels[idx],
            "attention_mask": self.chosen_attention_mask[idx],
            "ref_log_probs": self.chosen_ref_log_probs[idx],
            "rejected_input_ids": self.rejected_input_ids[idx],
            "rejected_labels": self.rejected_labels[idx],
            "rejected_attention_mask": self.rejected_attention_mask[idx],
            "rejected_ref_log_probs": self.rejected_ref_log_probs[idx],
        }