import os
import glob
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Dict, Any, List
from redteam.utils.data_utils import read_json
from fastchat.model.model_adapter import get_conversation_template
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from redteam.train.datasets import get_reward_to_gos


def get_conversations(data_dir: str, agent_type: str) -> List[Dict]:
    data = RLHFDatasetHelperBase.read_files(data_dir)
    # filter based on type of messages to be returned
    return filter_messages(data, agent_type)


def filter_messages(messages: List[Dict], agent_type: str) -> List[Dict]:
    """
    Filter messages based on the agent type
    """
    agent_config = {
        "attacker": (is_good_attacker_message, RLHFDatasetHelperBase.create_attacker_message),
        "defender": (is_good_defender_message, RLHFDatasetHelperBase.create_defender_message),
        "llama_attacker": (
            is_good_llama_attacker_message,
            RLHFDatasetHelperBase.create_attacker_message,
        ),
    }
    if agent_type not in agent_config:
        raise ValueError(f"Invalid agent type: {agent_type}")
    is_good_message, get_message = agent_config[agent_type]
    return list(map(get_message, filter(is_good_message, messages)))


def is_good_llama_attacker_message(message: Dict) -> bool:
    return (
        is_good_attacker_message(message)
        and message["generation_model"] == "Meta-Llama-3.1-8B-Instruct"
    )


def is_good_attacker_message(message: Dict) -> bool:
    """Attacker condition: is_unsafe == True [breaks the model] and is_goal_achieved == True [follows the goal instruction provided]"""
    return (
        message["goal_achieved"].lower().strip() == "yes"  # Follows the goal provided
        and message["safety_label"].lower().strip() == "unsafe"  # Breaks the defender model
    )


def is_good_defender_message(message: Dict) -> bool:
    """Defender condition: is_unsafe == False [does not break the model]; We do not care about the goal.
    This will be the model (a) refusing to answer or (b) providing a safe answer."""
    return message["safety_label"].lower().strip() == "safe"


class RLHFDatasetHelperBase:
    def __init__(self, data_dir: str, length_key: str = "", max_length=-1) -> None:
        self.raw_data = pd.DataFrame(RWRDatasetHelper.read_files(data_dir))
        # Convert from list to numpy array
        self.raw_data["rewards"] = self.raw_data["rewards"].apply(lambda x: np.array(x))
        if length_key != "":
            assert max_length > 0, "Max length of tokens should be greater than 0"
            self.raw_data["length_filtered"] = self.raw_data[length_key].apply(
                lambda x: x < max_length
            )

    @classmethod
    def read_files(cls, data_dir):
        if os.path.isdir(data_dir):
            data = []
            for file in sorted(glob.glob(data_dir + "/*.json")):
                data.extend(read_json(file))
        elif os.path.isfile(data_dir):
            data = read_json(data_dir)
        return data

    @classmethod
    def create_attacker_message(
        cls, raw_msg: Dict[str, Any], remove_last_defender_message=False
    ) -> List[str]:
        """"""
        # TODO: Use only raw dictionaries. Do not use the conversation object.
        conv = get_conversation_template("gpt-4")
        goal = raw_msg["goal"]
        conv.append_message(role="user", message=goal)

        for d in range(len(raw_msg["conversation"])):
            if d % 2 == 0:
                conv.append_message(
                    role="assistant",
                    message=raw_msg["conversation"][d][
                        "content"
                    ].strip(),  # remove leading and trailing spaces. -> This breaks the tokenizer masking.
                )
            else:
                conv.append_message(
                    role="user", message=raw_msg["conversation"][d]["content"].strip()
                )
        if remove_last_defender_message:
            # Remove the last defender response -> losses do not depend on it for SFT and RWR
            return conv.to_openai_api_messages()[1:-1]  # remove system prompt;
        else:
            return conv.to_openai_api_messages()[1:]

    @classmethod
    def create_defender_message(cls, raw_msg: Dict[str, Any]) -> List[str]:
        conv = get_conversation_template("gpt-4")
        for d in range(len(raw_msg["conversation"])):
            if d % 2 == 0:
                conv.append_message(
                    role="user", message=raw_msg["conversation"][d]["content"].strip()
                )
            else:
                conv.append_message(
                    role="assistant", message=raw_msg["conversation"][d]["content"].strip()
                )
        return conv.to_openai_api_messages()[1:]  # remove system prompt

    def get_conversations(self):
        raise NotImplementedError

    @staticmethod
    def is_positive_defender(rewards):
        return rewards.sum() == 0.0

    @staticmethod
    def is_positive_attacker(rewards):
        return rewards.sum() > 0.0

    @staticmethod
    def flip_rewards(rewards):
        return 1.0 - rewards

    @staticmethod
    def calculate_reward_to_gos(rewards, gamma=0.9):
        return get_reward_to_gos(rewards, gamma)


class SFTDatasetHelper(RLHFDatasetHelperBase):
    def __init__(
        self,
        data_dir: str,
        agent_type: str,
        dataset_type: str,
        length_key: str = "",
        max_length=-1,
    ) -> None:
        super().__init__(data_dir=data_dir, length_key=length_key, max_length=max_length)
        self.agent_type = agent_type
        self.dataset_type = dataset_type

        if agent_type == "defender":
            self.raw_data["defender_messages"] = self.raw_data[["conversation"]].apply(
                lambda x: RLHFDatasetHelperBase.create_defender_message(x), axis=1
            )
            # All turns are not jailbroken.
            self.raw_data["positives"] = self.raw_data["rewards"].apply(
                lambda x: x.sum() == 0.0
            )
            # Flip rewards for defender | # Defender is independent of the goal
            self.raw_data["rewards"] = self.raw_data["rewards"].apply(lambda x: 1.0 - x)

        elif agent_type == "attacker":
            self.raw_data["attacker_messages"] = self.raw_data[["goal", "conversation"]].apply(
                lambda x: RLHFDatasetHelperBase.create_attacker_message(
                    x, remove_last_defender_message=True
                ),
                axis=1,
            )
            # Anything with rewards > 0 is a positive example/ turnwise - sum works.
            self.raw_data["positives"] = self.raw_data["rewards"].apply(
                lambda x: x.sum() > 0.0
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

    def get_conversations(self):
        subsampled_indices = self._get_indices()
        return {
            "conversations": self.raw_data[f"{self.agent_type}_messages"][
                subsampled_indices
            ].to_list(),
            "rewards": None,
        }

    def _get_indices(self):
        positive_indices = np.where(self.raw_data.positives)[0]
        sampling_indices = positive_indices
        if (
            "length_filtered" in self.raw_data.keys()
        ):  # filter by length if available. #TODO: default length filtered to True
            sampling_indices = np.intersect1d(
                sampling_indices, np.where(self.raw_data.length_filtered)[0]
            )

        return sampling_indices


class RWRDatasetHelper(RLHFDatasetHelperBase):
    def __init__(
        self,
        data_dir: str,
        agent_type: str,
        dataset_type: str = "naive_balance",
        length_key="",
        max_length=-1,
    ) -> None:
        super().__init__(data_dir=data_dir, length_key=length_key, max_length=max_length)
        assert agent_type in ["attacker", "defender"], f"Invalid agent type: {agent_type}"

        self.agent_type = agent_type
        self.dataset_type = dataset_type

        if agent_type == "defender":
            self.raw_data["defender_messages"] = self.raw_data[["conversation"]].apply(
                lambda x: RLHFDatasetHelperBase.create_defender_message(x), axis=1
            )
            # All turns are not jailbroken.
            self.raw_data["positives"] = self.raw_data["rewards"].apply(
                lambda x: x.sum() == 0.0
            )
            # Flip rewards for defender | # Defender is independent of the goal
            self.raw_data["rewards"] = self.raw_data["rewards"].apply(lambda x: 1.0 - x)

        elif agent_type == "attacker":
            self.raw_data["attacker_messages"] = self.raw_data[["goal", "conversation"]].apply(
                lambda x: RLHFDatasetHelperBase.create_attacker_message(
                    x, remove_last_defender_message=True
                ),
                axis=1,
            )
            # Anything with rewards > 0 is a positive example/ turnwise - sum works.
            self.raw_data["positives"] = self.raw_data["rewards"].apply(
                lambda x: x.sum() > 0.0
            )

    def get_conversations(self):
        subsampled_indices = self._get_indices()

        return {
            "conversations": self.raw_data[f"{self.agent_type}_messages"][
                subsampled_indices
            ].to_list(),
            "rewards": self.raw_data["rewards"][subsampled_indices].to_list(),
        }

    def _get_indices(self):
        positive_indices = np.where(self.raw_data.positives)[0]
        negative_indices = np.where(~self.raw_data.positives)[0]

        if "length_filtered" in self.raw_data.keys():  # filter by length if available.
            length_filtered_indices = np.where(self.raw_data.length_filtered)[0]
            # Only pick the ones with length < max_length
            positive_indices = np.intersect1d(positive_indices, length_filtered_indices)
            negative_indices = np.intersect1d(negative_indices, length_filtered_indices)

        if self.dataset_type == "naive_balance":
            if len(positive_indices) > len(negative_indices):
                # Subsample to same length
                positive_indices = np.random.choice(positive_indices, len(negative_indices))
            else:
                negative_indices = np.random.choice(negative_indices, len(positive_indices))

            sampling_indices = np.concatenate([positive_indices, negative_indices], axis=0)
        elif self.dataset_type == "all":
            sampling_indices = np.arange(len(self.raw_data))
            if (
                "length_filtered" in self.raw_data.keys()
            ):  # filter by length if available. #TODO: default length filtered to True
                sampling_indices = np.intersect1d(
                    sampling_indices, np.where(self.raw_data.length_filtered)[0]
                )

        return sampling_indices

        # model_indices = self.raw_data.groupby("generation_model").indices

        # if self.dataset_type == "balance_by_model_llama":
        #     llama_indices = model_indices["Meta-Llama-3.1-8B-Instruct"]
        #     p_llama_indices = np.intersect1d(llama_indices, positive_indices)
        #     n_llama_indices = np.intersect1d(llama_indices, negative_indices)

    # def filter_by_model(self):
    #     model_indices =
    #     return model_indices
    # goal_wise = self.raw_data.groupby("goal").indices


class RWRDatasetValueFunctionHelper(RLHFDatasetHelperBase):
    def __init__(
        self,
        data_dir: str,
        agent_type: str,
        dataset_type: str = "naive_balance",
        length_key="",
        max_length=-1,
        # good, mid, bad weights
        dataset_type_weights=[1 / 3.0, 1 / 3.0, 1 / 3.0],
    ) -> None:
        super().__init__(data_dir=data_dir, length_key=length_key, max_length=max_length)
        assert agent_type in ["attacker", "defender"], f"Invalid agent type: {agent_type}"

        self.agent_type = agent_type
        self.dataset_type = dataset_type
        self.dataset_type_weights = dataset_type_weights
        # NOTE TO SELF: Stop writing functional programming code in a language with mutable state.
        # It is not worth it. The debugging will be a nightmare.

        if agent_type == "defender":
            # Flip rewards for defender
            self.raw_data["rewards"] = 1.0 - self.raw_data["rewards"]
            self.raw_data["defender_messages"] = self.raw_data[["conversation"]].apply(
                lambda x: RLHFDatasetHelperBase.create_defender_message(x), axis=1
            )
        elif agent_type == "attacker":
            self.raw_data["attacker_messages"] = self.raw_data[["goal", "conversation"]].apply(
                lambda x: RLHFDatasetHelperBase.create_attacker_message(
                    x, remove_last_defender_message=True
                ),
                axis=1,
            )

        # Get reward to gos; with default value for gamma =0.9
        self.raw_data["reward_to_gos"] = (
            self.raw_data["rewards"].apply(self.calculate_reward_to_gos).apply(np.array)
        )
        self.raw_data["trajectory_quality"] = self.raw_data["reward_to_gos"].apply(np.sum)

        # data = sorted(self.raw_data["trajectory_quality"].unique())
        # [0.0, 1.0, 1.9, 2.71, 2.9, 3.71, 4.609999999999999, 5.609999999999999]
        # Split into 3 groups and add a new column for categorization
        def categorize_trajectory(quality):
            if quality > 5.0:
                return "good"
            elif quality > 2.8:
                return "mid"
            else:
                return "bad"

        self.raw_data["trajectory_category"] = self.raw_data["trajectory_quality"].apply(
            categorize_trajectory
        )
        # split into 3 groups:
        self.indices = {
            "good": self.raw_data.index[self.raw_data["trajectory_quality"] > 5.0].tolist(),
            "mid": self.raw_data.index[
                (self.raw_data["trajectory_quality"] <= 5.0)
                & (self.raw_data["trajectory_quality"] > 2.8)
            ].tolist(),
            "bad": self.raw_data.index[self.raw_data["trajectory_quality"] <= 2.8].tolist(),
        }
        if "length_filtered" in self.raw_data.keys():  # filter by length if available.
            length_filtered_indices = np.where(self.raw_data.length_filtered)[0]
            for key in self.indices.keys():
                self.indices[key] = np.intersect1d(
                    np.array(self.indices[key]), length_filtered_indices
                )

    def _get_indices(self, num_samples=None):

        def is_close_to_one_decimal(a, b):
            return np.isclose(
                a, b, atol=0.05
            )  # 0.05 is half the distance between two adjacent values in the first decimal place

        if self.dataset_type == "class_balanced":
            expected_weights = [1 / 3.0, 1 / 3.0, 1 / 3.0]
            if not all(
                is_close_to_one_decimal(a, b)
                for a, b in zip(self.dataset_type_weights, expected_weights)
            ):
                print(
                    f"Warning: Using custom weights {expected_weights} for class balanced sampling"
                )
                self.dataset_type_weights = expected_weights
        elif self.dataset_type == "naive_balance":
            expected_weights = [1 / 2.0, 1 / 4.0, 1 / 4.0]
            if not all(
                is_close_to_one_decimal(a, b)
                for a, b in zip(self.dataset_type_weights, expected_weights)
            ):
                print(
                    f"Warning: Using custom weights {expected_weights} for naive balance sampling"
                )
                self.dataset_type_weights = expected_weights
        elif self.dataset_type == "weighted":
            assert is_close_to_one_decimal(
                sum(self.dataset_type_weights), 1.0
            ), "Weights should sum to 1.0"
        elif self.dataset_type == "all":
            print("Warning: Ignoring dataset type weights for all dataset")
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset_type}")

        num_good_samples = int(num_samples * self.dataset_type_weights[0])
        num_mid_samples = int(num_samples * self.dataset_type_weights[1])
        num_bad_samples = int(num_samples * self.dataset_type_weights[2])

        if self.dataset_type == "naive_balance":
            sampling_indices = np.concatenate(
                [
                    np.random.choice(self.indices["good"], num_good_samples),
                    np.random.choice(
                        self.indices["mid"] + self.indices["bad"],
                        num_mid_samples + num_bad_samples,
                    ),
                ],
                axis=0,
            )

        elif self.dataset_type == "all":
            sampling_indices = np.arange(len(self.raw_data))
        else:
            sampling_indices = np.concatenate(
                [
                    np.random.choice(self.indices["good"], num_good_samples),
                    np.random.choice(self.indices["mid"], num_mid_samples),
                    np.random.choice(self.indices["bad"], num_bad_samples),
                ],
                axis=0,
            )
        return sampling_indices

    def get_conversations(self, num_samples):
        subsampled_indices = self._get_indices(num_samples)

        return {
            "conversations": self.raw_data[f"{self.agent_type}_messages"][
                subsampled_indices
            ].to_list(),
            "rewards": self.raw_data["rewards"][subsampled_indices].to_list(),
            "trajectory_quality": self.raw_data["trajectory_quality"][
                subsampled_indices
            ].to_list(),
            "trajectory_category": self.raw_data["trajectory_category"][
                subsampled_indices
            ].to_list(),
            "reward_to_gos": self.raw_data["reward_to_gos"][subsampled_indices].to_list(),
        }
