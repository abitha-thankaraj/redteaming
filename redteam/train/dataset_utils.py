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
import itertools
from redteam.utils.data_utils import write_json

class RLHFDatasetHelperBase:
    def __init__(self, data_dir: str, length_key: str = "", max_length=-1) -> None:
        self.raw_data = pd.DataFrame(RLHFDatasetHelperBase.read_files(data_dir))
        # Convert from list to numpy array
        self.raw_data["rewards"] = self.raw_data["rewards"].apply(lambda x: np.array(x))
        # Filter by length
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

    def get_conversations(self, num_samples=None):
        subsampled_indices = self._get_indices(num_samples)
        return {
            "conversations": self.raw_data[f"{self.agent_type}_messages"][
                subsampled_indices
            ].to_list(),
            "rewards": None,
        }

    def _get_indices(self, num_samples=None):
        
        positive_indices = np.where(self.raw_data.positives)[0]
        if num_samples is not None:
            positive_indices = np.random.choice(positive_indices, num_samples)
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
                print(f"Before filtering: {key} {len(self.indices[key])}")
                self.indices[key] = np.intersect1d(
                    np.array(self.indices[key]), length_filtered_indices
                )
                print(f"After filtering: {key} {len(self.indices[key])}")

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
                        np.concatenate([self.indices["mid"], self.indices["bad"]]),
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



class DPODatasetHelper(RWRDatasetValueFunctionHelper):
    def __init__(self, data_dir: str,
        agent_type: str,
        length_key="",
        dataset_type="naive",
        max_length=-1,
    ):
        super().__init__(data_dir=data_dir, 
                        agent_type=agent_type, 
                        dataset_type=None, 
                        length_key=length_key, 
                        max_length=max_length, 
                        dataset_type_weights=None)
        self.dataset_type = dataset_type

    def _get_indices(self, num_samples=None):
        good_indices = self.indices["good"]
        if self.dataset_type == "naive":
            not_good_indices = np.concatenate([np.array(self.indices["mid"]), np.array(self.indices["bad"])])
        elif self.dataset_type == "high_contrast":
            not_good_indices = np.array(self.indices["bad"])
        elif self.dataset_type == "hard_negatives":
            not_good_indices = np.array(self.indices["mid"])
        # pair by goal
        good_goals = set(self.raw_data.iloc[good_indices]["goal"])
        not_good_goals = set(self.raw_data.iloc[not_good_indices]["goal"])


        # get the intersection of the goals
        common_goals = good_goals.intersection(not_good_goals)

        goals = []
        chosen_responses = []
        rejected_responses = []
        
        for goal in tqdm(common_goals):
            good_chosen_idxs = np.intersect1d(self.raw_data[self.raw_data["goal"] == goal].index, good_indices)
            rejected_idxs = np.intersect1d(self.raw_data[self.raw_data["goal"] == goal].index, not_good_indices)
            
            # Use itertools.product to find all combinations
            for chosen, rejected in itertools.product(good_chosen_idxs, rejected_idxs):
                goals.append(goal)
                chosen_responses.append(chosen)
                rejected_responses.append(rejected)
                assert self.raw_data.iloc[chosen]["goal"] == goal

        return goals, chosen_responses, rejected_responses
    
    def get_conversations(self, num_samples=None):
        goals, chosen_responses, rejected_responses = self._get_indices(None)

        return {
            "chosen_conversations": self.raw_data[f"{self.agent_type}_messages"][
                chosen_responses
            ].to_list(),
            "rejected_conversations": self.raw_data[f"{self.agent_type}_messages"][
                rejected_responses
            ].to_list(),
            "chosen_rewards": self.raw_data["rewards"][chosen_responses].to_list(),
            "rejected_rewards": self.raw_data["rewards"][rejected_responses].to_list(),
            "chosen_reward_to_gos": self.raw_data["reward_to_gos"][chosen_responses].to_list(),
            "rejected_reward_to_gos": self.raw_data["reward_to_gos"][rejected_responses].to_list(),
            "chosen_trajectory_quality": self.raw_data["trajectory_quality"][
                chosen_responses
            ].to_list(),
            "rejected_trajectory_quality": self.raw_data["trajectory_quality"][
                rejected_responses
            ].to_list(),
            "goals": goals
        }



if __name__ == "__main__":

    split_type = "high_contrast"
    # split_type = "hard_negatives"

    ds = DPODatasetHelper(
                    data_dir="/data/group_data/rl/datasets/redteaming/combined", 
                    agent_type="defender",
                    length_key = "Meta-Llama-3.1-8B-Instruct_length", 
                    dataset_type=split_type,
                    max_length=4096)  

    convs = ds.get_conversations(None)

    np.random.seed(42)
    chosen_idxs = np.random.choice(len(convs["chosen_conversations"]), 1000)

    save_file = []
    for i in chosen_idxs:
        save_file.append({"goal": convs["goals"][i],
                          "chosen_conversation": convs["chosen_conversations"][i],
                          "rejected_conversation": convs["rejected_conversations"][i],
                          "chosen_reward": convs["chosen_rewards"][i].tolist(),
                          "rejected_reward": convs["rejected_rewards"][i].tolist(),
                          "chosen_reward_to_gos": convs["chosen_reward_to_gos"][i].tolist(),
                          "rejected_reward_to_gos": convs["rejected_reward_to_gos"][i].tolist(),
                          "chosen_trajectory_quality": convs["chosen_trajectory_quality"][i],
                          "rejected_trajectory_quality": convs["rejected_trajectory_quality"][i]
                          })
    write_json(save_file, f"/data/group_data/rl/datasets/redteaming/quality_filtered/{split_type}_1000.json")
    # pd.DataFrame(save_file).to_json(f"/data/group_data/rl/datasets/redteaming/quality_filtered/{split_type}_1000.json")


    # rejected_dict = {}
    # for k, v in convs.items():
    #     if k.startswith("rejected_"):
    #         rejected_dict[k.split("rejected_")[-1]]= v
    # rejected_dict["goals"]=convs["goals"]
    # pd.DataFrame.from_dict(rejected_dict).to_json(f"/data/group_data/rl/datasets/redteaming/quality_filtered/rejected_bad_{split_type}.json")
    # print("Saved rejected data")
    
    # selected_dict = {}
    # for k, v in convs.items():
    #     if k.startswith("chosen_"):
    #         selected_dict[k.split("chosen_")[-1]]= v
    # selected_dict["goals"]=convs["goals"]
    # pd.DataFrame.from_dict(selected_dict).to_json(f"/data/group_data/rl/datasets/redteaming/quality_filtered/chosen_good_{split_type}.json")

    # print("Saved chosen data")



    # print("Saving subsampled data: {}".format(len(chosen_idxs)))
    # selected_subsampled_dict = {}
    # for k, v in selected_dict.items():
    #     selected_subsampled_dict[k]= [v[i] for i in chosen_idxs]
    # pd.DataFrame.from_dict(selected_subsampled_dict).to_json(f"/data/group_data/rl/datasets/redteaming/quality_filtered/chosen_good_{split_type}_subsampled.json")

    # print("Saved selected subsampled data")

    # rejected_subsampled_dict = {}
    # for k, v in rejected_dict.items():
    #     rejected_subsampled_dict[k]=[v[i] for i in chosen_idxs]
    # pd.DataFrame.from_dict(rejected_subsampled_dict).to_json(f"/data/group_data/rl/datasets/redteaming/quality_filtered/rejected_bad_{split_type}_subsampled.json")

    # print("Saved rejected subsampled data")








    # # plot counts of chosen_reward_to_gos
    # np.array(convs["chosen_reward_to_gos"]).sum(axis=1)
    # rejected_scores = np.array(convs["rejected_reward_to_gos"]).sum(axis=1)
    # unique_values, counts = np.unique(rejected_scores, return_counts=True)

    # import matplotlib.pyplot as plt
    # # Create the bar plot
    # plt.figure(figsize=(10, 6))
    # plt.bar(str(unique_values), counts)

    # # Customize the plot
    # plt.title('Bar Plot of Unique Values and Counts')
    # plt.xlabel('rtg sums')
    # plt.ylabel('Counts')

    # # Add count labels on top of each bar
    # for i, count in enumerate(counts):
    #     plt.text(unique_values[i], count, str(count), ha='center', va='bottom')

    # plt.savefig("/data/tir/projects/tir7/user_data/athankar/redteaming/data/rejected_score_counts/barplot.png")
    # plt.close()



    
    from IPython import embed; embed()