import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Dict, Any, List
from redteam.utils.data_utils import read_json
from fastchat.model.model_adapter import get_conversation_template


class MultiturnDataset(Dataset):
    def __init__(self, data_dir:str, agent_type:str):
        self.conversations = get_conversations(data_dir, agent_type)
        #TODO: Add mask logic here
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        return self.conversations[idx]

def get_conversations(data_dir: str, agent_type: str) -> List[Dict]:
    # if is directory, read all files in the directory
    if os.path.isdir(data_dir):
        data = []
        for file in sorted(glob.glob(data_dir + "/*.json")):
            data.extend(read_json(file))
    elif os.path.isfile(data_dir):
        data = read_json(data_dir)

    return filter_messages(data, agent_type)

def filter_messages(messages: List[Dict], agent_type: str) -> List[Dict]:
    """
    Filter messages based on the agent type
    """
    agent_config = {
        "attacker": (is_good_attacker_message, get_attacker_message),
        "defender": (is_good_defender_message, get_defender_message),
    }
    if agent_type not in agent_config:
        raise ValueError(f"Invalid agent type: {agent_type}")

    is_good_message, get_message = agent_config[agent_type]
    # filter messages based on the agent type + is_good_message_criteria;
    # format the filtered messages
    return list(map(get_message, filter(is_good_message, messages)))


def is_good_attacker_message(message: Dict) -> bool:
    """Attacker condition: is_unsafe == True and is_goal_achieved == True"""
    return (
        message["goal_achieved"].lower().strip() == "yes"
        and message["safety_label"].lower().strip() == "unsafe"
    )


def is_good_defender_message(message: Dict) -> bool:
    """Defender condition: is_unsafe == True and is_goal_achieved == False"""
    return (
        message["goal_achieved"].lower().strip() == "no"
        and message["safety_label"].lower().strip() == "safe"
    )


def get_attacker_message(raw_msg: Dict[str, Any]) -> List[str]:
    conv = get_conversation_template("gpt-4")
    goal = raw_msg["goal"]
    conv.append_message(role="user", message=goal)

    for d in range(len(raw_msg["conversation"])):
        if d % 2 == 0:
            conv.append_message(
                role="assistant", message=raw_msg["conversation"][d]["content"]
            )
        else:
            conv.append_message(role="user", message=raw_msg["conversation"][d]["content"])
    return conv.to_openai_api_messages()[1:]  # remove system prompt


def get_defender_message(raw_msg: Dict[str, Any]) -> List[str]:
    conv = get_conversation_template("gpt-4")
    for d in range(len(raw_msg["conversation"])):
        if d % 2 == 0:
            conv.append_message(role="user", message=raw_msg["conversation"][d]["content"])
        else:
            conv.append_message(
                role="assistant", message=raw_msg["conversation"][d]["content"]
            )
    return conv.to_openai_api_messages()[1:]  # remove system prompt


if __name__ == "__main__":
    EXAMPLE_FNAME = "/data/tir/projects/tir7/user_data/athankar/redteaming/data/gen_judge_multiturn_conversation/gpt-4_judge_generated_multiturn_conversations_22-00-1722564014.json"
    raw_data = read_json(EXAMPLE_FNAME)
    attacker_msgs = filter_messages(raw_data, "attacker")
    defender_msgs = filter_messages(raw_data, "defender")
    from IPython import embed

    embed()
