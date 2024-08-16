import os
import glob
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Dict, Any, List
from redteam.utils.data_utils import read_json
from fastchat.model.model_adapter import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_conversations(data_dir: str, agent_type: str) -> List[Dict]:
    # if is directory, read all files in the directory
    if os.path.isdir(data_dir):
        data = []
        for file in sorted(glob.glob(data_dir + "/*.json")):
            data.extend(read_json(file))
    elif os.path.isfile(data_dir):
        data = read_json(data_dir)
    # filter based on type of messages to be returned
    return filter_messages(data, agent_type)

def filter_messages(messages: List[Dict], agent_type: str) -> List[Dict]:
    """
    Filter messages based on the agent type
    """
    agent_config = {
        "attacker": (is_good_attacker_message, get_attacker_message),
        "defender": (is_good_defender_message, get_defender_message),
        "llama_attacker": (is_good_llama_attacker_message, get_attacker_message),
        # "all": (pass_through, lambda x: x),
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


def get_attacker_message(raw_msg: Dict[str, Any]) -> List[str]:
    # TODO: Use only raw dictionaries. Do not use the conversation object.
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




##############################################Reward Functions##############################################
def get_conversation_rewards(conversations, reward_type):
    pass