import os
import numpy as np
from typing import Dict, Any
import argparse
from collections import defaultdict

from redteam.utils.data_utils import (
    read_json,
    find_all_files,
)


def load_all_arguments() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-root_dir", 
        "--root_dir", 
        type=str, 
        default='/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_chunked', 
    )

    script_arguments = vars(ap.parse_args())
    return script_arguments


def get_statistics(filename: str) -> defaultdict:
    stats = {
        "model_name": (
            "Mistral-7B-Instruct-v0.1" if "Mistral-7B-Instruct-v0.1" in filename 
            else "Llama-3.1-8B-Instruct"
        )
    }

    all_data = read_json(fname=filename)
    counter = defaultdict(int)

    for datapoint in all_data:
        key = f"goal_achived_{datapoint['goal_achieved']}_safety_label_{datapoint['safety_label']}"
        counter[key] += 1

    stats["counter"] = counter
    return stats


def collate_statistics(root_dir: str) -> None:
    collated_stats = {
        "Mistral-7B-Instruct-v0.1": {
            "stats": defaultdict(int),
        },
        "Llama-3.1-8B-Instruct": {
            "stats": defaultdict(int),
        },
    }

    all_files = find_all_files(root_dir=root_dir)
    for filename in all_files:
        stats = get_statistics(filename=filename)
        for key in stats["counter"]:
            collated_stats[stats["model_name"]]["stats"][key] += stats["counter"][key]

    print(collated_stats)


def main():
    script_arguments = load_all_arguments()
    collate_statistics(root_dir=script_arguments["root_dir"])


if __name__ == "__main__":
    main()



