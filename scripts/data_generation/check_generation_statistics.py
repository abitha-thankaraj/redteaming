from matplotlib import pyplot as plt
from typing import Dict, Any, List
import argparse
from collections import defaultdict
import transformers
import numpy as np
import os

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

    ap.add_argument(
        "-cache_dir", 
        "--cache_dir", 
        type=str, 
        default="/home/ftajwar/training_cache",
    )

    ap.add_argument(
        "-plot_save_dir",
        "--plot_save_dir",
        type=str,
        default="/home/ftajwar/redteaming/data/length_plots"
    )

    script_arguments = vars(ap.parse_args())
    return script_arguments


def get_length(
    datapoint: Dict, 
    tokenizer: transformers.AutoTokenizer,
) -> int:
    conversation = datapoint["conversation"]
    tokenized = tokenizer.apply_chat_template(conversation, tokenize=True)
    return len(tokenized)
    

def get_statistics(
    filename: str,
    cache_dir: str,
) -> defaultdict:
    stats = {
        "model_name": (
            "mistralai/Mistral-7B-Instruct-v0.1" if "Mistral-7B-Instruct-v0.1" in filename 
            else "meta-llama/Meta-Llama-3.1-8B-Instruct"
        ),
        "length": []
    }

    tokenizer = get_tokenizer(
        model_name_or_path=stats["model_name"],
        cache_dir=cache_dir,
    )

    all_data = read_json(fname=filename)
    counter = defaultdict(int)

    for datapoint in all_data:
        key = f"goal_achived_{datapoint['goal_achieved']}_safety_label_{datapoint['safety_label']}"
        counter[key] += 1
        datapoint_length = get_length(
            datapoint=datapoint, 
            tokenizer=tokenizer,
        )
        stats["length"].append(datapoint_length)

    stats["counter"] = counter
    return stats


def get_tokenizer(
    model_name_or_path: str,
    cache_dir: str,
) -> transformers.AutoTokenizer:
    config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    return transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=cache_dir,
        padding_side="right",
        use_fast=False,
    )


def draw_histogram(
    data: List[int], 
    save_path: str,
    model_name: str,
) -> None:
    fig = plt.figure(figsize=(5,10), dpi=300)
    plt.hist(
        data,
        bins=30,
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.title(model_name)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig=fig)


def collate_statistics(
    root_dir: str,
    cache_dir: str,
    plot_save_dir: str,
) -> None:
    collated_stats = {
        "mistralai/Mistral-7B-Instruct-v0.1": {
            "stats": defaultdict(int),
            "length": [],
        },
        "meta-llama/Meta-Llama-3.1-8B-Instruct": {
            "stats": defaultdict(int),
            "length": [],
        },
    }

    all_files = find_all_files(root_dir=root_dir, file_suffix=".json")

    for filename in all_files:
        stats = get_statistics(filename=filename, cache_dir=cache_dir)
        for key in stats["counter"]:
            collated_stats[stats["model_name"]]["stats"][key] += stats["counter"][key]
        collated_stats[stats["model_name"]]["length"].extend(
            stats["length"]
        )

    for model_name in collated_stats:
        draw_histogram(
            data=collated_stats[model_name]["length"],
            model_name=model_name,
            save_path=os.path.join(plot_save_dir, model_name.split("/")[-1] + ".png")
        )
        collated_stats[model_name]["length"] = np.max(
            collated_stats[model_name]["length"]
        )
    print(collated_stats)


def main():
    script_arguments = load_all_arguments()
    collate_statistics(
        root_dir=script_arguments["root_dir"],
        cache_dir=script_arguments["cache_dir"],
        plot_save_dir=script_arguments["plot_save_dir"],
    )


if __name__ == "__main__":
    main()



