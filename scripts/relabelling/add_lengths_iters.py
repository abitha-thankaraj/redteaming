import json
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoConfig, AutoTokenizer
from redteam.utils.data_utils import write_json, read_json


def load_tokenizers():
    tokenizers = {}
    for model_name in [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ]:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            padding_side="left",
            use_fast=False,
        )
        tokenizers[model_name] = tokenizer

    return tokenizers


def main(filename, savefile):
    tokenizers = load_tokenizers()
    data = read_json(fname=filename)

    for record in data:
        for conversation_type in ["positives", "negatives"]:
            for datapoint in record[conversation_type]:
                new_convs = deepcopy(datapoint["conversation"])

                for i in range(len(datapoint["conversation"])):
                    new_convs[i]["content"] = datapoint["conversation"][i]["content"].strip(
                        " \t\n\r"
                    )

                for model_name in tokenizers:
                    tokenizer = tokenizers[model_name]
                    tokenized = tokenizer.apply_chat_template(new_convs, tokenize=True)
                    length = len(tokenized)
                    datapoint[model_name.split("/")[-1] + "_length"] = length

                datapoint["conversation"] = new_convs

    write_json(data, savefile)


def main_flat(filename, savefile):
    tokenizers = load_tokenizers()
    data = read_json(fname=filename)
    save_data = []

    for record in tqdm(data):
        # for conversation_type in ["positives", "negatives"]:
        # for datapoint in record[conversation_type]:
        # new_convs = deepcopy(record["conversation"])
        goal = record["game"][0]["content"]
        new_convs = deepcopy(record["game"][1:])


        for i in range(len(new_convs)):
            if new_convs[i]["role"] == "attacker":
                new_convs[i]["role"] = "user"
            elif new_convs[i]["role"] == "defender":
                new_convs[i]["role"] = "assistant"
            new_convs[i]["content"] = new_convs[i]["content"].strip(" \t\n\r")

        new_record = {
            "goal": goal,
            "conversation": new_convs,
            "rewards": record["judge"]["rewards"],
            "category": record["judge"]["categories"],
            "generation_model": "Meta-Llama-3.1-8B-Instruct"
        }
        for model_name in tokenizers:
            tokenizer = tokenizers[model_name]
            tokenized = tokenizer.apply_chat_template(new_convs, tokenize=True)
            length = len(tokenized)
            new_record[model_name.split("/")[-1] + "_length"] = length
        
        save_data.append(new_record)
    
    write_json(save_data, savefile)


if __name__ == "__main__":

    DATA_DIR = (
        "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined"
    )
    # main(
    #     filename=f"{DATA_DIR}/combined_eval_data_llama_rewards_paired.json",
    #     savefile=f"{DATA_DIR}/combined_eval_data_llama_rewards_paired_length_added.json",
    # )
    DATA_DIR = "/scratch/bcgv/datasets/redteaming/gen_judge_multiturn_conversation_combined"

    fnames = ["combined_rwr_trained_attacker.rwr_defender.iter1.json", "combined_sft_trained_attacker.sft_defender.iter1.json", "combined_rwr_trained_attacker.untrained_defender.iter0.json", "combined_untrained_defender.sft_trained_attacker.iter0.json"]
    for fname in fnames:
        print(f"Processing {fname}")
        main_flat(
            filename=f"{DATA_DIR}/iterative/{fname}",
            savefile=f"{DATA_DIR}/iterative/combined/{fname.split('.')[0]}_llama_rewards_flat_length_added.json"
        )
    # main_flat(
    #     filename=f"/scratch/bcgv/datasets/redteaming/gen_judge_multiturn_conversation_combined/iterative/combined_rwr_trained_attacker.rwr_defender.iter1.json",
    #     savefile=f"/scratch/bcgv/datasets/redteaming/gen_judge_multiturn_conversation_combined/iterative/combined/combined_rwr_trained_attacker.rwr_defender.iter1_llama_rewards_flat_length_added.json",
    # )
