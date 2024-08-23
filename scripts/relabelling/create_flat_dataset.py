import argparse
from tqdm import tqdm
import pandas as pd
from redteam.utils.data_utils import read_json, write_json

def main(fname):
    save_fname = fname.replace("_paired.json", "_flat.json")

    nested_records = read_json(fname)
    flat_records = []
    
    for obj in tqdm(nested_records):
        recs = obj["positives"] + obj["negatives"]
        

        for p_conv in recs:
            flat_record = {}
            flat_record["goal"] = obj["goal"]
            flat_record["category"] = obj["category"]

            for key in ["conversation", "rewards", "generation_model"]:
                flat_record[key] = p_conv[key]
            # print(flat_record["rewards"])
            for key in ["safety_label", "reasoning", "goal_achieved"]:
                flat_record["gpt4_"+key] = p_conv[key]
            flat_records.append(flat_record)
        
        # for n_conv in obj["negatives"]:
        #     flat_record = {}
        #     flat_record["goal"] = obj["goal"]
        #     flat_record["category"] = obj["category"]
        #     for key in ["conversation", "rewards", "generation_model"]:
        #         flat_record[key] = n_conv[key]
        #     for key in ["safety_label", "reasoning", "goal_achieved"]:
        #         flat_record["gpt4_"+key] = n_conv[key]
        #     flat_records.append(flat_record)

        if len(flat_records)%20 ==0:
            write_json(flat_records, save_fname)
    write_json(flat_records, save_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, default="/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_train_data_llama_rewards_paired.json", help="Path to the json file")
    args = parser.parse_args()
    main(args.fname)