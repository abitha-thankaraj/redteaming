import argparse
from tqdm import tqdm
import pandas as pd
from redteam.utils.data_utils import read_json, write_json


def main(fname):
    save_fname = fname.replace(".json", "_paired.json")

    data = read_json(fname)
    df = pd.DataFrame(data)
    goal_idxs = df.groupby("goal").indices

    dss = []

    for goal in tqdm(goal_idxs.keys(), desc="Goals"):
        ds = {"goal": goal, "positives": [], "negatives": [], "category": None}

        for idx in tqdm(goal_idxs[goal], desc="Positive/Negative"):
            d = {}
            for k in [
                "conversation",
                "rewards",
                "generation_model",
                "safety_label",
                "reasoning",
                "goal_achieved",
            ]:
                d[k] = df.iloc[idx][k]

            if sum(df.iloc[idx]["rewards"]) > 0.0:
                ds["positives"].append(d)
                for cat in df.iloc[idx]["categories"]:
                    if not (cat is None):
                        ds["category"] = cat
                        break
            else:
                ds["negatives"].append(d)
        dss.append(ds)
        # Intermediate saves
        if len(dss) % 10 == 0:
            write_json(dss, save_fname)

    write_json(dss, save_fname)
    return dss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname",
        type=str,
        default="/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_eval_data_llama_rewards.json",
        help="Path to the json file",
    )
    args = parser.parse_args()
    main(args.fname)
