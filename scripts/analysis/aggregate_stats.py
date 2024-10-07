import numpy as np
import glob
from redteam.utils.data_utils import read_json, write_json
import yaml

def aggregate_results(trajs):
    rewards = []
    for traj in trajs:
        rewards.append(traj["judge"]["rewards"])
    rewards = np.array(rewards)
    return  {
        "jailbreaks_per_turn": np.sum(rewards, axis=0).tolist(),
        "num_jailbreaks": int(np.sum(np.any(rewards == 1.0, axis=1))),
        "num_evals": len(trajs),
    }

if __name__ == "__main__":
    FOLDERS = [
        # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_sft_eval_temp0/2024.09.27/11-13-1727450020",
        # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_sft_eval_temp0/2024.09.27/11-29-1727450958",
        # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_sft_eval_temp0.7/2024.09.27/11-13-1727450020",
        # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_sft_eval_temp0.7/2024.09.27/11-29-1727450951",
        # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_sft_eval_temp1.0/2024.09.27/11-13-1727450020",
        # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_sft_eval_temp1.0/2024.09.27/11-30-1727451044"
    
        "/data/group_data/rl/datasets/redteaming/redteaming_evals/sft_eval_pre_dpo_temp0/2024.09.27/11-14-1727450052",
        "/data/group_data/rl/datasets/redteaming/redteaming_evals/sft_eval_pre_dpo_temp0/2024.09.27/11-25-1727450759",
        "/data/group_data/rl/datasets/redteaming/redteaming_evals/sft_eval_pre_dpo_temp0.7/2024.09.27/11-14-1727450052",
        "/data/group_data/rl/datasets/redteaming/redteaming_evals/sft_eval_pre_dpo_temp0.7/2024.09.27/11-25-1727450700",
        "/data/group_data/rl/datasets/redteaming/redteaming_evals/sft_eval_pre_dpo_temp1.0/2024.09.27/11-14-1727450056",
        "/data/group_data/rl/datasets/redteaming/redteaming_evals/sft_eval_pre_dpo_temp1.0/2024.09.27/11-26-1727450804"
    
    ]

    for folder in FOLDERS:
        json_files = glob.glob(f"{folder}/*.json")
        assert len(json_files) ==1, f"More than one json file found in {folder}"
        trajs = read_json(json_files[0])
        # read yaml
        with open(f"{folder}/config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        results = aggregate_results(trajs)
        results["config"] = config
        write_json(results, f"{folder}/_results.json")
        

