from redteam.utils.data_utils import write_json, read_json
import glob
import numpy as np


def aggregate_results(trajs):
    rewards = []
    for traj in trajs:
        rewards.append(np.array(traj["rewards"]))
    rewards = np.array(rewards)
    # TODO: Value function metrics
    return {
        "jailbreaks_per_turn": np.sum(rewards, axis=0).tolist(),
        "num_jailbreaks": int(np.sum(np.any(rewards == 1.0, axis=1))),
        "num_evals": len(trajs),
    }


if __name__ == "__main__":
    DATA_DIR = "/scratch/bcgv/datasets/redteaming/gen_judge_multiturn_conversation_combined/iterative/combined"
    fnames = glob.glob(f"{DATA_DIR}/**.json")

    for fname in fnames:
        data = read_json(fname)
        results = aggregate_results(data)
        # write_json(results, f"{DATA_DIR}/results_{fname.split('/')[-1]}")
        print(f"Processed {fname}")
        print(results)
