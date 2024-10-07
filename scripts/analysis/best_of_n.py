import hydra
import numpy as np
import os
from omegaconf import OmegaConf, DictConfig
from redteam.constants import PARENT_DIR, SCRIPTS_CONFIG_DIR
import logging
from tqdm import tqdm
from redteam.envs.common import get_policy
from redteam.inference.judge import LlamaGuardJudge
from redteam.utils.data_utils import read_json, write_json
from redteam.data_generation.attack_prompts_dataset import get_dataset
from redteam.utils.slack_me import slack_notification
from hydra.core.hydra_config import HydraConfig
from redteam.envs.evaluation import Game, evaluate_value_function
from redteam.train.common import set_seed_everywhere
from redteam.envs.common import GameConversation

# fnames = [
#     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-51-1726581110/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-50-1726581053/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-52-1726581147/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580079/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580078/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json"
# ]
fnames = [
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580079/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580078/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-51-1726581110/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-50-1726581053/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-52-1726581147/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp0/2024.09.28/18-08-1727561336/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp0.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp0.7/2024.09.28/18-11-1727561485/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp0.7.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp1.0/2024.09.28/20-12-1727568750/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp1.0.json",
]

def load_config(fname):
    parent_dir = os.path.dirname(fname)
    config = OmegaConf.load(os.path.join(parent_dir, "config.yaml"))
    return config


def main(fname):
    records = read_json(fname)
    config = load_config(fname)
    config.defender.device="cuda:2"
    if "/scratch/bcgv/datasets/redteaming/redteaming_evals/" in config.out_dir:
        config.out_dir = config.out_dir.replace("/scratch/bcgv/datasets/redteaming/redteaming_evals/", "/data/group_data/rl/datasets/redteaming/redteaming_evals/")
    set_seed_everywhere(config.seed)


    config.defender.model_cache_dir = "/data/tir/projects/tir6/bisk/athankar/projects/.cache"

    defender = get_policy(config.defender)
    judge = LlamaGuardJudge(config.defender.device)
    redteaming_game = Game(config.seed, None, defender, judge, None, None)
    trajs, errors = [], []

    for record in records:
        if sum(record["judge"]["rewards"]) > 0.0:
            game_conv = record["game"]
            goal = game_conv[0]["content"]
            conv = game_conv[1:]
            for i in range(len(conv)):
                if conv[i]["role"] == "attacker":
                    conv[i]["role"] = "user"
                if conv[i]["role"] == "defender":
                    conv[i]["role"] = "assistant"

            flipped_rewards = [1.0 - rew for rew in record["judge"]["rewards"]]

            # env_state, judge_rewards = redteaming_game.repair(goal, conv, record["judge"]["rewards"])
            env_state, judge_rewards = redteaming_game.repair(goal, conv, flipped_rewards)

            if judge_rewards is None:
                errors.append((goal, conv))
            else:
                trajs.append((env_state.to_game_message(), judge_rewards))
        
    out_fname = os.path.join(config.out_dir, "flipped_best_of_n.json")
    out = {
        "num_fixed": len(trajs),
        "num_errors": len(errors),
        "total": len(trajs) + len(errors),
        "trajs": trajs,
        "errors": errors
    }
    print(out)
    write_json(out, out_fname)
    print(f"Saved to {out_fname}")

            
        

if __name__ == "__main__":
    for fname in fnames:
        print(f"Processing {fname}")
        main(fname)




