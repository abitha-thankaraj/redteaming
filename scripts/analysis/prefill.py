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
from redteam.envs.common import Conversation

# # fnames = [
# #     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-51-1726581110/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
# #     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-50-1726581053/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
# #     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-52-1726581147/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
# #     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580079/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
# #     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580078/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json"
# # ]
# fnames = [
#     # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580079/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580078/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json"
#     # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-51-1726581110/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-50-1726581053/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-52-1726581147/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp0/2024.09.28/18-08-1727561336/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp0.json",
#     "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp0.7/2024.09.28/18-11-1727561485/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp0.7.json",
#     "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp1.0/2024.09.28/20-12-1727568750/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp1.0.json",
# ]

fnames = [
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_on_value_labeled_sft_all_defender_temp0/2024.09.28/21-34-1727573671/simplesafetytests_sft_trained_attacker_dpo_on_value_labeled_sft_all_defender_dpo_on_value_labeled_sft_all_defender_temp0.json",
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_on_value_labeled_sft_all_defender_temp0.7/2024.09.28/22-19-1727576399/simplesafetytests_sft_trained_attacker_dpo_on_value_labeled_sft_all_defender_dpo_on_value_labeled_sft_all_defender_temp0.7.json",
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_on_value_labeled_sft_all_defender_temp1.0/2024.09.28/22-25-1727576731/simplesafetytests_sft_trained_attacker_dpo_on_value_labeled_sft_all_defender_dpo_on_value_labeled_sft_all_defender_temp1.0.json",
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_on_value_labeled_sft_all_temp_0.0/2024.09.27/17-30-1727472611/openai_sft_trained_attacker_dpo_on_value_labeled_sft_all_defender_dpo_on_value_labeled_sft_all_temp_0.0.json",
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_on_value_labeled_sft_all_temp_0.0/2024.09.27/17-45-1727473552/jailbreakbench_sft_trained_attacker_dpo_on_value_labeled_sft_all_defender_dpo_on_value_labeled_sft_all_temp_0.0.json",
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_on_value_labeled_sft_all_temp_0.7/2024.09.27/17-59-1727474361/openai_sft_trained_attacker_dpo_on_value_labeled_sft_all_defender_dpo_on_value_labeled_sft_all_temp_0.7.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_on_value_labeled_sft_all_temp_0.7/2024.09.27/18-12-1727475129/jailbreakbench_sft_trained_attacker_dpo_on_value_labeled_sft_all_defender_dpo_on_value_labeled_sft_all_temp_0.7.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_on_value_labeled_sft_all_temp_1.0/2024.09.27/18-53-1727477639/openai_sft_trained_attacker_dpo_on_value_labeled_sft_all_defender_dpo_on_value_labeled_sft_all_temp_1.0.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/dpo_on_value_labeled_sft_all_temp_1.0/2024.09.27/19-08-1727478500/jailbreakbench_sft_trained_attacker_dpo_on_value_labeled_sft_all_defender_dpo_on_value_labeled_sft_all_temp_1.0.json",
]


def load_config(fname):
    parent_dir = os.path.dirname(fname)
    config = OmegaConf.load(os.path.join(parent_dir, "config.yaml"))
    return config


def main(fname, prefill_text="<UNSAFE>"):
    records = read_json(fname)
    config = load_config(fname)
    config.defender.device = "cuda:2"
    if "/scratch/bcgv/datasets/redteaming/redteaming_evals/" in config.out_dir:
        config.out_dir = config.out_dir.replace(
            "/scratch/bcgv/datasets/redteaming/redteaming_evals/",
            "/data/group_data/rl/datasets/redteaming/redteaming_evals/",
        )

    config.defender.model_cache_dir = "/data/tir/projects/tir6/bisk/athankar/projects/.cache"
    set_seed_everywhere(config.seed)

    defender = get_policy(config.defender)
    print("Loaded defender")
    judge = LlamaGuardJudge(config.defender.device)
    print("Loaded judge")
    redteaming_game = Game(config.seed, None, defender, judge, None, None)
    trajs, errors = [], []
    corrected_response = 0
    judge_score = None
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

            env_state, judge_score = redteaming_game.prefilled_response(
                goal, conv, flipped_rewards, prefill_text=prefill_text
            )

            trajs.append((env_state.to_game_message(), judge_score["rewards"]))
            if sum(judge_score["rewards"]) == 0.0:
                corrected_response += 1

    out_f = (
        "prefill_full_fix.json" if prefill_text == "<UNSAFE> <FIX>  " else "prefill_full.json"
    )
    out_fname = os.path.join(config.out_dir, out_f)
    out = {
        "trajs": trajs,
        "corrected_responses": corrected_response,
        "total_incorrect": len(trajs),
    }
    print(out)
    write_json(out, out_fname)
    print(f"Saved to {out_fname}")


if __name__ == "__main__":
    for fname in fnames:
        prefill_text = "<UNSAFE> <FIX>  "

        print(f"Processing {fname}")
        main(fname, prefill_text=prefill_text)
        main(fname)
        slack_notification(f"Finished processing {fname}| {prefill_text}")
