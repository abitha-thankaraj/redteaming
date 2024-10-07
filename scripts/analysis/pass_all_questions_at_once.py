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


# fnames = [
#     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-51-1726581110/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-50-1726581053/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-52-1726581147/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580079/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
#     "/scratch/bcgv/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580078/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json"
# ]
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# global_config = None
fnames = [
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp0/2024.09.28/18-08-1727561336/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp0.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp0.7/2024.09.28/18-11-1727561485/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp0.7.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp1.0/2024.09.28/20-12-1727568750/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp1.0.json",
]



def get_questions(fname):
    data = read_json(fname)
    all_questions = []
    for record in data:
        if sum(record["judge"]["rewards"]) > 0.0:
            questions = []
            conversation = record["game"]

            for turn in conversation:
                if turn["role"] == "goal":
                    goal = turn["content"]
                if turn["role"] == "attacker":
                    questions.append(turn["content"])
            all_questions.append((goal, " ".join(questions)))
    return all_questions


def eval_all_at_once(questions, defender, judge):
    results = []
    for (goal, q) in tqdm(questions):
        print(goal)
        conv = Conversation(messages=[("goal", goal)])
        conv.messages.append(("attacker", q))
        defender_response = defender.act(conv.to_defender_message())
        conv.messages.append(("defender", defender_response))
        judge_score = judge.score(conv.to_judge_input())
        print(judge_score)
        results.append({"goal": goal,
                        "question": q,
                        "defender_response": defender_response,
                        "judge": judge_score})
    return results


def load_config(fname):
    parent_dir = os.path.dirname(fname)
    config = OmegaConf.load(os.path.join(parent_dir, "config.yaml"))

    return config

def aggregate_results(results):
    rewards = []
    for result in results:
        rewards.append(result["judge"]["rewards"][0])
    return {
        "all_at_once_num_jailbreaks": sum(rewards),
        "all_at_once_num_evals": len(results)
    }

def main(fname):
    questions = get_questions(fname)
    config = load_config(fname)
    set_seed_everywhere(config.seed)

    config.defender.device="cuda:0"
    defender = get_policy(config.defender)
    judge = LlamaGuardJudge(device= config.defender.device)
    results = eval_all_at_once(questions, defender, judge)

    if "/scratch/bcgv/datasets/redteaming/redteaming_evals/" in config.out_dir:
        config.out_dir = config.out_dir.replace("/scratch/bcgv/datasets/redteaming/redteaming_evals/", "/data/group_data/rl/datasets/redteaming/redteaming_evals/")

    write_json(results, os.path.join(config.out_dir, "all_questions_at_once_results.json"))
    aggregated_results = aggregate_results(results)

    write_json(aggregated_results, os.path.join(config.out_dir, "all_questions_at_once_aggregated_results.json"))

if __name__ == "__main__":
    # main(fname)
    for fname in fnames:
        main(fname)
    

