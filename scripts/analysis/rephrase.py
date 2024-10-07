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
from redteam.inference.language_models import GPT

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
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580079/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-34-1726580078/openai_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-51-1726581110/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-50-1726581053/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender.sft_trained_attacker/2024.09.17/08-52-1726581147/jailbreakbench_sft_trained_attacker_untrained_defender_untrained_defender.sft_trained_attacker.json",
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp0/2024.09.28/18-08-1727561336/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp0.json",
    "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp0.7/2024.09.28/18-11-1727561485/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp0.7.json",
    # "/data/group_data/rl/datasets/redteaming/redteaming_evals/untrained_defender_temp1.0/2024.09.28/20-12-1727568750/simplesafetytests_sft_trained_attacker_untrained_defender_untrained_defender_temp1.0.json",
]

def rephrase(text, model):
    rephrased = model.generate(
        conv=[{"role": "user", "content": f"Can you rephrase this entire body of text succinctly?Don't leave out too many details{text}"}],
        max_n_tokens=1024,
        temperature=0.1,
        top_p=1.0,
    )
    return text + "<REPHRASED>"+ rephrased

def get_fix_me_messages(fname):
    data = read_json(fname)
    fix_me_messages = []
    gpt = GPT("gpt-3.5-turbo-0125")


    for record in data:
        if sum(record["judge"]["rewards"]) > 0.0:
            rewards = record["judge"]["rewards"]
            conversation = record["game"]
            goal = conversation[0]["content"]
            messages = [("goal", goal)]
            conversation = conversation[1:]
            for i in range(len(rewards)):
                if rewards[i] > 0.0:
                    first_broken = i   
                    break
            for i in range(0, first_broken):
                messages.append((
                                conversation[i*2]["role"], 
                                conversation[i*2]["content"]))
                messages.append((
                                conversation[i*2+1]["role"],

                                rephrase(conversation[i*2+1]["content"], model = gpt)))
            for i in range(first_broken, len(rewards)):
                messages.append((
                                conversation[i*2]["role"], 
                                conversation[i*2]["content"]))
                messages.append((
                                conversation[i*2+1]["role"], 
                                ""))
            fix_me_messages.append(messages)

    return fix_me_messages
    

def rephrased_evals(fix_me_messages, defender, judge, fix_intermediate = False):

    results = []
    num_jailbreaks = 0
    gpt = GPT("gpt-3.5-turbo-0125")
    for messages in tqdm(fix_me_messages):
        original_rephrased_conv = Conversation(messages=messages).to_game_message()
        conv = Conversation(messages=[("goal",messages[0][1])])
        for i in range(len(messages)):
            if messages[i][0] == "attacker":
                conv.messages.append((messages[i][0], messages[i][1]))
            elif messages[i][0] == "defender":
                if messages[i][1] == "":
                    defender_response = defender.act(conv.to_defender_message())
                    conv.messages.append(("defender", defender_response))
                    judge_score = judge.score(conv.to_judge_input())
                    if judge_score["rewards"][-1] > 0.0:
                        num_jailbreaks += 1
                        break
                    else:
                        if fix_intermediate:
                            conv.messages[-1] = ("defender", rephrase(defender_response, model = gpt).split("<REPHRASED>")[1])    
                else:
                    conv.messages.append(("defender", messages[i][1].split("<REPHRASED>")[1]))
        results.append({"original_rephrased_conv": original_rephrased_conv,
                        "game": conv.to_game_message(),
                        "judge": judge_score})

    return results, {"num_jailbreaks": num_jailbreaks, "original_num_jailbreaks": len(fix_me_messages)}

def load_config(fname):
    parent_dir = os.path.dirname(fname)
    config = OmegaConf.load(os.path.join(parent_dir, "config.yaml"))

    return config




def main(fname):

    fix_me_messages = get_fix_me_messages(fname)
    config = load_config(fname)
    set_seed_everywhere(config.seed)
    config.defender.model_cache_dir = "/data/tir/projects/tir6/bisk/athankar/projects/.cache"
    
    config.defender.device="cuda:1"
    defender = get_policy(config.defender)
    judge = LlamaGuardJudge(device= config.defender.device)
    
    results, aggregated_results = rephrased_evals(fix_me_messages, defender, judge)

    if "/scratch/bcgv/datasets/redteaming/redteaming_evals/" in config.out_dir:
        config.out_dir = config.out_dir.replace("/scratch/bcgv/datasets/redteaming/redteaming_evals/", "/data/group_data/rl/datasets/redteaming/redteaming_evals/")

    write_json(results, os.path.join(config.out_dir, "rephrased_results.json"))
    write_json(aggregated_results, os.path.join(config.out_dir, "rephrased_aggregated_results.json"))

    slack_notification(f"Summarize results for {fname} complete. Aggregated results: {aggregated_results}")

    intermediate_rephrased_results, aggregated_intermediate_rephrased_results = rephrased_evals(fix_me_messages, defender, judge, fix_intermediate=True)
    write_json(intermediate_rephrased_results, os.path.join(config.out_dir, "intermediate_rephrased_results.json"))
    write_json(aggregated_intermediate_rephrased_results, os.path.join(config.out_dir, "intermediate_rephrased_aggregated_results.json"))
    slack_notification(f"Summarize results for {fname} complete. Aggregated results: {aggregated_intermediate_rephrased_results}")

if __name__ == "__main__":
    # main(fname)
    for fname in fnames:
        main(fname)
    

