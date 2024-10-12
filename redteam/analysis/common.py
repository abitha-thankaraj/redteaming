import os
from omegaconf import OmegaConf
from redteam.train.common import set_seed_everywhere
from redteam.envs.common import get_policy
from redteam.utils.data_utils import read_json
from redteam.inference.language_models import GPT


# Folder structure:
#
# eval_folder/
# ├── config.yaml
# ├── <dataset>_**.json
# └── results.json


def load_eval_and_config(fname):
    evals = read_json(fname)
    parent_dir = os.path.dirname(fname)
    config = OmegaConf.load(os.path.join(parent_dir, "config.yaml"))
    # Moved models from delta to babel
    if "/scratch/bcgv/datasets/redteaming/redteaming_evals/" in config.out_dir:
        config.out_dir = config.out_dir.replace(
            "/scratch/bcgv/datasets/redteaming/redteaming_evals/",
            "/data/group_data/rl/datasets/redteaming/redteaming_evals/",
        )
    # Hardcode to cuda:0
    config.defender.device = "cuda:0"
    return evals, config


def load_target_lm(config):
    set_seed_everywhere(config.seed)
    defender = get_policy(config.defender)
    return defender

# def get_gpt():
#     return GPT("gpt-3.5-turbo-0125")

# def concatenation_attack(fname):
#     pass


# def single_turn_attack(fname):
#     pass


# def summarization(fname):
#     pass


def rephrase(fname):
    pass


def best_of_n(fname):
    pass


def prefill(fname):
    pass


def self_belief(fname):
    pass
