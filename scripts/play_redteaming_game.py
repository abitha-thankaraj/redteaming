import hydra
import numpy as np
import os
from omegaconf import OmegaConf, DictConfig
from redteam.constants import PARENT_DIR
import logging
from tqdm import tqdm
from redteam.utils.logger import Logger
from redteam.common.chat_completion import ChatCompletionConfig
from redteam.envs.game import RedteamGame, get_lm_agent
from redteam.utils.data_utils import read_json, write_json
from redteam.data_generation.attack_prompts_dataset import get_dataset
from redteam.utils.slack_me import slack_notification

log = logging.getLogger(__name__)
global_config = None

@hydra.main(
    version_base=None,
    config_path="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/configs/",
    config_name="play_redteaming_game_config.yaml",
)
def main(config: DictConfig):
    global global_config

    config.repo_dir = PARENT_DIR
    OmegaConf.resolve(config)
    np.random.seed(config.seed)
    os.makedirs(config.out_dir, exist_ok=True)
    global_config = OmegaConf.to_yaml(config)
    
    log.info(OmegaConf.to_yaml(config))

    attacker = get_lm_agent(ChatCompletionConfig(**config.attacker.chat_completion_config), "attacker")
    defender = get_lm_agent(ChatCompletionConfig(**config.defender.chat_completion_config), "defender")
    judge = get_lm_agent(ChatCompletionConfig(**config.judge.chat_completion_config), "judge")

    goals = get_dataset(**config.dataset_configs)["prompt"]

    redteaming_game = RedteamGame(config.seed, 
                attacker,
                defender,
                judge,
                goals, 
                config.max_turns)
    
    redteaming_game.reset()

    trajs = []
    for goal in tqdm(goals):
        redteaming_game.reset(goal = goal)
        traj, judgement = redteaming_game.simulate(judge=True)
        trajs.append({"game": traj, "judge":judgement})
    
    os.makedirs(config.out_dir, exist_ok=True)
    config.out_fname = os.path.join(config.out_dir,config.out_fname.replace("/", ".."))
    write_json(trajs, config.out_fname)
    
    save_config = OmegaConf.to_yaml(config)
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(save_config)


if __name__ == "__main__":
    try:
        main()
        slack_notification(f"Redteaming game completed successfully: config: {global_config}")
    except Exception as e:
        log.error(e)
        slack_notification(f"Error in running redteaming game: {e}, config: {global_config}")
