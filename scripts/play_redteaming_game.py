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

log = logging.getLogger(__name__)

@hydra.main(
    version_base=None,
    config_path=None,
    config_name="play_redteaming_game_config",
)
def main(config: DictConfig):
    config.repo_dir = PARENT_DIR
    OmegaConf.resolve(config)
    np.random.seed(config.seed)
    os.makedirs(config.out_dir, exist_ok=True)
    
    logger = Logger(log_wandb=False, simple_log = log, cfg=config)

    attacker = get_lm_agent(ChatCompletionConfig(**config.attacker.chat_completion_config), "attacker")
    defender = get_lm_agent(ChatCompletionConfig(**config.defender.chat_completion_config), "defender")
    judge = get_lm_agent(ChatCompletionConfig(**config.judge.chat_completion_config), "judge")

    goals = read_json(config.goals_fname)

    redteaming_game = RedteamGame(config.seed, 
                attacker,
                defender,
                judge,
                goals, 
                config.max_turns)
    
    redteaming_game.reset()

    trajs = []
    for _ in tqdm(range(config.num_games)):
        trajs.append(redteaming_game.simulate())


    write_json(trajs, config.out_fname)
    #Relabel rewards later






if __name__ == "__main__":
    main()