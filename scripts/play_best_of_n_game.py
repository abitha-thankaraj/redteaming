import hydra
import numpy as np
import os
from omegaconf import OmegaConf, DictConfig
from redteam.constants import PARENT_DIR
import logging
from tqdm import tqdm
from redteam.utils.logger import Logger
from redteam.common.chat_completion import ChatCompletionConfig
from redteam.envs.best_of_n_game import Attacker, Defender, BestOfNRedteamingGame
from redteam.inference.judge import LlamaGuardJudge
from redteam.utils.data_utils import write_json
from redteam.data_generation.attack_prompts_dataset import get_dataset
from redteam.utils.slack_me import slack_notification
from hydra.core.hydra_config import HydraConfig
import time
from redteam.utils.timer import get_timeelapsed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

global_config = None

@hydra.main(
    version_base=None,
    config_path="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/configs/",
    config_name="play_best_of_n_game_config.yaml",
)
def main(config: DictConfig):
    global global_config
    config.repo_dir = PARENT_DIR
    OmegaConf.resolve(config)
    np.random.seed(config.seed)
    os.makedirs(config.out_dir, exist_ok=True)
    global_config = OmegaConf.to_yaml(config)
    logger.info(OmegaConf.to_yaml(config))
    logger.info(f"Logging to {HydraConfig.get().run.dir}")

    attacker = Attacker(ChatCompletionConfig(**config.attacker.chat_completion_config))
    defender = Defender(ChatCompletionConfig(**config.defender.chat_completion_config))
    judge = LlamaGuardJudge(config.judge.device)

    goals = get_dataset(**config.dataset_configs)["prompt"]
    
    save_config = OmegaConf.to_yaml(config)
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(save_config)

    redteaming_game = BestOfNRedteamingGame(config.seed,
                attacker=attacker,
                defender=defender,
                judge= judge,
                goals = goals, 
                max_turns = config.max_turns)
    config.out_fname = os.path.join(config.out_dir,config.out_fname.replace("/", ".."))
    trajs = []
    errors = []
    for i, goal in tqdm(enumerate(goals)):
        try:
            logger.info(f"Simulating game for goal: {goal}")
            start_time = time.time()
            env_state = redteaming_game.simulate(goal=goal)
            trajs.append(env_state)
            end_time = time.time()
            logger.info(f"Goal: {goal} completed in {get_timeelapsed(start_time, end_time)}")

        except Exception as e:
            logger.error(f"Error in simulating game for goal: {goal}")
            logger.error(e)
            errors.append({"goal": goal, "error": str(e)})
            slack_notification(f"Error in simulating game for goal: {goal}, config: {global_config}")
        
        # Keep dumping the trajs to file
        write_json(trajs, config.out_fname)
        if len(errors) > 0:
            write_json(errors, config.out_fname.replace(".json", "_errors.json"))

    
    write_json(trajs, config.out_fname)
    if len(errors) > 0:
        write_json(errors, config.out_fname.replace(".json", "_errors.json"))
    

if __name__ == "__main__":
    try:
        main()
        slack_notification(f"Redteaming game completed successfully: config: {global_config}")
    except Exception as e:
        logger.error(str(e))
        slack_notification(f"Error in running redteaming game: {str(e)}, config: {global_config}")
