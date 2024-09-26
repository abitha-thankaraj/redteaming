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


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

global_config = None



@hydra.main(
    version_base=None,
    config_path=SCRIPTS_CONFIG_DIR,
    config_name="best_of_n_generate.yaml",
)
def main(config: DictConfig):
    global global_config
    # config.repo_dir = PARENT_DIR

    OmegaConf.resolve(config)
    set_seed_everywhere(config.seed)
    config.out_fname = os.path.join(config.out_dir, config.out_fname)
    os.makedirs(config.out_dir, exist_ok=True)
    global_config = OmegaConf.to_yaml(config)

    logger.debug(OmegaConf.to_yaml(config))
    logger.debug(f"Logging to {HydraConfig.get().run.dir}")

    logger.debug("Loading defender")
    defender = get_policy(config.defender)
    logger.debug("Loading judge")
    
    if config.judge.judge_type == "llama-guard":
        judge = LlamaGuardJudge(config.judge.device)
    else:
        raise NotImplementedError(f"Judge {config.judge.judge_type} not implemented")

    logger.debug("Loading dataset")

    records = read_json(config.dataset_fname)
    
    save_config = OmegaConf.to_yaml(config)
    
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(save_config)

    redteaming_game = Game(config.seed, None, defender, judge, None, None)

    trajs, errors = [], []

    if config.start_idx !=0:
        records = records[config.start_idx:]
    from IPython import embed; embed()


    for record in tqdm(records):
        goal = record["goal"]
        conversation = record["chosen_conversation"]
        reward = record["chosen_reward"]

        rejected_conversation = record["rejected_conversation"]
        rejected_reward = record["rejected_reward"]

        try:
            env_state, judge_rewards = redteaming_game.repair(goal, rejected_conversation, rejected_reward)
            if env_state is None or judge_rewards is None:
                errors.append({
                    "goal": goal, 
                    "rejected_conversation": rejected_conversation,
                    "rejected_reward": rejected_reward, "error": "No repair found"})
                logger.info(f"No repair found for goal: {goal}")
                continue

            trajs.append(
                {
                    "goal": goal,
                    "rejected_conversation": rejected_conversation,
                    "rejected_reward": rejected_reward,
                    "chosen_conversation": conversation,
                    "chosen_reward": reward,
                    "repaired_conversation": env_state.to_defender_message(),
                    "repaired_reward": judge_rewards,
                }
            )

            write_json(trajs, config.out_fname)
            


        except Exception as e:
            logger.error(f"Error in repairing traj for: {goal}")
            logger.error(e)
            errors.append({"goal": goal, "error": str(e)})
            slack_notification(
                f"Error in simulating game for goal: {goal}, config: {global_config}"
            )



if __name__ == "__main__":
    try:
        main()
        slack_notification(f"Redteaming game completed successfully: config: {global_config}")
    except Exception as e:
        logger.error(e)
        slack_notification(f"Error in running redteaming game: {e}, config: {global_config}")
