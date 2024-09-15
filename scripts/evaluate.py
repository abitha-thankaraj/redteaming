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
    config_name="evaluate.yaml",
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

    logger.debug("Loading attacker")
    attacker = get_policy(config.attacker)
    logger.debug("Loading defender")
    defender = get_policy(config.defender)
    logger.debug("Loading judge")
    if config.judge.judge_type == "llama-guard":
        judge = LlamaGuardJudge(config.judge.device)
    else:
        raise NotImplementedError(f"Judge {config.judge.judge_type} not implemented")

    logger.debug("Loading dataset")
    goals = get_dataset(**config.dataset_configs)["prompt"]

    save_config = OmegaConf.to_yaml(config)
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(save_config)

    redteaming_game = Game(config.seed, attacker, defender, judge, goals, config.max_turns)

    trajs = []
    errors = []

    for goal in tqdm(goals):
        try:
            redteaming_game.reset(goal=goal)
            env_state, judge_score = redteaming_game.simulate()
            attacker_value_function_evals, defender_value_function_evals = None, None
            if config.value_function_evaluate.do_eval:
                # This function evaluates thre accuracy of the value function prediction for the attacker and defender
                if config.value_function_evaluate.attacker.evaluate:
                    attacker_value_function_evals = evaluate_value_function(
                        conversation=env_state.to_attacker_message(),
                        value_function_type=config.value_function_evaluate.attacker.value_function_type,
                        rewards=judge_score["rewards"],
                        policy_type="attacker",
                    )
                if config.value_function_evaluate.defender.evaluate:
                    defender_value_function_evals = evaluate_value_function(
                        conversation=env_state.to_defender_message(),
                        value_function_type=config.value_function_evaluate.defender.value_function_type,
                        rewards=judge_score["rewards"],
                        policy_type="defender",
                    )
            trajs.append(
                {
                    "game": env_state.to_game_message(),
                    "judge": judge_score,
                    "attacker_value_function_evals": attacker_value_function_evals,
                    "defender_value_function_evals": defender_value_function_evals,
                }
            )
            write_json(trajs, config.out_fname)
        except Exception as e:
            logger.error(f"Error in simulating game for goal: {goal}")
            logger.error(e)
            errors.append({"goal": goal, "error": str(e)})
            slack_notification(
                f"Error in simulating game for goal: {goal}, config: {global_config}"
            )

    write_json(trajs, config.out_fname)
    if len(errors) > 0:
        write_json(errors, config.out_fname.replace(".json", "_errors.json"))


if __name__ == "__main__":
    try:
        main()
        slack_notification(f"Redteaming game completed successfully: config: {global_config}")
    except Exception as e:
        logger.error(e)
        slack_notification(f"Error in running redteaming game: {e}, config: {global_config}")
