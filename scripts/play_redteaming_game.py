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
from hydra.core.hydra_config import HydraConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

    logger.info(OmegaConf.to_yaml(config))
    logger.info(f"Logging to {HydraConfig.get().run.dir}")

    attacker = get_lm_agent(
        ChatCompletionConfig(**config.attacker.chat_completion_config), "attacker"
    )
    defender = get_lm_agent(
        ChatCompletionConfig(**config.defender.chat_completion_config), "defender"
    )
    judge = get_lm_agent(ChatCompletionConfig(**config.judge.chat_completion_config), "judge")

    goals = get_dataset(**config.dataset_configs)["prompt"]

    save_config = OmegaConf.to_yaml(config)
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(save_config)

    redteaming_game = RedteamGame(
        config.seed, attacker, defender, judge, goals, config.max_turns
    )

    redteaming_game.reset()

    trajs = []
    errors = []
    for goal in tqdm(goals):
        try:
            redteaming_game.reset(goal=goal)
            if config.no_judge:
                traj = redteaming_game.simulate(judge=False)
                trajs.append({"game": traj, "judge": None})
            else:
                traj, judgement = redteaming_game.simulate(judge=True)
                trajs.append({"game": traj, "judge": judgement})
        except Exception as e:
            logger.error(f"Error in simulating game for goal: {goal}")
            logger.error(e)
            errors.append({"goal": goal, "error": str(e)})
            slack_notification(
                f"Error in simulating game for goal: {goal}, config: {global_config}"
            )
        # valid_judge_response = False
        # while not valid_judge_response:
        #     response = chat_completion.multiturn_chat_completion(
        #         system_prompt=system_prompt, messages=messages
        #     )
        #     llm_judge_trace = parse_llm_judge_evaluation(
        #         chat_completion_dict=response[-1],
        #     )
        #     valid_judge_response = is_valid_llm_judge_trace(
        #         llm_judge_trace=llm_judge_trace,
        #     )

        # output_dict = {key: llm_judge_trace[key] for key in llm_judge_trace}

    config.out_fname = os.path.join(config.out_dir, config.out_fname.replace("/", ".."))
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
