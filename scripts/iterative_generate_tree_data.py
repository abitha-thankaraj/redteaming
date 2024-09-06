import hydra
import numpy as np
import os
from omegaconf import OmegaConf, DictConfig
from redteam.constants import PARENT_DIR
import logging
from tqdm import tqdm
from redteam.utils.logger import Logger
from redteam.common.chat_completion import ChatCompletionConfig
from redteam.envs.best_of_n_game import Attacker, Defender, BestOfNRedteamingGame, GameConversation
from redteam.inference.judge import LlamaGuardJudge
from redteam.utils.data_utils import write_json
from redteam.data_generation.attack_prompts_dataset import get_dataset
from redteam.utils.slack_me import slack_notification
from hydra.core.hydra_config import HydraConfig
from redteam.utils.data_utils import read_json, write_json
from redteam.utils.timer import get_timeelapsed
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

global_config = None

@hydra.main(
    version_base=None,
    config_path="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/configs/",
    config_name="generate_tree_data.yaml",
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
    # judge = LlamaGuardJudge(config.judge.device)

    goals = get_dataset(**config.dataset_configs)["prompt"]
    
    save_config = OmegaConf.to_yaml(config)
    with open(os.path.join(config.out_dir, f"config_chunk_{config.chunk_num}.yaml"), "w") as f:
        f.write(save_config)
    config.out_fname = os.path.join(config.out_dir, config.out_fname)

    conversations, errors = [], []
    if config.turn==0 and config.role=="attacker":
        start_time = time.time()
        for i, goal in tqdm(enumerate(goals)):
            try:
                conversation = GameConversation()
                conversation.set_system_message("")
                conversation.append_message("goal", goal)
                responses = attacker.act(conversation)
                for response in responses:
                    conversation.append_message("attacker", response)
                    conversations.append(conversation.to_game_message())
                    conversation.messages.pop(-1)
                del conversation
                write_json(conversations, config.out_fname)
                if i % 50 == 0:
                    slack_notification(f"Completed {i} goals in {get_timeelapsed(start_time, end=time.time())}")
            except Exception as e:
                errors.append({"goal": goal, "error": str(e)})
                logger.error(f"Error in creating conversation for goal: {goal}")
                logger.error(e)
                slack_notification(f"Error in creating conversation for goal: {goal}, config: {global_config}")
    write_json(conversations, config.out_fname) 
    if len(errors) > 0:
        write_json(errors, config.out_fname.replace(".json", "_errors.json"))
    
    else:
        convs = read_json(config.conversation_file)
        start_time = time.time()
        if config.role == "defender":
            for i, conv in tqdm(enumerate(convs)):
                try:
                    conversation = GameConversation("", GameConversation.parse_messages(conv))
                    response = defender.sample_response(conversation)
                    conversation.append_message("defender", response)
                    conversations.append(conversation.to_game_message())
                    del conversation
                    write_json(conversations, config.out_fname)
                    if i % 50 == 0:
                        slack_notification(f"Completed {i} goals in {get_timeelapsed(start_time, end=time.time())}")
                except Exception as e:
                    logger.error(f"Error in creating conversation for conv: {str(conv)}")
                    logger.error(str(e))
                    slack_notification(f"Error in creating conversation for goal: {conv}, error :{str(e)}")
        elif config.role =="attacker":
            for i, conv in tqdm(enumerate(convs)):
                try:
                    conversation = GameConversation("", GameConversation.parse_messages(conv))
                    responses = attacker.act(conversation)
                    for response in responses: # Attacker generates multiple responses
                        conversation.append_message("attacker", response)
                        conversations.append(conversation.to_game_message())
                        conversation.messages.pop(-1)
                    del conversation
                    write_json(conversations, config.out_fname)
                    if i % 50 == 0:
                        slack_notification(f"Completed {i} goals in {get_timeelapsed(start_time, end=time.time())}")
                except Exception as e:
                    errors.append({"goal": str(conv), "error": str(e)})
                    logger.error(f"Error in creating conversation for conv: {str(conv)}")
                    logger.error(str(e))
                    slack_notification(f"Error {str(e)}in creating conversation for goal: {str(conv)}")
        write_json(conversations, config.out_fname) 
        if len(errors) > 0:
            write_json(errors, config.out_fname.replace(".json", "_errors.json"))

    
    slack_notification(f"Completed generating tree data: config: {global_config}")
if __name__ == "__main__":
    try:
        main()
        slack_notification(f"Redteaming game completed successfully: config: {global_config}")
    except Exception as e:
        logger.error(str(e))
        slack_notification(f"Error in running redteaming game: {str(e)}, config: {global_config}")
