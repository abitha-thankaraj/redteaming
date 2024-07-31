import hydra
import numpy as np
import json
import os
from omegaconf import OmegaConf, DictConfig
from typing import Set
import datetime
from tqdm import tqdm
from redteam.constants import PARENT_DIR, DATAGEN_CONFIG_DIR
from redteam.utils.data_utils import write_json, read_json

from redteam.data_generation.attack_prompts_dataset import get_dataset
from redteam.chat_completion import ChatCompletion, ChatCompletionConfig, OAIChatCompletion
from redteam.data_generation.mt_attack_prompts_parser import parse_multiturn_attack_prompt
from redteam.templates import GENERATE_MULTITURN_ATTACK_QUESTIONS_PROMPTS

@hydra.main(version_base=None, config_path=DATAGEN_CONFIG_DIR, config_name="config")
def main(config: DictConfig):
    """
    Main entry point for running question generation.
    Follows the method of this paper: Leveraging the Context through Multi-Round Interactions for Jailbreaking Attacks (https://arxiv.org/abs/2402.09177)
    """
    config.repo_dir=PARENT_DIR
    OmegaConf.resolve(config)
    np.random.seed(config.seed)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    
    chat_completion = OAIChatCompletion(ChatCompletionConfig(**config.chat_completion))
    system_prompt = GENERATE_MULTITURN_ATTACK_QUESTIONS_PROMPTS[config.chat_completion.model]["system"]
    chat_completion.set_system_prompt(system_prompt)

    user_message_template = GENERATE_MULTITURN_ATTACK_QUESTIONS_PROMPTS[config.chat_completion.model]["user_template"]

    in_context_example_prompts = read_json(config.in_context_examples_fname)["in_context_examples"]
    harmful_questions_dataset = get_dataset(**config.dataset_configs.harmbench)

    all_multiturn_attacks = []

    for i in tqdm(range(len(harmful_questions_dataset))):
        random_index = np.random.choice(a=len(in_context_example_prompts))
        in_context_example = in_context_example_prompts[random_index]
        harmful_question = harmful_questions_dataset[i]['prompt']

        user_message = user_message_template.format(in_context_examples = in_context_example, goal = harmful_question, num_turns = config.num_turns)   

        response = chat_completion.multiturn_chat_completion(system_prompt=system_prompt, messages=[user_message])
        attack_output_dict = {
            "goal": harmful_question,
            "response": response,
            "questions": parse_multiturn_attack_prompt(
                chat_completion_dict=response[-1],
                num_turns=config.num_turns,
            )
        }
        all_multiturn_attacks.append(attack_output_dict)

        if i%10 == 0:
            os.makedirs(config.out_dir, exist_ok=True)
            write_json(all_multiturn_attacks, config.save_file)
    
    write_json(all_multiturn_attacks, config.save_file)


if __name__ == "__main__":
    main()