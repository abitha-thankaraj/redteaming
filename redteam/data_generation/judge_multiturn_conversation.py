import hydra
import numpy as np
import json
import os
from omegaconf import OmegaConf, DictConfig
from typing import Set
import datetime
from tqdm import tqdm

from redteam.constants import (
    PARENT_DIR, 
    DATAGEN_CONFIG_DIR,
)
from redteam.utils.data_utils import (
    write_json, 
    read_json,
)
from redteam.data_generation.templates import MULTITURN_CONVERSATION_JUDGE_PROMPTS
from redteam.common.chat_completion import (
    ChatCompletionConfig, 
    OAIChatCompletion,
)
from redteam.data_generation.parsers import (
    parse_llm_judge_evaluation,
    is_valid_llm_judge_trace,
)


@hydra.main(
    version_base=None,
    config_path=DATAGEN_CONFIG_DIR,
    config_name="judge_multiturn_conversation_config",
)
def main(config: DictConfig):
    """
    Main entry point for running question generation.
    Follows the method of this paper: Leveraging the Context through Multi-Round Interactions for Jailbreaking Attacks (https://arxiv.org/abs/2402.09177)
    """
    config.repo_dir = PARENT_DIR
    OmegaConf.resolve(config)
    np.random.seed(config.seed)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    os.makedirs(config.out_dir, exist_ok=True)

    chat_completion = OAIChatCompletion(ChatCompletionConfig(**config.chat_completion))
    system_prompt = MULTITURN_CONVERSATION_JUDGE_PROMPTS[config.chat_completion.model][
        "system"
    ]

    chat_completion.set_system_prompt(system_prompt)
    user_template = MULTITURN_CONVERSATION_JUDGE_PROMPTS[config.chat_completion.model][
        "user_template"
    ]

    multiturn_conversations = read_json(config.multiturn_conversations_fname)

    all_multiturn_judgements = []

    for i in tqdm(range(len(multiturn_conversations))):
        multiturn_conversation = multiturn_conversations[i]
        messages = [
            user_template.format(
                goal=multiturn_conversation["goal"],
                conversation=multiturn_conversation["conversation"],
            )
        ]

        valid_judge_response = False
        while not valid_judge_response:
            response = chat_completion.multiturn_chat_completion(
                system_prompt=system_prompt, messages=messages
            )
            llm_judge_trace = parse_llm_judge_evaluation(
                chat_completion_dict=response[-1],
            )
            valid_judge_response = is_valid_llm_judge_trace(
                llm_judge_trace=llm_judge_trace,
            )
        
        output_dict = {key: llm_judge_trace[key] for key in llm_judge_trace}
        output_dict["conversation_with_judge"] = response
        output_dict["conversation"] = multiturn_conversation["conversation"]
        output_dict["goal"] = multiturn_conversation["goal"]

        all_multiturn_judgements.append(output_dict)

        if i % 2 == 0:
            write_json(all_multiturn_judgements, config.save_file)

    write_json(all_multiturn_judgements, config.save_file)


if __name__ == "__main__":
    main()
