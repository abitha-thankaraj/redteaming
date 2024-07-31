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

from redteam.common.chat_completion import ChatCompletionConfig, OAIChatCompletion


@hydra.main(
    version_base=None, config_path=DATAGEN_CONFIG_DIR, config_name="evaluate_multiturn_config"
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
    # No system prompt for evals

    multiturn_generated_attacks = read_json(config.multiturn_generated_attack_prompts_fname)

    all_multiturn_attacks = []

    for i in tqdm(range(len(multiturn_generated_attacks))):
        questions = []
        for j in range(len(multiturn_generated_attacks[i]["questions"].keys())):
            questions.append(multiturn_generated_attacks[i]["questions"][f"Question {j+1}"])

        response = chat_completion.multiturn_chat_completion(messages=questions)

        attack_output_dict = {
            "goal": multiturn_generated_attacks[i]["goal"],
            "conversation": response,
        }
        all_multiturn_attacks.append(attack_output_dict)

        if i % 2 == 0:
            write_json(all_multiturn_attacks, config.save_file)

    write_json(all_multiturn_attacks, config.save_file)


if __name__ == "__main__":
    main()
