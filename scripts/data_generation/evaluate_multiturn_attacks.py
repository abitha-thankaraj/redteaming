import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
from typing import Set
from tqdm import tqdm
from redteam.constants import PARENT_DIR, DATAGEN_CONFIG_DIR
from redteam.utils.data_utils import write_json, read_json

from redteam.common.chat_completion import ChatCompletionConfig, OAIChatCompletion


def rename_savefile(save_file, input_file):
    input_file = input_file.split("/")[-1]
    save_file = save_file.replace(".json", f"_input_{input_file}")
    return save_file


@hydra.main(
    version_base=None, config_path=DATAGEN_CONFIG_DIR, config_name="evaluate_multiturn_config"
)
def main(config: DictConfig):
    """
    Main entry point for generating responses for the multi-turn attacks.
    
    Follows the method of this paper: 
    Leveraging the Context through Multi-Round Interactions for Jailbreaking Attacks (https://arxiv.org/abs/2402.09177)
    """
    config.repo_dir = PARENT_DIR
    OmegaConf.resolve(config)
    np.random.seed(config.seed)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    # rename so that we know what input file was used
    config.save_file = rename_savefile(
        config.save_file, config.multiturn_generated_attack_prompts_fname
    )
    print(config)

    chat_completion = OAIChatCompletion(ChatCompletionConfig(**config.chat_completion))
    # No system prompt for evals

    multiturn_generated_attacks = read_json(config.multiturn_generated_attack_prompts_fname)

    all_multiturn_attacks = []

    for i in tqdm(range(len(multiturn_generated_attacks))):
        questions = []
        for j in range(len(multiturn_generated_attacks[i]["questions"].keys())):
            questions.append(multiturn_generated_attacks[i]["questions"][f"Question {j+1}"])

        response = chat_completion.special_tokens_aware_multiturn_chat_completion(
            messages=questions,
            use_special_tokens=config.use_special_tokens,
        )

        attack_output_dict = {
            "goal": multiturn_generated_attacks[i]["goal"],
            "conversation": response["conversation"],
            "actual_conversation": response["actual_conversation"],
        }
        all_multiturn_attacks.append(attack_output_dict)

        if i % 10 == 0:
            write_json(all_multiturn_attacks, config.save_file)
    write_json(all_multiturn_attacks, config.save_file)


if __name__ == "__main__":
    main()
