from collections import defaultdict
import torch
from redteam.inference.language_models import HuggingFaceLM
from redteam.train.dataset_utils import DPODatasetHelper
from redteam.inference.language_models import load_model_and_tokenizer
from redteam.train.common import set_seed_everywhere
from redteam.utils.slack_me import slack_notification
from hydra.core.hydra_config import HydraConfig
from redteam.constants import PARENT_DIR, SCRIPTS_CONFIG_DIR
import logging
from omegaconf import OmegaConf, DictConfig
from redteam.train.common import get_tokenizer_separators
from redteam.train.datasets import MultiturnDPODataset
from tqdm import tqdm
import hydra
import numpy as np

@torch.no_grad() 
def precompute_logits(
    dataset,
    inference_engine,
) -> defaultdict:
    """
    Precomputes the log probabilities for inputs in a
    DPO dataset, and returns them.

    Input:
        dataset (GameDPODataset):
            The DPO dataset for which we want to precompute
            log probabilities

        inference_engine (HuggingFaceLLMInferenceEngine):
            Inference engine to use, to precompute
            log probabilities

    Output:
        outputs (defaultdict):
            Should have the following format:
            {
                "chosen_input_ids": input_id (tensor),
                "chosen_labels": label (tensor),
                "chosen_attention_mask": attention_mask (tensor),
                "chosen_log_probs": chosen_log_probs (tensor),
                "rejected_input_ids": input_id (tensor),
                "rejected_labels": label (tensor),
                "rejected_attention_mask": attention_mask (tensor),
                "rejected_log_probs": rejected_log_probs (tensor),
            }
    """
    output = defaultdict(list)

    for idx in tqdm(range(len(dataset))):
        datapoint = dataset[idx]

        datapoint["chosen_log_probs"] = inference_engine.compute_log_probs(
            input_ids=datapoint["chosen_input_ids"].unsqueeze(dim=0),
            labels=datapoint["chosen_labels"].unsqueeze(dim=0),
            attention_mask=datapoint["chosen_attention_mask"].unsqueeze(dim=0),
        ).squeeze()

        datapoint["rejected_log_probs"] = inference_engine.compute_log_probs(
            input_ids=datapoint["rejected_input_ids"].unsqueeze(dim=0),
            labels=datapoint["rejected_labels"].unsqueeze(dim=0),
            attention_mask=datapoint["rejected_attention_mask"].unsqueeze(dim=0),
        ).squeeze()

        for key in datapoint:
            output[key].append(datapoint[key])

    for key in output:
        output[key] = torch.stack(output[key], dim=0)

    return output



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

global_config = None


@hydra.main(
    version_base=None,
    config_path=SCRIPTS_CONFIG_DIR,
    config_name="precompute_logits.yaml",
)
def main(config: DictConfig):
    global global_config
    set_seed_everywhere(config.seed)
    model, tokenizer = load_model_and_tokenizer(model_name=config.model_name, 
                                                model_dir=config.model_dir, 
                                                model_cache_dir=config.model_cache_dir, 
                                                device=config.device,
                                                model_max_length=config.max_length)
    tokenizer, tokenizer_separator = get_tokenizer_separators(tokenizer)

    print(config)
    ds_helper = DPODatasetHelper(
                    data_dir=config.data_dir, 
                    agent_type=config.agent_type,
                    dataset_type=config.dataset_type,
                    length_key = config.length_key, 
                    max_length=config.max_length)

    conversations = ds_helper.get_conversations()
    if config.subsample_size != -1:
        idxs = np.random.choice(len(conversations["chosen_conversations"]), config.subsample_size)

    chosen_conversations = [conversations["chosen_conversations"][i] for i in idxs]
    rejected_conversations = [conversations["rejected_conversations"][i] for i in idxs]
    print(f"Chosen conversations: {len(chosen_conversations)}")

    dataset = MultiturnDPODataset(
        chosen_conversations=chosen_conversations,
        rejected_conversations=rejected_conversations,
        tokenizer=tokenizer,
        tokenizer_separator=tokenizer_separator,
        ignore_token_id=-100    
        )

    print(f"Dataset length: {len(dataset)}")

    inference_engine = HuggingFaceLM(config.model_name, model, tokenizer)
    out = precompute_logits(dataset, inference_engine)
    torch.save(out, config.out_fname)
    print(f"Saved output to {config.out_fname}")
      
[]
if __name__ == "__main__":
    try:
        main()
        slack_notification(f"Created dataset: {global_config}")
    except Exception as e:
        logger.error(e)
        slack_notification(f"Error creating dataset: {e}, config: {global_config}")
