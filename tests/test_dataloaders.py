import transformers
import torch
from redteam.train.dataset_utils import read_json, RWRDatasetHelper
from redteam.train.common import get_tokenizer_separators
from redteam.train.datasets import MultiturnRWRDataset, MultiturnSFTDataset
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
CACHE_DIR = "/data/tir/projects/tir6/bisk/athankar/projects/.cache"
def get_tokenizer(model_name):
    if model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        config = transformers.AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, 
            config=config,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            model_max_length=4096,
            model_max_length=4096,
            padding_side="right",
            use_fast=False)
        return tokenizer
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        config = transformers.AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, 
                config=config,
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
                model_max_length=4096,
                model_max_length=4096,
                padding_side="right",
                use_fast=False)
        return tokenizer
    
    else:
        raise Exception("Not implemented")


def stripped_decode(tokenizer, masked_tokens):
    return tokenizer.decode(torch.where(masked_tokens == IGNORE_TOKEN_ID, tokenizer.unk_token_id, masked_tokens)).replace(tokenizer.unk_token, "")

if __name__ == "__main__":
    MODEL_NAMES = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.1"]
    # conversations = [c["conversation"] for c in read_json("dummy_conversations.json")]

    for model_name in MODEL_NAMES:
        print(f"Model: {model_name} | Outputs:")
        tokenizer, tokenizer_separators = get_tokenizer_separators(get_tokenizer(model_name))


        # sft_dataset = MultiturnSFTDataset(
        #     conversations,
        #     tokenizer,
        #     tokenizer_separators,
        #     ignore_token_id=IGNORE_TOKEN_ID,
        # )

        eval_conversation_reward_dict = RWRDatasetHelper(
        "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_eval_data_llama_rewards_flat.json",
        "attacker",
        dataset_type="naive_balance",
        ).get_conversations()

        from IPython import embed; embed()

        rwr_dataset = MultiturnRWRDataset(
            eval_conversation_reward_dict["conversations"],
            eval_conversation_reward_dict["conversations"],
            tokenizer,
            tokenizer_separators,
            ignore_token_id=IGNORE_TOKEN_ID,
            reward_per_turns=eval_conversation_reward_dict["rewards"],
            reward_per_turns=eval_conversation_reward_dict["rewards"],
            gamma=0.9,
            min_reward=0.0
        )





        # rwr_dataset = MultiturnRWRDataset(
        #     conversations,
        #     tokenizer,
        #     tokenizer_separators,
        #     ignore_token_id=IGNORE_TOKEN_ID,
        #     reward_per_turns=[[1., 2., 3.0]]*len(conversations),
        #     gamma=0.9,
        #     min_reward=0.0
        # )

        for i in range(len(rwr_dataset)):
            print(stripped_decode(tokenizer, rwr_dataset[i]["labels"]))
            print(i)
            if i==1:
                from IPython import embed; embed()
            print(i)
            if i==1:
                from IPython import embed; embed()
            assert torch.where(rwr_dataset[i]["rewards"])[0].equal(torch.where(rwr_dataset[i]["labels"]!=IGNORE_TOKEN_ID)[0]), "Only non-masked tokens should have rewards"

        print(f"Passed conversation level rewards test for {model_name}")    




    
    
