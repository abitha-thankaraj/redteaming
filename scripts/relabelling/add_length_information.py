import json
from copy import deepcopy
from transformers import AutoConfig, AutoTokenizer
from redteam.utils.data_utils import write_json, read_json


def load_tokenizers():
    tokenizers = {}
    for model_name in [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ]:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            padding_side="left",
            use_fast=False,
        )
        tokenizers[model_name] = tokenizer

    return tokenizers


def main(filename, savefile):
    tokenizers = load_tokenizers()
    data = read_json(fname=filename)

    for record in data:
        for conversation_type in ["positives", "negatives"]:
            for datapoint in record[conversation_type]:
                new_convs = deepcopy(datapoint['conversation'])

                for i in range(len(datapoint['conversation'])):
                    new_convs[i]["content"] = datapoint['conversation'][i]["content"].strip(" \t\n\r")
        
                for model_name in tokenizers:
                    tokenizer = tokenizers[model_name]
                    tokenized = tokenizer.apply_chat_template(new_convs, tokenize=True)
                    length = len(tokenized)
                    datapoint[model_name.split("/")[-1] + "_length"] = length

                datapoint['conversation'] = new_convs

    write_json(data, savefile)


if __name__ == "__main__":
    main(
        filename="./combined_train_data_llama_rewards_paired.json",
        savefile="./combined_train_data_llama_rewards_paired_length_added.json",
    )
