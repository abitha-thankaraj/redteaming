import torch
import transformers

def select_positives(input_dict):
    output_dict = {}
    for key, tensor in input_dict.items():
        if key.startswith("chosen_"):
            key = key.strip("chosen_")
            if key == "input_id":
                output_dict[key+"s"] = tensor
            elif key == "label":
                output_dict[key+"s"] = tensor
            else:
                output_dict[key] = tensor
    return output_dict

if __name__ == "__main__":

 
    
    data = torch.load("/data/group_data/rl/datasets/redteaming/best_of_n/value_labeled/combined_no_fixed_conversations_value_labeled.pt")
    sft_dataset = select_positives(data)
    torch.save(sft_dataset, "/data/group_data/rl/datasets/redteaming/best_of_n/value_labeled/sft_combined_no_fixed_conversations_value_labeled.pt")
    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=False)
    tokenizer.pad_token_id = 128004
    tokenizer.padding_side = "left"
    from IPython import embed; embed()

    print(tokenizer.decode(data["chosen_input_ids"][0]))
    print(tokenizer.decode(data["chosen_input_ids"][1]))
