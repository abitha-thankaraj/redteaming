import torch
import transformers

def select_even_indices(input_dict):
    output_dict = {}
    for key, tensor in input_dict.items():

        output_dict[key] = tensor[::2]
    return output_dict




if __name__ == "__main__":

    # fnames = [
    #     # "/data/group_data/rl/datasets/redteaming/best_of_n/paired/best_of_n_repaired_conversations_best_of_n_high_contrast_1000_unlabeled.pt",
    #     #     "/data/group_data/rl/datasets/redteaming/best_of_n/paired/best_of_n_repaired_conversations_hard_negatives_1000_unlabeled.pt",
    #     #     "/data/group_data/rl/datasets/redteaming/best_of_n/value_labeled/best_of_n_repaired_conversations_best_of_n_high_contrast_1000_value_labeled.pt",
    #         # "/data/group_data/rl/datasets/redteaming/best_of_n/value_labeled/best_of_n_repaired_hard_negatives_1000_value_labeled.pt"
    # ]
    
    # for fname in fnames:
    #     print(f"Processing {fname}")
    #     new_save_fname = fname.replace("best_of_n_repaired_conversations", "no_fixed_conversations")
        
    #     data = torch.load(fname)
    #     # choose all evens
    #     even_data = select_even_indices(data)   
    #     torch.save(even_data, new_save_fname)    

    
    data = torch.load("/data/group_data/rl/datasets/redteaming/best_of_n/value_labeled/combined_no_fixed_conversations_value_labeled.pt")
    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=False)
    tokenizer.pad_token_id = 128004
    tokenizer.padding_side = "left"
    from IPython import embed; embed()

    print(tokenizer.decode(data["chosen_input_ids"][0]))
    print(tokenizer.decode(data["chosen_input_ids"][1]))
