import torch

if __name__ == "__main__":
    # fnames = [
    #     "/data/group_data/rl/datasets/redteaming/best_of_n/paired/best_of_n_repaired_conversations_best_of_n_high_contrast_1000_unlabeled.pt","/data/group_data/rl/datasets/redteaming/best_of_n/paired/best_of_n_repaired_conversations_hard_negatives_1000_unlabeled.pt"
    # ]

    # combined_fname = "/data/group_data/rl/datasets/redteaming/best_of_n/paired/combined_unlabeled.pt"

    # fnames = [
    #     "/data/group_data/rl/datasets/redteaming/best_of_n/value_labeled/best_of_n_repaired_conversations_best_of_n_high_contrast_1000_value_labeled.pt",
    #     "/data/group_data/rl/datasets/redteaming/best_of_n/value_labeled/best_of_n_repaired_hard_negatives_1000_value_labeled.pt"
    # ]

    # fnames = [
    #     "/data/group_data/rl/datasets/redteaming/precomputed_datasets/meta-llama/Meta-Llama-3.1-8B-Instruct_defender_hard_negatives_precomputed_logits_1000.pt",
    #     "/data/group_data/rl/datasets/redteaming/precomputed_datasets/meta-llama/Meta-Llama-3.1-8B-Instruct_defender_high_contrast_precomputed_logits_1000.pt"
    # ]

    # combined_fname = "/data/group_data/rl/datasets/redteaming/best_of_n/value_labeled/combined_value_labeled.pt"
    # combined_fname = "/data/group_data/rl/datasets/redteaming/precomputed_datasets/meta-llama/combined_precomputed_logits.pt"

    # fnames = [
    #     "/data/group_data/rl/datasets/redteaming/best_of_n/paired/no_fixed_conversations_best_of_n_high_contrast_1000_unlabeled.pt",
    #     "/data/group_data/rl/datasets/redteaming/best_of_n/paired/no_fixed_conversations_hard_negatives_1000_unlabeled.pt"
    # ]
    # combined_fname = "/data/group_data/rl/datasets/redteaming/best_of_n/paired/combined_no_fixed_conversations_unlabeled.pt"

    fnames = [
        "/data/group_data/rl/datasets/redteaming/best_of_n/value_labeled/no_fixed_conversations_best_of_n_high_contrast_1000_value_labeled.pt",
        "/data/group_data/rl/datasets/redteaming/best_of_n/value_labeled/no_fixed_conversations_best_of_n_hard_negatives_1000_value_labeled.pt"
    ]
    
    
    combined_fname = "/data/group_data/rl/datasets/redteaming/best_of_n/value_labeled/combined_no_fixed_conversations_value_labeled.pt"
    combined_data = {}

    data = torch.load(fnames[0])
    data_2 = torch.load(fnames[1])
    for k, v in data.items():
        combined_data[k] = torch.concatenate((v, data_2[k]), dim=0)
    
    torch.save(combined_data, combined_fname)
