import os
from redteam.utils.data_utils import read_json, write_json


# fnames ={
#     "combined_mistral_best_of_n_new_value_labeled.json":
#     [
#         "/data/group_data/rl/datasets/redteaming/best_of_n/mistral/hard_negatives_mistral_best_of_n_new_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/best_of_n/mistral/high_contrast_mistral_best_of_n_new_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/hard_negatives_1000_new_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/high_contrast_1000_new_value_labeled.json"
#     ]
#     ,
    
#     "combined_mistral_no_best_of_n_new_value_labeled.json":
#     [
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/hard_negatives_1000_new_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/high_contrast_1000_new_value_labeled.json"
#     ],

#     "combined_mistral_best_of_n_unlabelled.json":
#     [
#         "/data/group_data/rl/datasets/redteaming/best_of_n/mistral/hard_negatives_mistral_best_of_n_unlabeled.json",
#         "/data/group_data/rl/datasets/redteaming/best_of_n/mistral/high_contrast_mistral_best_of_n_unlabeled.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/hard_negatives_1000.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/high_contrast_1000.json"
#     ],

#     "combined_mistral_no_best_of_n_unlabelled.json":
#     [
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/hard_negatives_1000.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/high_contrast_1000.json"
#     ],

#     "combined_mistral_best_of_n_value_labeled.json":
#     [
#         "/data/group_data/rl/datasets/redteaming/best_of_n/mistral/hard_negatives_mistral_best_of_n_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/best_of_n/mistral/high_contrast_mistral_best_of_n_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/hard_negatives_1000_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/high_contrast_1000_value_labeled.json"
#     ],
#     "combined_mistral_no_best_of_n_value_labeled.json":
#     [
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/hard_negatives_1000_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/high_contrast_1000_value_labeled.json"
#     ]
# }

fnames ={
    "combined_llama_best_of_n_new_value_labeled.json":[
        "/data/group_data/rl/datasets/redteaming/best_of_n/llama/best_of_n_repaired_conversations_best_of_n_high_contrast_1000_new_value_labeled.json",
        "/data/group_data/rl/datasets/redteaming/best_of_n/llama/best_of_n_repaired_conversations_hard_negatives_1000_new_value_labeled.json",
        "/data/group_data/rl/datasets/redteaming/quality_filtered/hard_negatives_1000_new_value_labeled.json",
        "/data/group_data/rl/datasets/redteaming/quality_filtered/high_contrast_1000_new_value_labeled.json"
    ],
    "combined_llama_no_best_of_n_new_value_labeled.json":    [
        "/data/group_data/rl/datasets/redteaming/quality_filtered/hard_negatives_1000_new_value_labeled.json",
        "/data/group_data/rl/datasets/redteaming/quality_filtered/high_contrast_1000_new_value_labeled.json"
    ],
}

# combined_fname = "/data/group_data/rl/datasets/redteaming/best_of_n/mistral/combined_mistral_best_of_n_new_value_labeled.json"

if __name__ == "__main__":
    # FOLDER = "/data/group_data/rl/datasets/redteaming/best_of_n/mistral/combined"
    FOLDER = "/data/group_data/rl/datasets/redteaming/best_of_n/llama/combined"

    for k, v in fnames.items():
        combined_data = []
        print(k)
        for fname in v:
            data = read_json(fname)
            combined_data.extend(data)
        write_json(combined_data, os.path.join(FOLDER, k))
        print(k, len(combined_data))
    # deduplicate combined data


    # write_json(combined_data, combined_fname)

