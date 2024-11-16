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

# fnames ={
#     "combined_llama_best_of_n_new_value_labeled.json":[
#         "/data/group_data/rl/datasets/redteaming/best_of_n/llama/best_of_n_repaired_conversations_best_of_n_high_contrast_1000_new_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/best_of_n/llama/best_of_n_repaired_conversations_hard_negatives_1000_new_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/hard_negatives_1000_new_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/high_contrast_1000_new_value_labeled.json"
#     ],
#     "combined_llama_no_best_of_n_new_value_labeled.json":    [
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/hard_negatives_1000_new_value_labeled.json",
#         "/data/group_data/rl/datasets/redteaming/quality_filtered/high_contrast_1000_new_value_labeled.json"
#     ],
# }


# fnames ={
#     "combined_gemma_no_best_of_n_unlabelled.json":["/scratch/bcgv/datasets/redteaming/quality_filtered/high_contrast_1000.json", 
#                                                  "/scratch/bcgv/datasets/redteaming/quality_filtered/hard_negatives_1000.json"],
#     "combined_gemma_no_best_of_n_value_labeled.json":["/scratch/bcgv/datasets/redteaming/quality_filtered/high_contrast_1000_value_labeled.json", 
#                                                     "/scratch/bcgv/datasets/redteaming/quality_filtered/hard_negatives_1000_value_labeled.json"],
#     "combined_gemma_best_of_n_unlabelled.json":["/scratch/bcgv/datasets/redteaming/gemma_best_of_n/hard_negatives_gemma_best_of_n.json", 
#                                                 "/scratch/bcgv/datasets/redteaming/gemma_best_of_n/high_contrast_gemma_best_of_n.json",
#                                                 "/scratch/bcgv/datasets/redteaming/quality_filtered/hard_negatives_1000.json",
#                                                 "/scratch/bcgv/datasets/redteaming/quality_filtered/high_contrast_1000.json"],
#     "combined_gemma_best_of_n_value_labeled.json":["/scratch/bcgv/datasets/redteaming/gemma_best_of_n/hard_negatives_gemma_best_of_n_value_labeled.json",
#                                                     "/scratch/bcgv/datasets/redteaming/gemma_best_of_n/high_contrast_gemma_best_of_n_value_labeled.json",
#                                                     "/scratch/bcgv/datasets/redteaming/quality_filtered/hard_negatives_1000_value_labeled.json",
#                                                     "/scratch/bcgv/datasets/redteaming/quality_filtered/high_contrast_1000_value_labeled.json"]
# }
# combined_fname = "/data/group_data/rl/datasets/redteaming/best_of_n/mistral/combined_mistral_best_of_n_new_value_labeled.json"

fnames ={
    "combined_qwen_no_best_of_n_unlabelled.json":["/scratch/bcgv/datasets/redteaming/quality_filtered/high_contrast_1000.json", 
                                                 "/scratch/bcgv/datasets/redteaming/quality_filtered/hard_negatives_1000.json"],
    "combined_qwen_no_best_of_n_value_labeled.json":["/scratch/bcgv/datasets/redteaming/quality_filtered/high_contrast_1000_value_labeled.json", 
                                                    "/scratch/bcgv/datasets/redteaming/quality_filtered/hard_negatives_1000_value_labeled.json"],
    "combined_qwen_best_of_n_unlabelled.json":["/scratch/bcgv/datasets/redteaming/qwen_best_of_n/hard_negatives_qwen_best_of_n.json", 
                                                "/scratch/bcgv/datasets/redteaming/qwen_best_of_n/high_contrast_qwen_best_of_n.json",
                                                "/scratch/bcgv/datasets/redteaming/quality_filtered/hard_negatives_1000.json",
                                                "/scratch/bcgv/datasets/redteaming/quality_filtered/high_contrast_1000.json"],
    "combined_qwen_best_of_n_value_labeled.json":["/scratch/bcgv/datasets/redteaming/qwen_best_of_n/hard_negatives_qwen_best_of_n_value_labeled.json",
                                                    "/scratch/bcgv/datasets/redteaming/qwen_best_of_n/high_contrast_qwen_best_of_n_value_labeled.json",
                                                    "/scratch/bcgv/datasets/redteaming/quality_filtered/hard_negatives_1000_value_labeled.json",
                                                    "/scratch/bcgv/datasets/redteaming/quality_filtered/high_contrast_1000_value_labeled.json"]
}

def get_sft_plus_pairs(data):
    sft_pairs = []
    for d in data:
        if "repaired_conversation" not in d.keys():
            raise Exception("repaired_conversaion not in keys")
        d["chosen_conversation"]=d.pop("repaired_conversation")
        d["chosen_reward"] = d.pop("repaired_reward")
        sft_pairs.append(d)
    return sft_pairs
        
def dedup(data):
    deduped = []
    for d in data:
        if sum(d["chosen_reward"])==3:
            continue
        else:
            deduped.append(d)
    return deduped

if __name__ == "__main__":
    # FOLDER = "/data/group_data/rl/datasets/redteaming/best_of_n/mistral/combined"
    # FOLDER = "/data/group_data/rl/datasets/redteaming/best_of_n/llama/combined"
    # FOLDER = "/scratch/bcgv/datasets/redteaming/best_of_n/gemma/combined"
    FOLDER = "/scratch/bcgv/datasets/redteaming/best_of_n/qwen/combined"

    os.path.exists(FOLDER) or os.makedirs(FOLDER)

    for k, v in fnames.items():
        combined_data = []
        print(k)
        for fname in v:
            data = read_json(fname)
            if "repaired_conversation" in data[0].keys():
                data = get_sft_plus_pairs(data)
            if sum(data[1]["chosen_reward"]) != 3:
                data = dedup(data)
            combined_data.extend(data)
        write_json(combined_data, os.path.join(FOLDER, k))
        print(k, len(combined_data))
    # deduplicate combined data


    # write_json(combined_data, combined_fname)

