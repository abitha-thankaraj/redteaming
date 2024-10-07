#!/bin/bash


# LATEST_CHECKPOINT=/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-27-01-48-46-440/checkpoint-44/
# DEFENDER_MODEL_TYPE=sft_value_labeled_defender
# EXPERIMENT_DESC=sft_eval_pre_dpo

# Done
# LATEST_CHECKPOINT=/data/group_data/rl/experiments/redteaming/multiturn_dpo_on_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-27-10-02-06-039/checkpoint-45/
# DEFENDER_MODEL_TYPE=dpo_sft_value_labeled_defender
# EXPERIMENT_DESC=dpo_sft_eval

# LATEST_CHECKPOINT=/data/group_data/rl/experiments/redteaming/multiturn_dpo_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-27-11-22-04-313/checkpoint-41/
# DEFENDER_MODEL_TYPE=vanilla_dpo_defender
# EXPERIMENT_DESC=vanilla_dpo_paired

# LATEST_CHECKPOINT=/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-27-12-25-28-872/checkpoint-40/
# DEFENDER_MODEL_TYPE=vanilla_sft_defender
# EXPERIMENT_DESC=vanilla_sft_paired

# Done
# LATEST_CHECKPOINT=meta-llama/Meta-Llama-3.1-8B-Instruct
# DEFENDER_MODEL_TYPE=untrained_defender
# EXPERIMENT_DESC=untrained_defender

# Done
# LATEST_CHECKPOINT=/data/group_data/rl/experiments/redteaming/multiturn_dpo_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-27-13-55-04-067/checkpoint-41
# DEFENDER_MODEL_TYPE=dpo_on_unlabeled_sft_paired_defender
# EXPERIMENT_DESC=dpo_on_unlabeled_sft_paired_defender

# Done
# LATEST_CHECKPOINT=/data/group_data/rl/experiments/redteaming/multiturn_dpo_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-27-16-08-19-925/checkpoint-45
# DEFENDER_MODEL_TYPE=dpo_on_value_labeled_sft_all_defender
# EXPERIMENT_DESC=dpo_on_value_labeled_sft_all_defender

# Done
# LATEST_CHECKPOINT=/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-29-11-34-39-682/checkpoint-30
# DEFENDER_MODEL_TYPE=vanilla_sft_paired_by_goal_no_fixed_trajs_defender
# EXPERIMENT_DESC=vanilla_sft_paired_by_goal_no_fixed_trajs

# # Done - sft positive data only+ value labelled
# LATEST_CHECKPOINT=/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-27-01-48-46-440/checkpoint-44/
# DEFENDER_MODEL_TYPE=sft_value_labeled_defender
# EXPERIMENT_DESC=sft_eval_pre_dpo
# Array of temperatures

# LATEST_CHECKPOINT=/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-27-12-25-28-872/checkpoint-40/
# DEFENDER_MODEL_TYPE=vanilla_sft_defender
# EXPERIMENT_DESC=vanilla_sft_paired


TEMPERATURES=(0 0.7 1.0)

# Loop through each temperature and submit a job
for TEMPERATURE in "${TEMPERATURES[@]}"
do
    JOB_NAME="${EXPERIMENT_DESC}_temp${TEMPERATURE}"
    
    # Submit the job using sbatch
    sbatch --job-name=$JOB_NAME \
           /data/user_data/athankar/redteaming/scripts/slurm/babel_untested/eval.sh $TEMPERATURE $LATEST_CHECKPOINT $DEFENDER_MODEL_TYPE "${EXPERIMENT_DESC}_temp${TEMPERATURE}"
    
    echo "Submitted job for temperature $TEMPERATURE"
done

echo "All jobs submitted"

