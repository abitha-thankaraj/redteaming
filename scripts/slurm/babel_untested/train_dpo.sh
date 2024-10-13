#!/bin/bash


source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
module load cuda-12.3
conda activate redteam
source /data/user_data/athankar/redteaming/scripts/slurm/env_files/.babel_env



# Command-line argument parsing
MODEL_PATH=${1:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
AGENT_TYPE=${2:-"defender"}
MASTER_PORT=${3:-29500}
DATASET_TYPE=${4:-"new_value_labeled_with_best_of_n"}
VALUE_FUNCTION_TYPE=${5:-""}
VALUE_FUNCTION_EXPERIMENT=${6:-""}
LEARNING_RATE=${7:-5e-7}
RWR_TEMPERATURE=${8:-1.0}
LENGTH_KEY=${9:-"Meta-Llama-3.1-8B-Instruct_length"}
EXPERIMENT_DESC=${10:-"dpo_on_sft_star_new_value_labeled_with_best_of_n"}

# dpo on labelled sft paired
# BASE_MODEL_PATH=${11:-"/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-27-01-48-46-440/checkpoint-44/"}

# Vanilla dpo on paired
# BASE_MODEL_PATH=${11:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}

# dpo_on_unlabeled_sft_paired
# BASE_MODEL_PATH=${11:-"/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-27-12-25-28-872/checkpoint-40/"}

# dpo_on_unlabeled_sft_all
# BASE_MODEL_PATH=${11:-"/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-27-15-06-25-868/checkpoint-45/"}


# dpo_on_sft_paired_no_fixed_trajs
# BASE_MODEL_PATH=${11:-"/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-29-11-34-39-682/checkpoint-30"}

# dpo_on_vanilla_sft_paired_by_goal_no_fixed_trajs_labelled
# BASE_MODEL_PATH=${11:-"/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-29-15-26-35-369/checkpoint-22"}
# MAX_LENGTH=4096
# RUN_NAME="multiturn_dpo_${AGENT_TYPE}_${MODEL_PATH}_$(date +'%Y-%m-%d-%H-%M-%S-%3N')"
# LOGDIR="$MODEL_PARENT_DIR/$RUN_NAME"

# new_value_labeled_with_best_of_n
BASE_MODEL_PATH=${11:-"/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-10-13-09-47-13-478/checkpoint-98"}
MAX_LENGTH=4096
RUN_NAME="multiturn_dpo_${AGENT_TYPE}_${MODEL_PATH}_$(date +'%Y-%m-%d-%H-%M-%S-%3N')"
LOGDIR="$MODEL_PARENT_DIR/$RUN_NAME"

# Run the first job
deepspeed --master_port $MASTER_PORT $REPO_DIR/redteam/train/train.py  \
        --algo "dpo" \
        --model_name_or_path $BASE_MODEL_PATH \
        --seed 42   \
        --data_path $DATA_DIR/precomputed_datasets/meta-llama/Meta-Llama-3.1-8B-Instruct_defender_precomputed_logits_combined_llama_best_of_n_new_value_labeled_deduplicated.pt \
        --eval_data_path "" \
        --agent_type $AGENT_TYPE \
        --dataset_type $DATASET_TYPE \
        --length_key $LENGTH_KEY \
        --max_length $MAX_LENGTH \
        --value_function_type "$VALUE_FUNCTION_TYPE" \
        --model_name $MODEL_PATH \
        --value_function_experiment "$VALUE_FUNCTION_EXPERIMENT" \
        --output_dir $LOGDIR  \
        --cache_dir $HF_HOME \
        --run_name $RUN_NAME \
        --exp_desc $EXPERIMENT_DESC \
        --deepspeed $REPO_DIR/scripts/configs/deepspeed/zero3.json \
        --bf16 True \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1   \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "steps" \
        --eval_steps 100000  \
        --save_strategy "steps"  \
        --save_steps 100000  \
        --save_total_limit 1 \
        --learning_rate $LEARNING_RATE \
        --weight_decay 0.0   \
        --warmup_ratio 0.04   \
        --lr_scheduler_type "cosine"   \
        --logging_steps 1     \
        --tf32 True    \
        --model_max_length $MAX_LENGTH \
        --torch_empty_cache_steps 1 \
        --gradient_checkpointing True \
        --remove_unused_columns False


# Find the latest checkpoint directory inside the output path
LATEST_CHECKPOINT=$(ls -td $LOGDIR/checkpoint-* | head -1)

for temperature in 0.0 0.7 1.0
do
        JOB_NAME="${EXPERIMENT_DESC}_temp${temperature}"
        sbatch --job-name=$JOB_NAME $REPO_DIR/scripts/slurm/babel_untested/eval.sh $temperature $LATEST_CHECKPOINT "${EXPERIMENT_DESC}_defender" "${EXPERIMENT_DESC}_temp_${temperature}" 
done
