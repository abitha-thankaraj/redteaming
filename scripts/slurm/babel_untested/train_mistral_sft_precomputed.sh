#!/bin/bash


source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
module load cuda-12.3
conda activate redteam
source /data/user_data/athankar/redteaming/scripts/slurm/env_files/.babel_env



# Command-line argument parsing
MODEL_PATH=${1:-"mistralai/Mistral-7B-Instruct-v0.1"}
AGENT_TYPE=${2:-"defender"}
MASTER_PORT=${3:-29500}
DATASET_TYPE=${4:-"sft_unlabelled_1ep"}
VALUE_FUNCTION_TYPE=${5:-""}
VALUE_FUNCTION_EXPERIMENT=${6:-""}
LEARNING_RATE=${7:-1e-6}
LENGTH_KEY=${9:-"Mistral-7B-Instruct-v0.1_length"}
EXPERIMENT_DESC=${10:-"mistral_sft_unlabelled_1ep"}

MAX_LENGTH=4096
RUN_NAME="multiturn_sft_${AGENT_TYPE}_${MODEL_PATH}_$(date +'%Y-%m-%d-%H-%M-%S-%3N')"
LOGDIR="$MODEL_PARENT_DIR/$RUN_NAME"

# DATAPATH=$DATA_DIR/precomputed_datasets/mistralai/Mistral-7B-Instruct-v0.1_defender_precomputed_logits_combined_mistral_best_of_n_unlabelled_deduplicated.pt

# DATAPATH="/data/group_data/rl/datasets/redteaming/precomputed_datasets/mistralai/Mistral-7B-Instruct-v0.1_defender_precomputed_logits_combined_mistral_best_of_n_value_labeled_deduplicated.pt"
# sft
DATAPATH=/data/group_data/rl/datasets/redteaming/precomputed_datasets/mistralai/Mistral-7B-Instruct-v0.1_combined_precomputed_logits.pt

# sft_plus_unlabelled
# /data/group_data/rl/datasets/redteaming/precomputed_datasets/mistralai/Mistral-7B-Instruct-v0.1_defender_precomputed_logits_combined_mistral_best_of_n_unlabelled_deduplicated.pt

# sft star value-labelled
# DATAPATH="/data/group_data/rl/datasets/redteaming/precomputed_datasets/mistralai/Mistral-7B-Instruct-v0.1_defender_precomputed_logits_combined_mistral_best_of_n_value_labeled_deduplicated.pt"

# sft plus value-labelled
# DATAPATH="/data/group_data/rl/datasets/redteaming/precomputed_datasets/mistralai/Mistral-7B-Instruct-v0.1_defender_precomputed_logits_combined_mistral_best_of_n_value_labeled_deduplicated.pt"

# Run the first job
deepspeed --master_port $MASTER_PORT $REPO_DIR/redteam/train/train.py  \
        --algo "sft_precomputed" \
        --model_name_or_path $MODEL_PATH \
        --seed 42   \
        --data_path $DATAPATH \
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
        --per_device_train_batch_size 2 \
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

slackme "Finished training ${EXPERIMENT_DESC}"

# Find the latest checkpoint directory inside the output path
LATEST_CHECKPOINT=$(ls -td $LOGDIR/checkpoint-* | head -1)


for temperature in 0.0 0.7 1.0
do
    sbatch $REPO_DIR/scripts/slurm/babel_untested/eval_mistral.sh $temperature $LATEST_CHECKPOINT "${EXPERIMENT_DESC}_defender" "${EXPERIMENT_DESC}_temp_${temperature}" 
done


# Launch DPO after SFT

DPO_DATASET_TYPE=$DATASET_TYPE
DPO_EXPERIMENT_DESC="mistral_dpo_on_${DPO_DATASET_TYPE}"
DPO_LEARNING_RATE=5e-7
DPO_RWR_TEMPERATURE=1.0

DPO_BASE_MODEL_PATH=

slackme "Starting DPO training on ${DPO_DATASET_TYPE}"

bash $REPO_DIR/scripts/slurm/babel_untested/train_dpo_mistral.sh \
    $MODEL_PATH \
    $AGENT_TYPE \
    $MASTER_PORT \
    $DPO_DATASET_TYPE \
    "$VALUE_FUNCTION_TYPE" \
    "$VALUE_FUNCTION_EXPERIMENT" \
    $DPO_LEARNING_RATE \
    $DPO_RWR_TEMPERATURE \
    $LENGTH_KEY \
    $DPO_EXPERIMENT_DESC \
    $LATEST_CHECKPOINT \
    $DATAPATH