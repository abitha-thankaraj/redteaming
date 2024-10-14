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
DATASET_TYPE=${4:-"sft_unlabelled"}
VALUE_FUNCTION_TYPE=${5:-""}
VALUE_FUNCTION_EXPERIMENT=${6:-""}
LEARNING_RATE=${7:-5e-7}
RWR_TEMPERATURE=${8:-1.0}
LENGTH_KEY=${9:-"Mistral-7B-Instruct-v0.1_length"}
EXPERIMENT_DESC=${10:-"mistral_dpo_on_sft_unlabelled"}

# new_value_labeled_with_best_of_n
BASE_MODEL_PATH=${11:-"/data/group_data/rl/experiments/redteaming/multiturn_sft_defender_mistralai/Mistral-7B-Instruct-v0.1_2024-10-13-14-15-12-257/checkpoint-30"}
DATAPATH=${12:-""}
MAX_LENGTH=4096
RUN_NAME="multiturn_dpo_${AGENT_TYPE}_${MODEL_PATH}_$(date +'%Y-%m-%d-%H-%M-%S-%3N')"
LOGDIR="$MODEL_PARENT_DIR/$RUN_NAME"

# Run the first job
deepspeed --master_port $MASTER_PORT $REPO_DIR/redteam/train/train.py  \
        --algo "dpo" \
        --model_name_or_path $BASE_MODEL_PATH \
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
        sbatch --job-name=$JOB_NAME $REPO_DIR/scripts/slurm/babel_untested/eval_mistral.sh $temperature $LATEST_CHECKPOINT "${EXPERIMENT_DESC}_defender" "${EXPERIMENT_DESC}_temp_${temperature}" 
done

slackme "Finished training $EXPERIMENT_DESC"