#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam
module load cuda-12.3

MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.1"
# "meta-llama/Meta-Llama-3.1-8B-Instruct"
AGENT_TYPE="defender"
MASTER_PORT=29500
# MODEL_PATH=$1
# AGENT_TYPE=$2
# MASTER_PORT=$3

export HF_HOME="/data/tir/projects/tir6/bisk/athankar/projects/.cache";
export HF_TOKEN="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";

HF_HOME="/data/tir/projects/tir6/bisk/athankar/projects/.cache";
RUN_NAME="multiturn_rwr_${AGENT_TYPE}_${MODEL_PATH}_$(date +'%Y-%m-%d-%H-%M-%S-%3N')";
LOGDIR="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/$RUN_NAME";
REPO_DIR="/data/tir/projects/tir7/user_data/athankar/redteaming";
DATA_DIR="/data/group_data/rl/datasets/redteaming";


deepspeed --master_port $MASTER_PORT $REPO_DIR/redteam/train/train_rwr.py  \
        --model_name_or_path $MODEL_PATH \
        --seed 42   \
        --data_path $DATA_DIR/gen_judge_multiturn_conversation_combined/combined_train_data_llama_rewards_flat.json \
        --output_dir $LOGDIR  \
        --agent_type $AGENT_TYPE \
        --train_ratio 0.99 \
        --cache_dir $HF_HOME \
        --run_name $RUN_NAME \
        --deepspeed $REPO_DIR/scripts/configs/deepspeed/zero3.json     \
        --bf16 True \
        --num_train_epochs 3  \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1   \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "steps" \
        --eval_steps 100000  \
        --save_strategy "steps"  \
        --save_steps 100000  \
        --save_total_limit 1 \
        --learning_rate 1e-5  \
        --weight_decay 0.    \
        --warmup_ratio 0.04   \
        --lr_scheduler_type "cosine"   \
        --logging_steps 1     \
        --tf32 True    \
        --model_max_length 8192   \
        --gradient_checkpointing True \
        --remove_unused_columns False \
        