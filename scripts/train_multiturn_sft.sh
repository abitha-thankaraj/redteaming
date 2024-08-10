#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam

AGENT_TYPE="attacker"
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"

HF_HOME="/data/tir/projects/tir6/bisk/athankar/projects/.cache";
RUN_NAME="multiturnsft_${AGENT_TYPE}_${MODEL_PATH}_$(date +'%Y-%m-%d-%H-%M-%S-%3N')"
LOGDIR="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/$RUN_NAME"
REPO_DIR="/data/tir/projects/tir7/user_data/athankar/redteaming"
DATA_DIR="/data/group_data/rl/datasets/redteaming"


deepspeed --master_port 32079 $REPO_DIR/redteam/train/sft.py  \
        --model_name_or_path $MODEL_PATH \
        --seed 42   \
        --data_path $DATA_DIR/gen_judge_multiturn_conversation_combined/combined_train_data.json \
        --output_dir $LOGDIR  \
        --agent_type $AGENT_TYPE \
        --train_ratio 0.99 \
        --cache_dir $HF_HOME \
        --run_name $RUN_NAME \
        --deepspeed $REPO_DIR/scripts/configs/deepspeed/zero3.json     \
        --bf16 True \
        --num_train_epochs 3  \
        --per_device_train_batch_size 1  \
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
        --model_max_length 4096   \
        --gradient_checkpointing True \
        --remove_unused_columns False \
        