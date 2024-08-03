#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate redteam

cd /data/tir/projects/tir7/user_data/athankar/redteaming/redteam/training

python multiturnm_sft.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
    --data_path /data/tir/projects/tir7/user_data/athankar/redteaming/data/dummy_conversation.json \
    --cache_dir /data/tir/projects/tir6/bisk/athankar/projects/.cache

    # --optim \
    # --model_max_length 1024 \


