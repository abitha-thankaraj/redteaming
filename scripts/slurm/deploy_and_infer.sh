#!/bin/bash

# source /home/ftajwar/.bashrc
# source /home/ftajwar/anaconda3/etc/profile.d/conda.sh
# conda activate fastchat
source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam

export HF_HOME="/data/tir/projects/tir6/bisk/athankar/projects/.cache";
export LOGDIR="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/$(date +'%Y-%m-%d-%H-%M-%S-%3N')";
# export TRANSFORMERS_CACHE=/data/user_data/ftajwar/training_cache
# export HF_HOME=/data/user_data/ftajwar/training_cache

# Starting the controller
python3 -m fastchat.serve.controller &
# Starting the model worker with the specified model path
python3 -m fastchat.serve.model_worker --model-path $1 &

python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port $2 &

sleep 300; # Sleep for 300 seconds. Wait until the model is loaded

python $3/redteam/data_generation/evaluate_multiturn_attacks.py chat_completion=$4 multiturn_generated_attack_prompts_fname=$5

# $1: model path
# $2: port
# $3: path to the script
# $4: chat_completion
# $5: dataset_type
