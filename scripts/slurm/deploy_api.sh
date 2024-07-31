#!/bin/bash

# source /home/ftajwar/.bashrc
# source /home/ftajwar/anaconda3/etc/profile.d/conda.sh
# conda activate fastchat
source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam

export HF_HOME="/data/tir/projects/tir6/bisk/athankar/projects/.cache";
export LOGDIR="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/$(date +'%Y-%m-%d-%H-%M-%S-%3N')"
# export TRANSFORMERS_CACHE=/data/user_data/ftajwar/training_cache
# export HF_HOME=/data/user_data/ftajwar/training_cache

# Starting the controller
python3 -m fastchat.serve.controller &
# Starting the model worker with the specified model path
python3 -m fastchat.serve.model_worker --model-path mistralai/Mistral-7B-Instruct-v0.1 &

python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8003
