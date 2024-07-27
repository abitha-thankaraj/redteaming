#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam

export HF_HOME="/data/tir/projects/tir6/bisk/athankar/projects/.cache";
export LOGDIR="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/$(date +'%Y-%m-%d-%H-%M-%S-%3N')"

# Starting the controller
python3 -m fastchat.serve.controller &
# Starting the model worker with the specified model path
python3 -m fastchat.serve.model_worker --model-path $1 &

python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port $2

#TODO: Write a script to ensure model is loaded; something short to ping the server for that.