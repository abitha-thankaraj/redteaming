#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam

export HF_HOME="/data/tir/projects/tir6/bisk/athankar/projects/.cache";
export LOGDIR="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/$(date +'%Y-%m-%d-%H-%M-%S-%3N')"

# Starting the controller
python3 -m fastchat.serve.controller &
# Starting the model worker with the specified model path
python3 -m fastchat.serve.model_worker --model-path mistralai/Mistral-7B-Instruct-v0.1 &
# Add model-list-mode because you're running the model-worker in the background and model loading takes time and does not 
python3 -m fastchat.serve.gradio_web_server  --host 0.0.0.0 --port 8001 --model-list-mode=reload