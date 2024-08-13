#!/bin/bash

source /home/ftajwar/.bashrc
source /home/ftajwar/anaconda3/etc/profile.d/conda.sh
conda activate fastchat

export TRANSFORMERS_CACHE=/data/user_data/ftajwar/training_cache
export HF_HOME=/data/user_data/ftajwar/training_cache
export LOGDIR="/home/ftajwar/redteaming/scripts/logs/$(date +'%Y-%m-%d-%H-%M-%S-%3N')"

# Starting the controller
python3 -m fastchat.serve.controller &
# Starting the model worker with the specified model path
python3 -m fastchat.serve.model_worker --model-path meta-llama/Meta-Llama-3.1-8B-Instruct &
# Add model-list-mode because you're running the model-worker in the background and model loading takes time and does not 
python3 -m fastchat.serve.gradio_web_server  --host 0.0.0.0 --port 8001 --model-list-mode=reload
