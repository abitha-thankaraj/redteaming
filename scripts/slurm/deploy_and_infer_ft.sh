#!/bin/bash

source /home/ftajwar/.bashrc
source /home/ftajwar/anaconda3/etc/profile.d/conda.sh
conda activate fastchat

export TRANSFORMERS_CACHE=/data/user_data/ftajwar/training_cache
export HF_HOME=/data/user_data/ftajwar/training_cache

# Starting the controller
python3 -m fastchat.serve.controller --host 0.0.0.0 --port $6 &
# Starting the model worker with the specified model path
python3 -m fastchat.serve.model_worker --model-path $1 --host 0.0.0.0 --controller-address "http://localhost:$6" --port $7 --worker-address "http://localhost:$7" &

python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port $2 --controller-address "http://localhost:$6" &

sleep 300; # Sleep for 300 seconds. Wait until the model is loaded

python $3/scripts/data_generation/evaluate_multiturn_attacks.py chat_completion=$4 chat_completion.port=$2 multiturn_generated_attack_prompts_fname=$5

# $1: model path
# $2: port
# $3: path to the script
# $4: chat_completion
# $5: dataset_type
