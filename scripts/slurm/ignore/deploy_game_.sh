#!/bin/bash

# Source environment files
source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate redteam

# Set environment variables
export HF_HOME="/data/tir/projects/tir6/bisk/athankar/projects/.cache"
export LOGDIR="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/$(date +'%Y-%m-%d-%H-%M-%S-%3N')"

# Parse command-line arguments
CONTROLLER_PORT=$1
ATTACKER_MODEL_PATH=$2
DEFENDER_MODEL_PATH=$3
ATTACKER_WORKER_PORT=$4
DEFENDER_WORKER_PORT=$5
API_PORT=$6
SCRIPT_PATH=$7

# Start the controller
python3 -m fastchat.serve.controller --host 0.0.0.0 --port $CONTROLLER_PORT &

# Start the attacker model worker
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker \
    --model-path $ATTACKER_MODEL_PATH \
    --host 0.0.0.0 \
    --model-name "attacker" \
    --controller-address "http://localhost:$CONTROLLER_PORT" \
    --port $ATTACKER_WORKER_PORT \
    --worker-address "http://localhost:$ATTACKER_WORKER_PORT" \
    --device cuda &

# Start the defender model worker
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker \
    --model-path $DEFENDER_MODEL_PATH \
    --model-name "defender" \
    --host 0.0.0.0 \
    --controller-address "http://localhost:$CONTROLLER_PORT" \
    --port $DEFENDER_WORKER_PORT \
    --worker-address "http://localhost:$DEFENDER_WORKER_PORT" \
    --device cuda &

# Start the OpenAI API server
python3 -m fastchat.serve.openai_api_server \
    --host 0.0.0.0 \
    --port $API_PORT \
    --controller-address "http://localhost:$CONTROLLER_PORT"

# Wait for models to load (TODO: Replace with a proper check)
echo "Waiting for models to load..."
sleep 300

# Run the redteaming game script
# python $SCRIPT_PATH/redteaming/scripts/play_redteaming_game.py \
#     attacker.chat_completion.port=$API_PORT attacker.model=attacker
#     defender.chat_completion.port=$API_PORT defender.model=defender

# python /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/play_redteaming_game.py attacker.chat_completion.port=8003 attacker.model=attacker defender.chat_completion.port=8003 defender.model=defender