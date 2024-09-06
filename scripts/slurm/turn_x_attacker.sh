#!/bin/bash

#SBATCH --job-name=llama_$SLURM_ARRAY_TASK_ID
#SBATCH --output=/home/athankar/slurm/%A_%a.out
#SBATCH --error=/home/athankar/slurm/%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-9%8

# Chunked files directory
CHUNKED_DATA_DIR="/data/group_data/rl/datasets/redteaming/"
SCRIPTS_DIR="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts"
SLURM_OUT_DIR="/home/athankar/slurm"

FAST_CHAT_API_PORT=$((6987 + SLURM_ARRAY_TASK_ID))
WORKER_PORT=$((26090 + SLURM_ARRAY_TASK_ID))
CONTROLLER_PORT=$((26191 + SLURM_ARRAY_TASK_ID))
# Start
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
module load cuda-12.3

conda activate redteam

# Deploy FastChat
echo "Deploying FastChat..."
srun python $SCRIPTS_DIR/fastchat_deploy.py ports.openai=$FAST_CHAT_API_PORT models.0.worker_port=$WORKER_PORT ports.controller=$CONTROLLER_PORT &
FASTCHAT_PID=$!

# Wait for FastChat to start up
echo "Waiting for FastChat to start..."
sleep 120

# Check if FastChat is running
if ! ps -p $FASTCHAT_PID > /dev/null; then
    echo "Error: FastChat failed to start. Check logs for details."
    exit 1
fi

# Run data generation script
echo "Starting data generation..."

python $SCRIPTS_DIR/iterative_generate_tree_data.py chunk_num=$SLURM_ARRAY_TASK_ID oai_server_port=$FAST_CHAT_API_PORT
# Clean up
echo "Job completed. Shutting down FastChat..."
kill $FASTCHAT_PID

echo "Script execution completed."

