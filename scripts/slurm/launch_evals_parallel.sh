#!/bin/bash
#SBATCH --job-name=openai_rwr_eval
#SBATCH --output=openai_rwr_eval_%A_%a.out
#SBATCH --error=openai_rwr_eval_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:A6000:3

# Load necessary modules and activate conda environment
source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam
module load cuda-12.3

# Arrays for the different configurations
defender_model_dirs=(
    "/data/group_data/rl/experiments/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-10-14-05-04-334/checkpoint-12/"
    "/data/group_data/rl/experiments/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-10-14-39-11-005/checkpoint-12/"
    "/data/group_data/rl/experiments/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-10-15-15-28-107/checkpoint-12/"
    "/data/group_data/rl/experiments/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-10-15-49-31-596/checkpoint-12/"
)

defender_model_types=(
    "trained_defender_natural_lang_binary"
    "trained_defender_binary"
    "trained_defender_natural_lang_multiclass"
    "trained_defender_multilabel"
)

value_function_types=(
    "natural_lang_binary"
    "binary"
    "natural_lang_multiclass"
    "multilabel"
)

# Check if all arrays have the same length
if [ ${#defender_model_dirs[@]} -ne ${#defender_model_types[@]} ] || [ ${#defender_model_dirs[@]} -ne ${#value_function_types[@]} ]; then
    echo "Error: All arrays must have the same length"
    exit 1
fi

# Get the current array task ID
task_id=$SLURM_ARRAY_TASK_ID

echo "Running configuration $((task_id+1)) of ${#defender_model_dirs[@]}"

# Run the Python script with the current configuration
python /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/evaluate.py \
    attacker.model_dir="/data/group_data/rl/experiments/redteaming/multiturn_rwr_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-23-13-23-23-840/checkpoint-183/" \
    dataset_configs="openai" \
    attacker.model_type="rwr_trained_attacker" \
    defender.model_dir="${defender_model_dirs[task_id]}" \
    defender.model_type="${defender_model_types[task_id]}" \
    value_function_evaluate.defender.value_function_type="${value_function_types[task_id]}"

# Check if the Python script executed successfully
if [ $? -ne 0 ]; then
    echo "Error: Python script failed for configuration $((task_id+1))"
    exit 1
fi

echo "Configuration $((task_id+1)) completed successfully"