#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
source /data/user_data/athankar/redteaming/scripts/slurm/env_files/.babel_env
# For yq
conda activate redteam

# Load dependencies to parse YAML (e.g., yq)
# Install yq if not available: `conda install -c conda-forge yq` or `sudo apt install yq`
if ! command -v yq &> /dev/null; then
    echo "yq is required but not installed. Please install it and try again."
    exit 1
fi

LAUNCH_SCRIPTS_DIR="$REPO_DIR/scripts/slurm/babel_untested/gemma"

CONFIG_FILE="$LAUNCH_SCRIPTS_DIR/launch_configs.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# Read and launch jobs
num_jobs=$(yq '.configs | length' "$CONFIG_FILE")
echo "Found $num_jobs jobs in the configuration file."

# set -x
for i in $(seq 0 $((num_jobs - 1))); do
    # Properly extract values from the YAML file
    dataset_type=$(yq ".configs[$i].dataset_type" "$CONFIG_FILE" | tr -d '"')
    data_path=$(yq ".configs[$i].data_path" "$CONFIG_FILE" | tr -d '"')
    master_port=$(yq ".configs[$i].master_port" "$CONFIG_FILE")
    learning_rate=$(yq ".configs[$i].learning_rate" "$CONFIG_FILE")
    num_epochs=$(yq ".configs[$i].num_epochs" "$CONFIG_FILE")
    algo=$(yq ".configs[$i].algo" "$CONFIG_FILE" | tr -d '"')
    experiment_desc=$(yq ".configs[$i].experiment_desc" "$CONFIG_FILE" | tr -d '"')
    
    # Pass arguments to sbatch without redundant quoting
    sbatch "$LAUNCH_SCRIPTS_DIR/launch_train_sft.sh" \
        "$dataset_type" \
        "$data_path" \
        "$master_port" \
        "$learning_rate" \
        "$num_epochs" \
        "$algo" \
        "$experiment_desc"
done
# set +x  # Disables debugging output


echo "All jobs launched."
