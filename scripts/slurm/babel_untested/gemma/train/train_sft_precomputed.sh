#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
module load cuda-12.3
conda activate redteam
source /data/user_data/athankar/redteaming/scripts/slurm/env_files/.babel_env

DATASET_TYPE=${1:-""}
DATA_PATH=${2:-""}
MASTER_PORT=${3:-29500}
LEARNING_RATE=${4:-1e-6}
NUM_EPOCHS=${5:-2}
ALGO=${6:-"sft_precomputed"}
EXPERIMENT_DESC=${7:-""}
# Command-line argument parsing
MODEL_PATH="google/gemma-2-2b-it"
AGENT_TYPE="defender"
MAX_LENGTH=4096
RUN_NAME="multiturn_sft_${AGENT_TYPE}_${MODEL_PATH}_$(date +'%Y-%m-%d-%H-%M-%S-%3N')"
LOGDIR="$MODEL_PARENT_DIR/$RUN_NAME"

# Run the first job
deepspeed --master_port $MASTER_PORT $REPO_DIR/redteam/train/train.py  \
        --algo $ALGO \
        --model_name_or_path $MODEL_PATH \
        --seed 42   \
        --data_path $DATA_PATH \
        --eval_data_path "" \
        --agent_type $AGENT_TYPE \
        --dataset_type $DATASET_TYPE \
        --max_length $MAX_LENGTH \
        --model_name $MODEL_PATH \
        --output_dir $LOGDIR  \
        --cache_dir $HF_HOME \
        --run_name $RUN_NAME \
        --exp_desc $EXPERIMENT_DESC \
        --deepspeed $REPO_DIR/scripts/configs/deepspeed/zero3.json \
        --bf16 True \
        --num_train_epochs 2  \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1   \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "steps" \
        --eval_steps 100000  \
        --save_strategy "steps"  \
        --save_steps 100000  \
        --save_total_limit 1 \
        --learning_rate $LEARNING_RATE \
        --weight_decay 0.0   \
        --warmup_ratio 0.04   \
        --lr_scheduler_type "cosine"   \
        --logging_steps 1     \
        --tf32 True    \
        --model_max_length $MAX_LENGTH \
        --torch_empty_cache_steps 1 \
        --gradient_checkpointing True \
        --remove_unused_columns False

# Find the latest checkpoint directory inside the output path
LATEST_CHECKPOINT=$(ls -td $LOGDIR/checkpoint-* | head -1)

DEFENDER_MODEL_DIR=$LATEST_CHECKPOINT
DEFENDER_MODEL_NAME="$EXPERIMENT_DESC.lr$$LEARNING_RATE"

# Launch evals only if algo is sft_precomputed. 
# SFT^* does not need to be evaluated. Directly launch DPO runs
if [ "$ALGO" == "sft_precomputed" ]; then
    echo "Launching evals for algo: $ALGO"
    for temperature in 0.0 0.7 1.0
    do
        sbatch $REPO_DIR/scripts/slurm/babel_untested/gemma/eval_bench/eval_gemma_oai.sh $temperature $DEFENDER_MODEL_DIR $DEFENDER_MODEL_NAME "$EXPERIMENT_DESC.temp$temperature"
        sbatch $REPO_DIR/scripts/slurm/babel_untested/gemma/eval_bench/eval_gemma_jbb.sh $temperature $DEFENDER_MODEL_DIR $DEFENDER_MODEL_NAME "$EXPERIMENT_DESC.temp$temperature"
        sbatch $REPO_DIR/scripts/slurm/babel_untested/gemma/eval_bench/eval_gemma_ss.sh $temperature $DEFENDER_MODEL_DIR $DEFENDER_MODEL_NAME "$EXPERIMENT_DESC.temp$temperature"
    done
else
    echo "Skipping evals because algo is not sft_precomputed. Current algo: $ALGO"
fi
# Launch DPO runs

sbatch $REPO_DIR/scripts/slurm/babel_untested/gemma/train/train_dpo.sh $DATASET_TYPE $DATA_PATH $MASTER_PORT $LEARNING_RATE "dpo_$EXPERIMENT_DESC" $DEFENDER_MODEL_DIR
