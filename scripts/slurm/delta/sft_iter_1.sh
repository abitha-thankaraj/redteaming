#!/bin/bash
#SBATCH --job-name=iter_1_sft_sweep
#SBATCH --output=/scratch/bcgv/athankaraj/logs/slurm/%A_%a.out
#SBATCH --error=/scratch/bcgv/athankaraj/logs/slurm/%A_%a.err
#SBATCH --account=bcgv-delta-gpu
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpuA40x4
#SBATCH --mem=220G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64 
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest 
#SBATCH --no-requeue
#SBATCH --time=12:00:00
#SBATCH --exclude=gpub054



source ~/.bashrc
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
module load cuda/12.3.0
conda activate redteam
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env


# Command-line argument parsing
MODEL_PATH=${1:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
AGENT_TYPE=${2:-"defender"}
MASTER_PORT=${3:-29500}
DATASET_TYPE=${4:-""}
VALUE_FUNCTION_TYPE=${5:-""}
VALUE_FUNCTION_EXPERIMENT=${6:-""}
LEARNING_RATE=${7:-1e-5}
RWR_TEMPERATURE=${8:-1.0}
LENGTH_KEY=${9:-"Meta-Llama-3.1-8B-Instruct_length"}
EXPERIMENT_DESC=${10:-"sft_lr_sweep"}
NUM_SAMPLES=${11:-null}


MAX_LENGTH=4096
RUN_NAME="multiturn_sft_${AGENT_TYPE}_${MODEL_PATH}_$(date +'%Y-%m-%d-%H-%M-%S-%3N')"
LOGDIR="$MODEL_PARENT_DIR/$RUN_NAME"


# Run the first job
deepspeed --master_port $MASTER_PORT $REPO_DIR/redteam/train/sft.py  \
        --model_name_or_path $MODEL_PATH \
        --seed 42   \
        --data_path $DATA_DIR/gen_judge_multiturn_conversation_combined/iter_1 \
        --eval_data_path $DATA_DIR/gen_judge_multiturn_conversation_combined/combined_eval_data_llama_rewards_flat_length_added.json \
        --agent_type $AGENT_TYPE \
        --length_key $LENGTH_KEY \
        --max_length $MAX_LENGTH \
        --output_dir $LOGDIR  \
        --cache_dir $HF_HOME \
        --run_name $RUN_NAME \
        --exp_desc $EXPERIMENT_DESC \
        --deepspeed $REPO_DIR/scripts/configs/deepspeed/zero3_mem_eff.json \
        --bf16 True \
        --num_train_epochs 2  \
        --per_device_train_batch_size 3 \
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
# Convert the latest checkpoint to fp32
python $LATEST_CHECKPOINT/zero_to_fp32.py -d $LATEST_CHECKPOINT $LATEST_CHECKPOINT/pytorch_model.bin


# Schedule the evals to run after the first job; 0, 0.7, 1.0 temperature

SFT_DEFENDER_MODEL_NAME="sft_trained_defender_1"
# Model parent dir is the parent directory for the checkpoint folder 
# The checkpoint folder is where the model is loaded from
SFT_DEFENDER_MODEL_DIR=$LATEST_CHECKPOINT

SFT_ATTACKER_MODEL_TYPE="sft_trained_attacker_0"
SFT_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135
RWR_ATTACKER_MODEL_TYPE="rwr_trained_attacker_0"
RWR_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturn_rwr_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-23-13-23-23-840/checkpoint-183



# Args: $DEFENDER_MODEL_PARENT_DIR $TEMPERATURE $DEFENDER_MODEL_NAME $ATTACKER_MODEL_DIR $ATTACKER_MODEL_NAME
# for loop through temperatures
for temperature in 0.0 0.7 1.0
do
    sbatch --dependency=afterok:$SLURM_JOB_ID $REPO_DIR/scripts/slurm/delta/evaluate_iter_0.sh $temperature $SFT_DEFENDER_MODEL_DIR $SFT_DEFENDER_MODEL_NAME $SFT_ATTACKER_MODEL_DIR $SFT_ATTACKER_MODEL_TYPE $EXPERIMENT_DESC
    # sbatch --dependency=afterok:$SLURM_JOB_ID $REPO_DIR/scripts/slurm/delta/evaluate_iter_0.sh $temperature $SFT_DEFENDER_MODEL_DIR $SFT_DEFENDER_MODEL_NAME $RWR_ATTACKER_MODEL_DIR $RWR_ATTACKER_MODEL_TYPE $EXPERIMENT_DESC
done
