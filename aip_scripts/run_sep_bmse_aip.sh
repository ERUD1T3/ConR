#!/bin/bash

#SBATCH --job-name=sep_bmse_aip       # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --mem=64GB                    # Memory per node
#SBATCH --time=infinite              # Time limit
#SBATCH --partition=gpu1          # Partition
#SBATCH --gres=gpu:1                 # Number of GPUs per node
#SBATCH --output=./logs/%x.%J.out     # Output file
#SBATCH --error=./logs/%x.%J.err      # Error file

echo "Starting SEP BMSE training at date $(date)"

echo "Running on hosts: $SLURM_NODELIST"

echo "Running on $SLURM_NNODES nodes."

echo "Running on $SLURM_NPROCS processors."

echo "Current working directory is $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Configuration based on run_sep_bmse_chan.sh ---
SEEDS="456789 42 123 0 9999"
DATASET="sep"
BATCH_SIZE=200
EPOCHS=5700
MLP_HIDDENS="512 32 256 32 128 32 64 32"
MLP_EMBED_DIM=32
MLP_DROPOUT=0.5

LR=5e-4
WEIGHT_DECAY=1

# BMSE / GAI specific settings - Updated to use K=64 based on GMM preprocessing
GMM_FILE="/home1/jmoukpe2016/BalancedMSE/neurips2025/gmm/sep_gmm_K128.pkl"
DATA_DIR="/home1/jmoukpe2016/BalancedMSE/neurips2025/data"
UPPER_THRESHOLD=2.30258509299

# --- Run Training ---
echo "Starting training for dataset: ${DATASET}, seeds: ${SEEDS}"
echo "Using GMM file: ${GMM_FILE}"
echo "=================================="

cd neurips2025

srun python train.py \
    --seeds ${SEEDS} \
    --data_dir ${DATA_DIR} \
    --dataset ${DATASET} \
    --batch_size ${BATCH_SIZE} \
    --epoch ${EPOCHS} \
    --mlp_hiddens ${MLP_HIDDENS} \
    --mlp_embed_dim ${MLP_EMBED_DIM} \
    --mlp_dropout ${MLP_DROPOUT} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --bmse \
    --imp gai \
    --gmm_file ${GMM_FILE} \
    --upper_threshold ${UPPER_THRESHOLD}

if [ $? -eq 0 ]; then
    echo "Successfully completed SEP BMSE training"
else
    echo "Error in SEP BMSE training"
    exit 1
fi

echo "Training finished for seeds: ${SEEDS} at date $(date)"

# Usage:
# sbatch aip_scripts/run_sep_bmse_aip.sh 