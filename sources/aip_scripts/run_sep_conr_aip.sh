#!/bin/bash

#SBATCH --job-name=sep_conr_aip        # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --mem=64GB                     # Memory per node
#SBATCH --time=infinite               # Time limit
#SBATCH --partition=gpu1              # Partition
#SBATCH --gres=gpu:1                  # Number of GPUs per node
#SBATCH --output=./logs/%x.%J.out      # Output file
#SBATCH --error=./logs/%x.%J.err       # Error file

echo "Starting SEP ConR training at date $(date)"

echo "Running on hosts: $SLURM_NODELIST"

echo "Running on $SLURM_NNODES nodes."

echo "Running on $SLURM_NPROCS processors."

echo "Current working directory is $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Configuration for ConR (Contrastive Regularizer) ---
SEEDS="456789 42 123 0 9999"
DATASET="sep"
BATCH_SIZE=200
EPOCHS=5602
MLP_HIDDENS="512 32 256 32 128 32 64 32"
MLP_EMBED_DIM=32
MLP_DROPOUT=0.5

LR=5e-4
WEIGHT_DECAY=1

# ConR specific settings
CONR_BETA=4        # ConR loss coefficient
CONR_W=1           # Similarity window for ConR loss
CONR_T=0.07        # Temperature parameter for ConR
CONR_E=0.01        # Coefficient for eta in ConR

DATA_DIR="/home1/jmoukpe2016/BalancedMSE/neurips2025/data"
UPPER_THRESHOLD=2.30258509299

# --- Run Training ---
echo "Starting training for dataset: ${DATASET}, seeds: ${SEEDS}"
echo "Using ConR regularization with beta: ${CONR_BETA}, w: ${CONR_W}, t: ${CONR_T}, e: ${CONR_E}"
echo "=================================="


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
    --conr \
    --beta ${CONR_BETA} \
    -w ${CONR_W} \
    -t ${CONR_T} \
    -e ${CONR_E} \
    --upper_threshold ${UPPER_THRESHOLD}

if [ $? -eq 0 ]; then
    echo "Successfully completed SEP ConR training"
else
    echo "Error in SEP ConR training"
    exit 1
fi

echo "Training finished for seeds: ${SEEDS} at date $(date)"

# Usage:
# sbatch aip_scripts/run_sep_conr_aip.sh 