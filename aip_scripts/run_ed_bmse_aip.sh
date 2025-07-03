#!/bin/bash

#SBATCH --job-name=ed_bmse_aip        # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --mem=64GB                    # Memory per node
#SBATCH --time=infinite              # Time limit
#SBATCH --partition=gpu1              # Partition
#SBATCH --gres=gpu:1                  # Number of GPUs per node
#SBATCH --output=./logs/%x.%J.out     # Output file
#SBATCH --error=./logs/%x.%J.err      # Error file

echo "Starting ED BMSE training at date $(date)"

echo "Running on hosts: $SLURM_NODELIST"

echo "Running on $SLURM_NNODES nodes."

echo "Running on $SLURM_NPROCS processors."

echo "Current working directory is $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Configuration based on run_ed_bmse_chan.sh ---
SEEDS="456789 42 123 0 9999"
DATASET="ed"
BATCH_SIZE=2400
EPOCHS=6000
MLP_HIDDENS="2048 128 1024 128 512 128 256 128"
MLP_EMBED_DIM=128
MLP_DROPOUT=0.2

LR=1e-4
WEIGHT_DECAY=0.1

# BMSE / GAI specific settings - Updated for ED dataset with appropriate K value
GMM_FILE="/home1/jmoukpe2016/BalancedMSE/neurips2025/gmm/ed_gmm_K4.pkl"
DATA_DIR="/home1/jmoukpe2016/BalancedMSE/neurips2025/data"

# Lower and upper thresholds for label range categorization (ED specific)
LOWER_THRESHOLD=-0.5
UPPER_THRESHOLD=0.5

# --- Run Training ---
echo "Starting training for dataset: ${DATASET}, seeds: ${SEEDS}"
echo "Using GMM file: ${GMM_FILE}"
echo "Lower threshold: ${LOWER_THRESHOLD}, Upper threshold: ${UPPER_THRESHOLD}"
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
    --lower_threshold ${LOWER_THRESHOLD} \
    --upper_threshold ${UPPER_THRESHOLD} \

if [ $? -eq 0 ]; then
    echo "Successfully completed ED BMSE training"
else
    echo "Error in ED BMSE training"
    exit 1
fi

echo "Training finished for seeds: ${SEEDS} at date $(date)"

# Usage:
# sbatch aip_scripts/run_ed_bmse_aip.sh 