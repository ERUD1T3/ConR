#!/bin/bash

#SBATCH --job-name=gmm_preprocess     # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --mem=32GB                    # Memory per node
#SBATCH --time=infinite              # Time limit
#SBATCH --partition=eternity               # CPU partition (GMM doesn't need GPU)
#SBATCH --output=./logs/%x.%J.out     # Output file
#SBATCH --error=./logs/%x.%J.err      # Error file

echo "Starting GMM preprocessing at date $(date)"

echo "Running on hosts: $SLURM_NODELIST"

echo "Running on $SLURM_NNODES nodes."

echo "Running on $SLURM_NPROCS processors."

echo "Current working directory is $(pwd)"

# Set K value
K=128

# Define datasets to process
datasets=("sep" "sarcos" "onp" "bf" "asc" "ed")

# Create logs directory if it doesn't exist
mkdir -p logs

# Set data directory to Linux path
DATA_DIR="/home1/jmoukpe2016/BalancedMSE/neurips2025/data"

echo "Processing GMM for all datasets with K=$K"

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "=================================="
    echo "Processing dataset: $dataset"
    echo "=================================="
    
    srun python -m preprocess_gmm \
        --dataset $dataset \
        --K $K \
        --data_dir $DATA_DIR \
        --batch_size 512 \
        --workers 4 \
        --store_root checkpoint 
    
    if [ $? -eq 0 ]; then
        echo "Successfully processed dataset: $dataset"
    else
        echo "Error processing dataset: $dataset"
    fi
    echo ""
done

echo "GMM preprocessing completed at date $(date)"

# Usage:
# To run all datasets: sbatch aip_scripts/slurm_gmm.sh
# To run specific dataset, modify the datasets array or create a variant script 