#!/bin/bash
#SBATCH --job-name=pseudopair
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# Activate the environment in which this package and SEACells are installed.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate perturbation

CONFIG=${1:-configs/replogle_rpe.ibex.example.yaml}
STAGE=${2:-all}

pseudopair validate --config "$CONFIG" --check-files
pseudopair run --config "$CONFIG" --stage "$STAGE"
