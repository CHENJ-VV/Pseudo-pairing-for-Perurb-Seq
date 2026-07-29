#!/bin/bash
#SBATCH --job-name=ppfm_encoder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch run_stage.sbatch <config.yaml> <geneformer|scgpt|sccello|scimilarity|scvi|hvg|umap|mlp>" >&2
  exit 2
fi

CONFIG=$(realpath "$1")
STAGE="$2"
FOLDER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$FOLDER_ROOT/logs"

# Activate only the lightweight launcher environment here. The YAML file selects
# a model-specific Python executable for the actual encoder subprocess.
python "$FOLDER_ROOT/launcher.py" validate --config "$CONFIG"
python "$FOLDER_ROOT/launcher.py" run --config "$CONFIG" --model "$STAGE"
