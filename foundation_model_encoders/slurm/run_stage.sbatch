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
  echo "Usage: sbatch slurm/run_stage.sbatch <config.yaml> <geneformer|scgpt|sccello|scimilarity|scvi|hvg|umap|mlp>" >&2
  exit 2
fi

FOLDER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_INPUT="$1"
STAGE="$2"
LAUNCHER_PYTHON="${ENCODER_LAUNCHER_PYTHON:-python}"

if [[ "${CONFIG_INPUT}" = /* ]]; then
  CONFIG_PATH="${CONFIG_INPUT}"
elif [[ -e "${CONFIG_INPUT}" ]]; then
  CONFIG_PATH="$(realpath "${CONFIG_INPUT}")"
else
  CONFIG_PATH="${FOLDER_ROOT}/${CONFIG_INPUT}"
fi

cd "${FOLDER_ROOT}"
mkdir -p logs

# The YAML selects a model-specific Python executable for the encoder subprocess.
"${LAUNCHER_PYTHON}" launcher.py validate --config "${CONFIG_PATH}"
"${LAUNCHER_PYTHON}" launcher.py run --config "${CONFIG_PATH}" --model "${STAGE}"
