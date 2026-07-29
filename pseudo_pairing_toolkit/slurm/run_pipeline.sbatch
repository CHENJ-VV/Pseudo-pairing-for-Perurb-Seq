#!/bin/bash
#SBATCH --job-name=pseudopair
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

FOLDER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_INPUT="${1:-configs/replogle_rpe.ibex.example.yaml}"
STAGE="${2:-all}"
PSEUDOPAIR_BIN="${PSEUDOPAIR_BIN:-pseudopair}"

if [[ "${CONFIG_INPUT}" = /* ]]; then
  CONFIG_PATH="${CONFIG_INPUT}"
elif [[ -e "${CONFIG_INPUT}" ]]; then
  CONFIG_PATH="$(realpath "${CONFIG_INPUT}")"
else
  CONFIG_PATH="${FOLDER_ROOT}/${CONFIG_INPUT}"
fi

cd "${FOLDER_ROOT}"
mkdir -p logs

"${PSEUDOPAIR_BIN}" validate --config "${CONFIG_PATH}" --check-files
"${PSEUDOPAIR_BIN}" run --config "${CONFIG_PATH}" --stage "${STAGE}"
