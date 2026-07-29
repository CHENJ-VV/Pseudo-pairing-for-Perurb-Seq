#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --job-name=gene_program_eval
#SBATCH --output=logs/gene_program_eval.%j.out
#SBATCH --error=logs/gene_program_eval.%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32

set -euo pipefail

CONFIG_INPUT="${1:?Usage: sbatch slurm/run_gene_program.sbatch <config.json> [build|evaluate|all]}"
MODE="${2:-all}"
FOLDER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GENE_PROGRAM_PYTHON:-python}"

if [[ "${CONFIG_INPUT}" = /* ]]; then
  CONFIG_PATH="${CONFIG_INPUT}"
elif [[ -e "${CONFIG_INPUT}" ]]; then
  CONFIG_PATH="$(realpath "${CONFIG_INPUT}")"
else
  CONFIG_PATH="${FOLDER_ROOT}/${CONFIG_INPUT}"
fi

cd "${FOLDER_ROOT}"
mkdir -p logs

case "${MODE}" in
  build)
    "${PYTHON_BIN}" run_gene_program_pipeline.py --config "${CONFIG_PATH}" --build
    ;;
  evaluate)
    "${PYTHON_BIN}" run_gene_program_pipeline.py --config "${CONFIG_PATH}" --evaluate
    ;;
  all)
    "${PYTHON_BIN}" run_gene_program_pipeline.py --config "${CONFIG_PATH}" --build --evaluate
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use build, evaluate, or all." >&2
    exit 2
    ;;
esac
