#!/bin/bash --login
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH -J Rpe_gen
#SBATCH -o batch_log/Rpe_gen.%J.out
#SBATCH -e batch_log/Rpe_gen.%J.err
#SBATCH --mail-user=chenj0i@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=2:00:00
#SBATCH --mem=256G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=v100
#SBATCH --gpus-per-node=1

#run the application:
# define some environment variables
# activate the conda environment
ENV_PREFIX="/ibex/user/chenj0i/conda/envs/seacells"

module purge
conda activate "$ENV_PREFIX"

python /ibex/user/chenj0i/Perturbation/SEACells/Junfan_scripts/Replogle_RPE/pseudo_pairing_refactor/execute_pseudo_pairing_repetitions_1.py