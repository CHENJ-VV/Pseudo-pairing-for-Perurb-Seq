#!/bin/bash --login
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH -J RPE_all_inverse
#SBATCH -o batch_log/RPE_all_inverse.%J.out
#SBATCH -e batch_log/RPE_all_inverse.%J.err
#SBATCH --mail-user=chenj0i@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=25:00:00
#SBATCH --mem=256G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=a100
#SBATCH --gpus-per-node=1

#run the application:
# define some environment variables
# activate the conda environment
ENV_PREFIX="/ibex/user/chenj0i/conda/envs/seacells"

module purge
conda activate "$ENV_PREFIX"
export LD_LIBRARY_PATH=/ibex/user/chenj0i/conda/envs/seacells/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/ibex/user/chenj0i/conda/envs/seacells/lib/libstdc++.so.6

python /ibex/user/chenj0i/Perturbation/SEACells/Junfan_scripts/Replogle_RPE/pseudo_pairing_evaluation_pipeline/execute_evaluation_pipeline.py