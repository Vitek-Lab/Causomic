#!/usr/bin/env bash
#SBATCH --job-name=benchmark_posterior
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

# Activate conda environment
source /home/kohler.d/miniconda/etc/profile.d/conda.sh
conda activate causomic_env

cd /home/kohler.d/Causomic/vignettes/simulations

python benchmark_posterior_accuracy.py
