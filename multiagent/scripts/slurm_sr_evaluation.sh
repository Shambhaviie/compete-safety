#!/bin/bash -l
#SBATCH --job-name=sr_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60gb
#SBATCH --time=04:00:00
#SBATCH --account=OD-202968
#SBATCH --partition=h24gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch3/sha511/logs/sr_evaluation_%j.out
#SBATCH --error=/scratch3/sha511/logs/sr_evaluation_%j.err

module load cuda/12.4.1
module load pytorch/2.5.1-py312-cu124-mpi-sota
source ~/cssv3-venv/bin/activate
export HF_HOME=/scratch3/sha511/huggingface_cache
export TRANSFORMERS_CACHE=/scratch3/sha511/huggingface_cache

python3 /scratch3/sha511/run_sr_evaluation.py
