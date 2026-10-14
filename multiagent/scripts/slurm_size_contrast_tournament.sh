#!/bin/bash -l
#SBATCH --job-name=css_size
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=95gb
#SBATCH --time=06:00:00
#SBATCH --account=OD-202968
#SBATCH --partition=h24gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sha511@csiro.au
#SBATCH --output=/scratch3/sha511/logs/size_contrast_tournament_%j.out
#SBATCH --error=/scratch3/sha511/logs/size_contrast_tournament_%j.err

module load cuda/12.4.1
module load pytorch/2.5.1-py312-cu124-mpi-sota
source ~/cssv3-venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch3/sha511/huggingface_cache
export TRANSFORMERS_CACHE=/scratch3/sha511/huggingface_cache
export TOKENIZERS_PARALLELISM=false

cd ~/compete-safety
python3 /scratch3/sha511/run_size_contrast_tournament.py
