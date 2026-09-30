#!/bin/bash -l
#SBATCH --job-name=cssv3_QS_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=120gb
#SBATCH --time=24:00:00
#SBATCH --account=OD-202968
#SBATCH --partition=h24gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sha511@csiro.au

module load pytorch/2.5.1-py312-cu124-mpi-sota
source ~/cssv3-venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch3/sha511/huggingface_cache
export TRANSFORMERS_CACHE=/scratch3/sha511/huggingface_cache
export TOKENIZERS_PARALLELISM=false

cd ~/compete-safety
python3 multiagent/scripts/verification/verify_model.py --agent QS --pass1_only
