#!/bin/bash -l
#SBATCH --job-name=cssv3_5round_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --time=2:00:00
#SBATCH --account=OD-202968
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --output=/scratch3/sha511/logs/cssv3_5round_%j.out
#SBATCH --error=/scratch3/sha511/logs/cssv3_5round_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.shambhavi@unsw.edu.au

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

module load pytorch/2.5.1-py312-cu124-mpi-sota

source ~/cssv3-venv/bin/activate

export HF_HOME=/scratch3/sha511/huggingface_cache
export TRANSFORMERS_CACHE=/scratch3/sha511/huggingface_cache
export TOKENIZERS_PARALLELISM=false

mkdir -p /scratch3/sha511/logs
mkdir -p /scratch3/sha511/cssv3_results/5round_test

cd ~/compete-safety

python multiagent/scripts/run_5round_test.py \
    --output_dir /scratch3/sha511/cssv3_results/5round_test \
    --checkpoint_dir /scratch3/sha511/cssv3_results/5round_test/checkpoints \
    --log_file /scratch3/sha511/logs/cssv3_5round_detail.log

echo "Job finished: $(date)"
