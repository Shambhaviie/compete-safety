#!/bin/bash -l
#SBATCH --job-name=cssv3_60round
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256gb
#SBATCH --time=05:00:00
#SBATCH --account=OD-202968
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sha511@csiro.au

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

module load pytorch/2.5.1-py312-cu124-mpi-sota
source ~/cssv3-venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch3/sha511/huggingface_cache
export TRANSFORMERS_CACHE=/scratch3/sha511/huggingface_cache
export TOKENIZERS_PARALLELISM=false

cd ~/compete-safety

python3 multiagent/scripts/run_5round_test.py \
    --n_rounds 60 \
    --n_phase1_rounds 30 \
    --output_dir /scratch3/sha511/cssv3_results/60round_smoke \
    --checkpoint_dir /scratch3/sha511/cssv3_results/60round_smoke/checkpoints \
    --log_file /scratch3/sha511/logs/cssv3_60round_detail.log

echo "Job finished: $(date)"
