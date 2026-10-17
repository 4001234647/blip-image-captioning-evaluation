#!/bin/bash
#SBATCH --job-name=blip-100k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --output=logs/blip-100k-%j.out
#SBATCH --error=logs/blip-100k-%j.err

set -euo pipefail
cd /home/msai/li0034qi/blip_project/BLIP
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate my647
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo '===== BLIP 100K START ====='
date
hostname
which python
nvidia-smi

# Log GPU usage, VRAM usage and GPU power every 60 seconds.
(
  while true; do
    date '+%F %T'
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader
    sleep 60
  done
) > "logs/blip-100k-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

python train_blip_100k.py \
  --config output/caption_coco_smoke_10k/config.yaml \
  --pretrained /home/msai/li0034qi/blip_project/checkpoints/pretrained/model_base_caption_capfilt_large.pth \
  --output_dir output/caption_coco_100k_lr5e6 \
  --train_samples 100000 \
  --lr 5e-6 \
  --epochs 1 \
  --num_workers 4

echo '===== BLIP 100K FINISHED ====='
date
