#!/bin/bash
#SBATCH --job-name=driftsense
#SBATCH --partition=cse-gpu-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log
# --nodelist is appended at submission time by slurm_train.sh, which picks
# the node via select_gpu.sh (A100 > V100 > P100 by live idle GPU count).

set -euo pipefail

CONFIG="${1:-configs/dram_finfet.yaml}"
RESUME="${2:-}"
PROJECT_ROOT="/u/student/2025/cs25mtech14023/DriftSense"
PYTHON_BIN="/u/student/2025/cs25mtech14023/.conda/envs/samamba/bin/python"

cd "$PROJECT_ROOT"

echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Config: $CONFIG"
echo "Resume: ${RESUME:-<none>}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv || true

if [[ -n "$RESUME" ]]; then
  "$PYTHON_BIN" training/train.py --config "$CONFIG" --resume "$RESUME"
else
  "$PYTHON_BIN" training/train.py --config "$CONFIG"
fi
