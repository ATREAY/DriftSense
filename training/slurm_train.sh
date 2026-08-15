#!/bin/bash
# GPU-priority-aware training submission: picks the best currently-idle
# GPU tier (A100 > V100 > P100, see select_gpu.sh) and submits the actual
# training job (_train_job.sh) pinned to that node.
#
# Usage: training/slurm_train.sh <config_path> [resume_checkpoint]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/dram_finfet.yaml}"
RESUME="${2:-}"

node=$("$SCRIPT_DIR/select_gpu.sh")
echo "[slurm_train] submitting to node: $node"

sbatch --nodelist="$node" "$SCRIPT_DIR/_train_job.sh" "$CONFIG" "$RESUME"
