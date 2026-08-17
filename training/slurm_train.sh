#!/bin/bash
# GPU-priority-aware training submission: picks the best currently-idle
# GPU tier (A100 > V100 > P100, see select_gpu.sh) and submits the actual
# training job (_train_job.sh) pinned to that node.
#
# Usage: training/slurm_train.sh <config_path> [resume_checkpoint] [--tiers A100,V100]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="configs/dram_finfet.yaml"
RESUME=""
TIERS_ARGS=()

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiers) TIERS_ARGS=(--tiers "$2"); shift 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
[[ ${#POSITIONAL[@]} -ge 1 ]] && CONFIG="${POSITIONAL[0]}"
[[ ${#POSITIONAL[@]} -ge 2 ]] && RESUME="${POSITIONAL[1]}"

node=$("$SCRIPT_DIR/select_gpu.sh" "${TIERS_ARGS[@]}")
echo "[slurm_train] submitting to node: $node"

sbatch --nodelist="$node" "$SCRIPT_DIR/_train_job.sh" "$CONFIG" "$RESUME"
