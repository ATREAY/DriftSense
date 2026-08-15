#!/bin/bash
# Dataset generation is CPU-bound (procedural rendering + noise, no GPU
# needed) and slow enough for larger counts that it shouldn't run on the
# shared login node -- submit it as a plain CPU job instead.
#
# Usage: training/slurm_generate.sh <output_dir> <num_pairs> [architecture]
set -euo pipefail

OUT_DIR="${1:?usage: slurm_generate.sh <output_dir> <num_pairs> [architecture]}"
NUM_PAIRS="${2:?usage: slurm_generate.sh <output_dir> <num_pairs> [architecture]}"
ARCH="${3:-both}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

sbatch --job-name=ds_gen --partition=cse-cpu-all --cpus-per-task=16 --mem=16G \
  --time=04:00:00 --output=%x_%j.log \
  --wrap="cd '$PROJECT_ROOT' && /u/student/2025/cs25mtech14023/.conda/envs/samamba/bin/python generator/generate_dataset.py --architecture '$ARCH' --num-pairs '$NUM_PAIRS' --output-dir '$OUT_DIR' --workers 16"
