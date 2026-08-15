#!/bin/bash
#SBATCH --job-name=ds_gen_big
#SBATCH --partition=cse-cpu-all
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail
PROJECT_ROOT="/u/student/2025/cs25mtech14023/DriftSense"
PYTHON_BIN="/u/student/2025/cs25mtech14023/.conda/envs/samamba/bin/python"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" generator/generate_dataset.py --architecture both --num-pairs 2000 --output-dir data/train --seed 500 --workers 16
