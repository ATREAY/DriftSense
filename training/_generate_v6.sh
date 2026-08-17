#!/bin/bash
#SBATCH --job-name=ds_gen_v6
#SBATCH --partition=cse-cpu-all
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail
PROJECT_ROOT="/u/student/2025/cs25mtech14023/DriftSense"
PYTHON_BIN="/u/student/2025/cs25mtech14023/.conda/envs/samamba/bin/python"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" generator/generate_dataset.py --architecture both --num-pairs 5000 --output-dir data/train --seed 5000 --workers 32
"$PYTHON_BIN" generator/generate_dataset.py --architecture both --num-pairs 200 --output-dir data/val --seed 6000 --workers 32 --harder-noise
