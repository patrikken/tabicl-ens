#!/bin/bash
#SBATCH --job-name=tfm-lofo
#SBATCH --account=def-CHANGEME
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# A5-LOFO: leave-one-fold-out context partitioning. Submit via submit_lofo.sh.
#
#     cd ~/tabicl-ens && bash slurm/submit_lofo.sh
#
# M is NOT a free parameter here: the coalition name fixes the fold count and
# the context fraction follows, f = 1 - 1/M. run_cell overrides
# --n-estimators accordingly, so do not pass one.
#
# COST: M members = M fits, and at high M each fit is nearly full-sized.
# A5lofo_M64 is ~64 near-full fits per cell -- roughly 4x A5lofo_M16 and well
# over an order of magnitude more than any feature-side coalition. Start with
# the small bucket and read the timings before going further.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cc_activate

export SPLIT_BACKEND=tabarena
BUCKET="${BUCKET:-small}"
COALITIONS=(A5lofo_M8 A5lofo_M16 A5lofo_M32 A5lofo_M64)
N_COAL=${#COALITIONS[@]}

mapfile -t DATASETS < <(python -c "
import os
from experiments.datasets import tabarena_datasets, dataset_info
b = os.environ['BUCKET']
for d in tabarena_datasets():
    if b == 'all' or dataset_info(d)['bucket'] == b:
        print(d)
")
[ ${#DATASETS[@]} -eq 0 ] && { echo "FATAL: no datasets in bucket '$BUCKET'" >&2; exit 1; }

DATASET=${DATASETS[$(( SLURM_ARRAY_TASK_ID / N_COAL ))]}
COALITION=${COALITIONS[$(( SLURM_ARRAY_TASK_ID % N_COAL ))]}

cc_banner
echo "=== LOFO bucket=$BUCKET task=${SLURM_ARRAY_TASK_ID}: $DATASET / $COALITION ==="

srun python -m experiments.run_cell \
  --dataset   "$DATASET" \
  --coalition "$COALITION" \
  --out       "$CACHE_DIR" \
  --seed      0 \
  --device    cuda

echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
