#!/bin/bash
#SBATCH --job-name=tfm-ens-a5
#SBATCH --account=aip-ebrahimi
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# A5: context-side perturbation. Submit through slurm/submit_a5.sh, which sets
# --array/--time/--mem per size bucket and passes BUCKET.
#
#     cd ~/tabicl-ens && bash slurm/submit_a5.sh
#
# WHY THIS IS SLOWER THAN 03_full_array.sh
# For every other axis the context is fixed, so one fit serves M cheap forward
# passes. A5 changes the context, which for an in-context learner IS the
# training data -- so it needs M FITS. Budget roughly M times the per-fit cost,
# not M times the per-forward-pass cost. Hence M=16 by default and the longer
# wall clocks in submit_a5.sh.

set -euo pipefail

source "slurm/_common.sh"
cc_activate

export SPLIT_BACKEND=tabarena
BUCKET="${BUCKET:-all}"
COALITIONS=(A5_f25 A5_f50 A5_f75 A5_f90 A5bal_f50)
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
echo "=== A5 bucket=$BUCKET task=${SLURM_ARRAY_TASK_ID}: $DATASET / $COALITION ==="

srun python -m experiments.run_cell \
  --dataset      "$DATASET" \
  --coalition    "$COALITION" \
  --n-estimators "${A5_M:-16}" \
  --out          "$CACHE_DIR" \
  --seed         0 \
  --device       cuda

echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
