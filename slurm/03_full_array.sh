#!/bin/bash
#SBATCH --job-name=tfm-ens-full
#SBATCH --account=aip-ebrahimi
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Full TabArena campaign. Do NOT sbatch this directly -- submit through
# slurm/submit_full.sh, which sets --array, --time and --mem per size bucket
# and passes BUCKET through.
#
#     cd ~/tabicl-ens && bash slurm/submit_full.sh
#
# One array task = one (dataset, coalition), looping splits internally and
# skipping any split already cached. Idempotent: resubmit to resume.

set -euo pipefail

source "slurm/_common.sh"
cc_activate

export SPLIT_BACKEND=tabarena
BUCKET="${BUCKET:-all}"

COALITIONS=(base A1 A2 A3 A1A2 A1A3 A2A3 A1A2A3 shipped)
N_COAL=${#COALITIONS[@]}

mapfile -t DATASETS < <(python -c "
import os
from experiments.datasets import tabarena_datasets, dataset_info
b = os.environ['BUCKET']
for d in tabarena_datasets():
    if b == 'all' or dataset_info(d)['bucket'] == b:
        print(d)
")

if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "FATAL: no datasets in bucket '$BUCKET'. Has prepare_tabarena.py run?" >&2
  exit 1
fi

DATASET=${DATASETS[$(( SLURM_ARRAY_TASK_ID / N_COAL ))]}
COALITION=${COALITIONS[$(( SLURM_ARRAY_TASK_ID % N_COAL ))]}

cc_banner
echo "=== bucket=$BUCKET task=${SLURM_ARRAY_TASK_ID}: $DATASET / $COALITION ==="

srun python -m experiments.run_cell \
  --dataset      "$DATASET" \
  --coalition    "$COALITION" \
  --n-estimators 32 \
  --out          "$CACHE_DIR" \
  --seed         0 \
  --device       cuda

echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
