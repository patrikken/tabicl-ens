#!/bin/bash
#SBATCH --job-name=tfm-ens-a4
#SBATCH --account=def-CHANGEME
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# A4: feature-side subsampling. Submit through slurm/submit_a4.sh, which sets
# --array/--time/--mem per size bucket and passes BUCKET and A4_SET.
#
#     cd ~/tabicl-ens && bash slurm/submit_a4.sh
#
# COST, same shape as A5: dropping columns changes the fit, so M members means
# M FITS, not M forward passes through one fit. Budget M times the per-fit
# cost. Hence M=16 by default.
#
# TASK INDEXING differs from 05_a5_array.sh on purpose. Not every coalition
# applies to every dataset -- A4 needs enough columns, and column-LOFO needs
# d >= 2M or its folds are mostly empty -- so the (dataset, coalition) grid is
# ragged. We materialise the applicable pairs in Python and index into that
# list, instead of the id/N_COAL arithmetic that assumes a full grid.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cc_activate

export SPLIT_BACKEND=tabarena
BUCKET="${BUCKET:-small}"
A4_SET="${A4_SET:-sweep}"          # sweep | variants | all

mapfile -t PAIRS < <(BUCKET="$BUCKET" A4_SET="$A4_SET" python -c "
import os
from experiments.coalitions import A4_SWEEP, A4_VARIANTS, a4_applicable
from experiments.datasets import tabarena_datasets, dataset_info
sets = {'sweep': A4_SWEEP, 'variants': A4_VARIANTS, 'all': A4_SWEEP + A4_VARIANTS}
coals = sets[os.environ['A4_SET']]
b = os.environ['BUCKET']
for d in tabarena_datasets():
    info = dataset_info(d)
    if b != 'all' and info['bucket'] != b:
        continue
    for c in coals:
        if a4_applicable(c, info['n_features']):
            print(f'{d} {c}')
")
[ ${#PAIRS[@]} -eq 0 ] && { echo "FATAL: no applicable (dataset, coalition) pairs for bucket='$BUCKET' set='$A4_SET'" >&2; exit 1; }

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#PAIRS[@]}" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID} beyond the ${#PAIRS[@]} applicable pairs; nothing to do"
  exit 0
fi

read -r DATASET COALITION <<<"${PAIRS[$SLURM_ARRAY_TASK_ID]}"

cc_banner
echo "=== A4 bucket=$BUCKET set=$A4_SET task=${SLURM_ARRAY_TASK_ID}/${#PAIRS[@]}: $DATASET / $COALITION ==="

# M: left to run_cell's FEATURE_SUB_DEFAULT_M unless A4_M is set. Never passed
# for the column-LOFO coalitions, where M is fixed by the fold count -- passing
# one there would be silently ignored and the logs would misreport the budget.
EXTRA=()
if [ -n "${A4_M:-}" ] && [[ "$COALITION" != *lofo* ]]; then
  EXTRA+=(--n-estimators "$A4_M")
fi

srun python -m experiments.run_cell \
  --dataset   "$DATASET" \
  --coalition "$COALITION" \
  --out       "$CACHE_DIR" \
  --seed      0 \
  --device    cuda \
  "${EXTRA[@]}"

echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
