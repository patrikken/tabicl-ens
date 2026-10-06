#!/bin/bash
#SBATCH --job-name=tfm-all
#SBATCH --account=aip-ebrahimi
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# One array task = one (dataset, coalition) pair of ONE model on ONE task.
# Do NOT sbatch this directly -- slurm/submit_all.sh writes the plan file, sets
# --array/--time/--mem per (model, task, bucket) and overrides --account/--gres
# from $ACCOUNT / $GRES if you export them.
#
# The pair is read from line (ARRAY_TASK_ID+1) of $PLAN, a file written at SUBMIT
# time. It is never re-derived here: cells finish while the array runs, so a list
# recomputed per task would shift underneath the index.
#
# Idempotent: the worker skips every split whose meta.json exists, so resubmit
# (submit_all.sh skips finished pairs) to resume after a time-out.

set -euo pipefail

source "slurm/_common.sh"
cc_activate

export SPLIT_BACKEND=tabarena
: "${MODEL:?MODEL (tabicl|tabfm) not set -- submit through slurm/submit_all.sh}"
: "${PLAN:?PLAN file not set -- submit through slurm/submit_all.sh}"
TASK_KIND="${TASK_KIND:-?}"
BUCKET="${BUCKET:-?}"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PLAN")"
if [ -z "$LINE" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID} is beyond the plan ($(wc -l < "$PLAN") lines); nothing to do"
  exit 0
fi
read -r DATASET COALITION <<<"$LINE"

cc_banner
echo "=== model=$MODEL task=$TASK_KIND bucket=$BUCKET #${SLURM_ARRAY_TASK_ID}: $DATASET / $COALITION ==="

case "$MODEL" in
  tabicl)
    # --n-estimators is left to the worker: 32 for the native axes, 16 for the
    # A4/A5 wrappers (M fits), and the partition size for LOFO.
    srun python -m experiments.run_cell \
      --dataset   "$DATASET" \
      --coalition "$COALITION" \
      --out       "$CACHE_DIR" \
      --seed      0 \
      --device    cuda
    ;;
  tabfm)
    export TABFM_CKPT="${TABFM_CKPT:-$PROJECT_ROOT/tabfm-ckpt}"
    export TABFM_REPO="${TABFM_REPO:-$CODE_ROOT/tabfm}"
    [ -d "$TABFM_CKPT" ] || { echo "FATAL: no TabFM checkpoint at $TABFM_CKPT (see slurm/prefetch_all.sh)" >&2; exit 1; }
    srun python -m experiments.run_cell_tabfm \
      --dataset      "$DATASET" \
      --coalition    "$COALITION" \
      --n-estimators "${TABFM_M:-32}" \
      --out          "$CACHE_DIR/tabfm" \
      --checkpoint   "$TABFM_CKPT" \
      --seed         0 \
      --device       cuda
    ;;
  *) echo "FATAL: unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
