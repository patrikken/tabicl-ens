#!/bin/bash
#SBATCH --job-name=tfm-tabfm
#SBATCH --account=def-CHANGEME
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# TabFM (PyTorch) member capture. Submit through slurm/submit_tabfm.sh.
#
#     cd ~/tabicl-ens && bash slurm/submit_tabfm.sh small
#
# COST SHAPE DIFFERS FROM A4/A5. TabFM fits once and runs M forward passes over
# the same encoded context, so M members cost M *passes*, not M fits -- the
# cheap regime, like the TabICLv2 feature-side axes. What is expensive here is
# the model: 400M parameters, a 6.56 GB bf16 checkpoint loaded once per process.
# One array task therefore covers a whole (dataset, coalition) cell, looping
# every split inside, rather than one split per task.
#
# OFFLINE CHECKPOINT IS MANDATORY: compute nodes have no internet and the
# estimator would otherwise try HuggingFace. Snapshot it once on a login node:
#
#     python - <<'PY'
#     from huggingface_hub import snapshot_download
#     import os
#     snapshot_download("google/tabfm-1.0.0-pytorch",
#                       local_dir=os.environ["PROJECT_ROOT"] + "/tabfm-ckpt",
#                       allow_patterns=["classification/**", "config.json"])
#     PY
#
# then export TABFM_CKPT=$PROJECT_ROOT/tabfm-ckpt. NB the repo pins 1.0.0, not
# the 1.1.0 the paper's text mentions. Weights are tabfm-non-commercial-v1.0:
# research use only.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cc_activate

export SPLIT_BACKEND=tabarena
export TABFM_CKPT="${TABFM_CKPT:-$PROJECT_ROOT/tabfm-ckpt}"
export TABFM_REPO="${TABFM_REPO:-$CODE_ROOT/tabfm}"
BUCKET="${BUCKET:-small}"
SET="${TABFM_SET:-axes}"

[ -d "$TABFM_CKPT" ] || { echo "FATAL: no checkpoint at $TABFM_CKPT. Snapshot it on a login node (see header)." >&2; exit 1; }

mapfile -t PAIRS < <(BUCKET="$BUCKET" TABFM_SET="$SET" python -c "
import os
from experiments.tabfm_sets import coalition_set
from experiments.datasets import tabarena_datasets, dataset_info
b = os.environ['BUCKET']
for d in tabarena_datasets():
    info = dataset_info(d)
    if b != 'all' and info['bucket'] != b:
        continue
    for c in coalition_set(os.environ['TABFM_SET'], info['n_features']):
        print(f'{d} {c}')
")
[ ${#PAIRS[@]} -eq 0 ] && { echo "FATAL: no pairs for bucket='$BUCKET' set='$SET'" >&2; exit 1; }

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#PAIRS[@]}" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID} beyond ${#PAIRS[@]} pairs; nothing to do"; exit 0
fi
read -r DATASET COALITION <<<"${PAIRS[$SLURM_ARRAY_TASK_ID]}"

cc_banner
echo "=== TabFM bucket=$BUCKET set=$SET task=${SLURM_ARRAY_TASK_ID}/${#PAIRS[@]}: $DATASET / $COALITION ==="
echo "    checkpoint: $TABFM_CKPT"

srun python -m experiments.run_cell_tabfm \
  --dataset      "$DATASET" \
  --coalition    "$COALITION" \
  --n-estimators "${TABFM_M:-32}" \
  --out          "$CACHE_DIR/tabfm" \
  --checkpoint   "$TABFM_CKPT" \
  --seed         0 \
  --device       cuda

echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
