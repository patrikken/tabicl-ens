#!/bin/bash
# Submit the TabFM campaign, one array per size bucket.
#
#     cd ~/tabicl-ens && bash slurm/submit_tabfm.sh small             # the axes set
#     TABFM_SET=withhold bash slurm/submit_tabfm.sh small medium
#     TABFM_SET=plus     bash slurm/submit_tabfm.sh small
#     DRY=1 bash slurm/submit_tabfm.sh small medium large
#
# RUN `axes` FIRST. It replays the TabICLv2 feature-side campaign on a second
# model, which is the transfer test every reviewer will ask for and the only
# thing that makes the recovery law a claim about in-context learners rather
# than about one checkpoint.
#
# Then, in order of what they buy:
#   expand    -- A8, the one regime the law has no data on (members GAIN columns)
#   plus      -- takes TabFM+'s unattributed +69.4 Elo apart into its four parts
#   withhold  -- A4/A5 at matched fractions; does the recovery curve transfer?
#   new       -- A7 categorical relabelling
#
# PREREQUISITE, once, on a LOGIN NODE:
#   python - <<'PY'
#   from huggingface_hub import snapshot_download; import os
#   snapshot_download("google/tabfm-1.0.0-pytorch",
#                     local_dir=os.environ["PROJECT_ROOT"]+"/tabfm-ckpt",
#                     allow_patterns=["classification/**","config.json"])
#   PY

set -euo pipefail
source "slurm/_common.sh"
cc_activate

SET="${TABFM_SET:-axes}"
BUCKETS=("${@:-small}")

#          time      mem  concurrent
CONF_small="00:25:00 64G 20"
CONF_medium="00:50:00 96G 15"
CONF_large="00:60:00 128G 10"

export SPLIT_BACKEND=tabarena
export TABFM_CKPT="${TABFM_CKPT:-$PROJECT_ROOT/tabfm-ckpt}"
export TABARENA_DIR="${TABARENA_DIR:-$PROJECT_ROOT/tabarena-splits}"
mkdir -p logs

if [ ! -d "$TABFM_CKPT" ]; then
  echo "WARNING: no checkpoint at $TABFM_CKPT -- snapshot it on a login node first (see header)."
fi

echo "=== TabFM plan (set=$SET, M=${TABFM_M:-32} members = ${TABFM_M:-32} forward passes, ONE fit) ==="
TABFM_SET="$SET" python - <<'PY'
import os
from experiments.tabfm_sets import coalition_set, SETS
from experiments.datasets import tabarena_datasets, dataset_info
s = os.environ["TABFM_SET"]
print(f"  coalitions: {' '.join(SETS[s])}\n")
ds = tabarena_datasets()
for b in ("small", "medium", "large"):
    sel = [d for d in ds if dataset_info(d)["bucket"] == b]
    pairs = [(d, c) for d in sel
             for c in coalition_set(s, dataset_info(d)["n_features"])]
    cells = sum(dataset_info(d)["n_splits"] for d, _ in pairs)
    print(f"  {b:7s} {len(sel):2d} datasets  {len(pairs):3d} array tasks  {cells:5d} cells")
PY
echo

for BUCKET in "${BUCKETS[@]}"; do
  N_TASK=$(BUCKET=$BUCKET TABFM_SET=$SET python -c "
import os
from experiments.tabfm_sets import coalition_set
from experiments.datasets import tabarena_datasets, dataset_info
b=os.environ['BUCKET']; s=os.environ['TABFM_SET']
print(sum(len(coalition_set(s, dataset_info(d)['n_features']))
          for d in tabarena_datasets() if dataset_info(d)['bucket']==b))
")
  [ "$N_TASK" -eq 0 ] && { echo "bucket $BUCKET: no tasks"; continue; }
  eval "CONF=\$CONF_$BUCKET"; read -r TIME MEM CONC <<<"$CONF"
  MAX=$(( N_TASK - 1 ))
  CMD=(sbatch --job-name="tfm-tabfm-$BUCKET-$SET"
       --array="0-${MAX}%${CONC}" --time="$TIME" --mem="$MEM"
       --export=ALL,BUCKET="$BUCKET",TABFM_SET="$SET",CODE_ROOT="$CODE_ROOT",TABFM_CKPT="$TABFM_CKPT",TABFM_M="${TABFM_M:-32}"
       slurm/08_tabfm_array.sh)
  echo "bucket $BUCKET: $N_TASK tasks -> array 0-${MAX}%${CONC}, time=$TIME"
  if [ -n "${DRY:-}" ]; then printf '  DRY: %s\n' "${CMD[*]}"; else "${CMD[@]}"; fi
done

cat <<'EOF'

The first cell of every task asserts that captured-then-aggregated equals
predict_proba bitwise. If that assertion fires, upstream moved the
member/aggregate boundary and the cache would be silently wrong -- stop and
re-read classifier_and_regressor.py::_process_logits before running anything else.
EOF
