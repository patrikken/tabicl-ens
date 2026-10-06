#!/bin/bash
# Submit the A4 (feature-side subsampling) campaign, one array per size bucket.
#
#     cd ~/tabicl-ens && bash slurm/submit_a4.sh                 # sweep, small
#     bash slurm/submit_a4.sh small medium                       # more buckets
#     A4_SET=variants bash slurm/submit_a4.sh small              # the mode contrast
#     DRY=1 bash slurm/submit_a4.sh small medium large           # preview
#
# RUN THE SWEEP FIRST. A4_g25..g90 are the experiment: they mirror the A5
# fractions exactly, so the two recovery curves -- diversity gained per unit of
# member quality destroyed -- land on the same x-axis and can be read against
# each other point for point. That comparison is the whole reason A4 exists:
#
#   recovery < 1, like A5  -> the exchange rate is about destroying information
#                             in general, and the study closes with one law.
#   recovery > 1, like A1-A3 -> rows are special, and feature bagging is a
#                             wrapper improvement no TFM currently ships.
#
# A4_SET=variants adds the mode contrast (round-robin, importance-weighted) and
# column-LOFO. Only worth the compute once the sweep says which side A4 is on.
#
# COST: like A5, M members = M fits. Not every dataset is eligible -- A4 needs
# >= 8 columns, and column-LOFO needs d >= 2M -- so the grid is ragged and the
# plan below prints the real task count rather than datasets x coalitions.

set -euo pipefail
source "slurm/_common.sh"
cc_activate

A4_SET="${A4_SET:-variants}"
BUCKETS=(small medium large)

#          time      mem  concurrent
CONF_small="00:20:00 48G 16"
CONF_medium="00:40:00 64G 16"
CONF_large="00:20:00 96G 16"

export SPLIT_BACKEND=tabarena
mkdir -p logs

echo "=== A4 plan (set=$A4_SET, M=${A4_M:-16} members = ${A4_M:-16} fits per cell) ==="
A4_SET="$A4_SET" python - <<'PY'
import os
from experiments.coalitions import (A4_SWEEP, A4_VARIANTS, A4_MIN_FEATURES,
                                    a4_applicable)
from experiments.datasets import tabarena_datasets, dataset_info
sets = {"sweep": A4_SWEEP, "variants": A4_VARIANTS, "all": A4_SWEEP + A4_VARIANTS}
coals = sets[os.environ["A4_SET"]]
ds = tabarena_datasets()
skipped = [d for d in ds if dataset_info(d)["n_features"] < A4_MIN_FEATURES]
for b in ("small", "medium", "large"):
    sel = [d for d in ds if dataset_info(d)["bucket"] == b]
    pairs = [(d, c) for d in sel for c in coals
             if a4_applicable(c, dataset_info(d)["n_features"])]
    cells = sum(dataset_info(d)["n_splits"] for d, _ in pairs)
    print(f"  {b:7s} {len(sel):2d} datasets  {len(pairs):3d} array tasks  {cells:5d} cells")
print("\n  coalitions: " + " ".join(coals))
names = ["%s(d=%d)" % (d, dataset_info(d)["n_features"]) for d in skipped]
print("  excluded for d < %d: %s" % (A4_MIN_FEATURES, ", ".join(names) or "none"))
PY
echo

for BUCKET in "${BUCKETS[@]}"; do
  N_TASK=$(BUCKET=$BUCKET A4_SET=$A4_SET python -c "
import os
from experiments.coalitions import A4_SWEEP, A4_VARIANTS, a4_applicable
from experiments.datasets import tabarena_datasets, dataset_info
sets={'sweep':A4_SWEEP,'variants':A4_VARIANTS,'all':A4_SWEEP+A4_VARIANTS}
coals=sets[os.environ['A4_SET']]; b=os.environ['BUCKET']
print(sum(1 for d in tabarena_datasets()
          if dataset_info(d)['bucket']==b
          for c in coals if a4_applicable(c, dataset_info(d)['n_features'])))
")
  [ "$N_TASK" -eq 0 ] && { echo "bucket $BUCKET: no applicable tasks"; continue; }

  eval "CONF=\$CONF_$BUCKET"
  read -r TIME MEM CONC <<<"$CONF"
  MAX=$(( N_TASK - 1 ))

  CMD=(sbatch --job-name="tfm-a4-$BUCKET-$A4_SET"
       --array="0-${MAX}%${CONC}" --time="$TIME" --mem="$MEM"
       --export=ALL,BUCKET="$BUCKET",A4_SET="$A4_SET",CODE_ROOT="$CODE_ROOT",A4_M="${A4_M:-}"
       slurm/07_a4_array.sh)
  echo "bucket $BUCKET: $N_TASK tasks -> array 0-${MAX}%${CONC}, time=$TIME"
  if [ -n "${DRY:-}" ]; then printf '  DRY: %s\n' "${CMD[*]}"; else "${CMD[@]}"; fi
done

cat <<'EOF'

After the small bucket, read the real cost before scaling, and sanity-check the
coverage diagnostics the wrapper records:

  python - <<'PY'
import json, glob, os, statistics as st
t=[json.load(open(f)) for f in glob.glob(f"{os.environ['CACHE_DIR']}/*/A4_g50/*/meta.json")]
if t:
    print("A4_g50 per-member s:", round(st.median(m["seconds_per_member"] for m in t),3))
    print("       per-cell total:", round(st.median(m["fit_seconds"]+m["predict_seconds"] for m in t),1))
    print("   mean subset size  :", round(st.mean(m["subset_size_mean"] for m in t),1))
    print("   columns never used:", max(m["n_unused_columns"] for m in t), "(max over cells)")
    print("   overlap realised  :", round(st.mean(m["pairwise_overlap_mean"] for m in t),3),
          "expected", round(st.mean(m["pairwise_overlap_expected"] for m in t),3))
PY

`columns never used` > 0 on the random draws is expected and is exactly what
A4rr_g50 (balanced dealing, same subset size) is there to control for.
EOF
