#!/bin/bash
# Submit the A5 (context-side) campaign, one array per size bucket.
#
#     cd ~/tabicl-ens && bash slurm/submit_a5.sh          # small bucket only
#     bash slurm/submit_a5.sh small medium                # more buckets
#     DRY=1 bash slurm/submit_a5.sh all                   # preview everything
#
# START WITH `small`. A5 needs M fits per cell rather than M forward passes
# through one fit, so it is roughly an order of magnitude dearer per member than
# the feature-side axes. Run the small bucket, read the real per-fit timings out
# of the cached meta.json, and only then decide whether medium/large are worth
# their compute.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cc_activate

N_COAL=5                      # A5_f25 f50 f75 f90 + A5bal_f50
BUCKETS=("${@:-small}")

#          time      mem  concurrent
CONF_small="06:00:00 48G 12"
CONF_medium="16:00:00 64G 6"
CONF_large="24:00:00 96G 3"

export SPLIT_BACKEND=tabarena
mkdir -p logs

echo "=== A5 plan (M=${A5_M:-16} members = ${A5_M:-16} fits per cell) ==="
python - <<'PY'
from experiments.datasets import tabarena_datasets, dataset_info
from collections import Counter
ds = tabarena_datasets()
c = Counter(dataset_info(d)["bucket"] for d in ds)
for b in ("small", "medium", "large"):
    sel = [d for d in ds if dataset_info(d)["bucket"] == b]
    cells = sum(dataset_info(d)["n_splits"] for d in sel) * 5
    print(f"  {b:7s} {len(sel):2d} datasets  {len(sel)*5:3d} array tasks  {cells:5d} cells")
PY
echo

for BUCKET in "${BUCKETS[@]}"; do
  [ "$BUCKET" = "all" ] && BUCKET=small   # 'all' still starts with small
  N_DS=$(BUCKET=$BUCKET python -c "
import os
from experiments.datasets import tabarena_datasets, dataset_info
b=os.environ['BUCKET']
print(sum(1 for d in tabarena_datasets() if dataset_info(d)['bucket']==b))
")
  [ "$N_DS" -eq 0 ] && { echo "bucket $BUCKET: empty"; continue; }

  eval "CONF=\$CONF_$BUCKET"
  read -r TIME MEM CONC <<<"$CONF"
  MAX=$(( N_DS * N_COAL - 1 ))

  CMD=(sbatch --job-name="tfm-a5-$BUCKET"
       --array="0-${MAX}%${CONC}" --time="$TIME" --mem="$MEM"
       --export=ALL,BUCKET="$BUCKET",CODE_ROOT="$CODE_ROOT",A5_M="${A5_M:-16}"
       slurm/05_a5_array.sh)
  echo "bucket $BUCKET: $N_DS datasets -> array 0-${MAX}%${CONC}, time=$TIME"
  if [ -n "${DRY:-}" ]; then printf '  DRY: %s\n' "${CMD[*]}"; else "${CMD[@]}"; fi
done

cat <<'EOF'

After the small bucket finishes, read the real cost before scaling up:
  python - <<'PY'
import json, glob, statistics as st
t=[json.load(open(f)) for f in glob.glob(f"{__import__('os').environ['CACHE_DIR']}/*/A5_f50/*/meta.json")]
if t:
    print("A5_f50 per-member seconds:", round(st.median(m["seconds_per_member"] for m in t),3))
    print("            per-cell total:", round(st.median(m["fit_seconds"]+m["predict_seconds"] for m in t),1))
PY
EOF
