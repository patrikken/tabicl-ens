#!/bin/bash
# Submit the full TabArena campaign, one array per size bucket.
#
#     cd ~/tabicl-ens && bash slurm/submit_full.sh
#     bash slurm/submit_full.sh small          # one bucket
#     DRY=1 bash slurm/submit_full.sh          # print without submitting
#
# Buckets exist because TabArena spans ~700 to ~150k rows. A single --time
# either wastes queue priority on the small datasets or kills the large ones.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cc_activate

N_COAL=9          # base A1 A2 A3 A1A2 A1A3 A2A3 A1A2A3 shipped
ONLY="${1:-}"

#         bucket  time       mem    concurrent
CONF_small="03:00:00 48G 16"
CONF_medium="08:00:00 64G 8"
CONF_large="16:00:00 96G 4"

export SPLIT_BACKEND=tabarena
mkdir -p logs

echo "=== campaign plan ==="
python - <<'PY'
from experiments.datasets import tabarena_datasets, dataset_info
from collections import Counter
ds = tabarena_datasets()
c = Counter(dataset_info(d)["bucket"] for d in ds)
tot_cells = sum(dataset_info(d)["n_splits"] for d in ds)
print(f"  datasets      : {len(ds)}")
print(f"  by bucket     : {dict(c)}")
print(f"  dataset-splits: {tot_cells}")
print(f"  array tasks   : {len(ds)*9}  (x9 coalitions)")
print(f"  cells total   : {tot_cells*9}")
PY
echo

for BUCKET in small medium large; do
  [ -n "$ONLY" ] && [ "$ONLY" != "$BUCKET" ] && continue

  N_DS=$(BUCKET=$BUCKET python -c "
import os
from experiments.datasets import tabarena_datasets, dataset_info
b=os.environ['BUCKET']
print(sum(1 for d in tabarena_datasets() if dataset_info(d)['bucket']==b))
")
  [ "$N_DS" -eq 0 ] && { echo "bucket $BUCKET: empty, skipping"; continue; }

  eval "CONF=\$CONF_$BUCKET"
  read -r TIME MEM CONC <<<"$CONF"
  MAX=$(( N_DS * N_COAL - 1 ))

  CMD=(sbatch --job-name="tfm-ens-$BUCKET"
       --array="0-${MAX}%${CONC}"
       --time="$TIME" --mem="$MEM"
       --export=ALL,BUCKET="$BUCKET",CODE_ROOT="$CODE_ROOT"
       slurm/03_full_array.sh)

  echo "bucket $BUCKET: $N_DS datasets -> array 0-${MAX}%${CONC}, time=$TIME mem=$MEM"
  if [ -n "${DRY:-}" ]; then
    printf '  DRY: %s\n' "${CMD[*]}"
  else
    "${CMD[@]}"
  fi
done

cat <<'EOF'

Monitor:   squeue -u $USER
Progress:  find $CACHE_DIR -name meta.json | wc -l
Requeue failures for one bucket:
  sbatch --array=$(sacct -j <JOBID> -n -X -o JobID,State \
         | awk '$2!="COMPLETED"{split($1,a,"_"); printf "%s,", a[2]}' | sed 's/,$//') \
         --export=ALL,BUCKET=<bucket> slurm/03_full_array.sh
EOF
