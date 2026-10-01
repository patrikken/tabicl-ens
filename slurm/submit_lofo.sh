#!/bin/bash
# Submit the LOFO sweep.
#
#     cd ~/tabicl-ens && bash slurm/submit_lofo.sh            # small bucket
#     bash slurm/submit_lofo.sh small medium
#     DRY=1 bash slurm/submit_lofo.sh small
#
# LOFO traces f = 1 - 1/M as M varies, which is the one path through the (M, f)
# plane the fixed-M fraction sweep could not reach:
#
#     M=8 -> f=0.875 | M=16 -> 0.9375 | M=32 -> 0.969 | M=64 -> 0.984
#
# Cost scales with M *and* with the per-fit size, which also grows with M, so
# M=64 is roughly 4x M=16 per cell. Small bucket first.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cc_activate

N_COAL=4
BUCKETS=("${@:-small}")

#          time      mem  concurrent
CONF_small="12:00:00 48G 10"
CONF_medium="24:00:00 64G 5"
CONF_large="24:00:00 96G 2"

export SPLIT_BACKEND=tabarena
mkdir -p logs

echo "=== LOFO plan ==="
python - <<'PY'
from experiments.datasets import tabarena_datasets, dataset_info
for b in ("small","medium","large"):
    sel=[d for d in tabarena_datasets() if dataset_info(d)["bucket"]==b]
    cells=sum(dataset_info(d)["n_splits"] for d in sel)*4
    fits=sum(dataset_info(d)["n_splits"] for d in sel)*(8+16+32+64)
    print(f"  {b:7s} {len(sel):2d} datasets  {len(sel)*4:3d} tasks  {cells:5d} cells  ~{fits:,} fits")
PY
echo

for BUCKET in "${BUCKETS[@]}"; do
  N_DS=$(BUCKET=$BUCKET python -c "
import os
from experiments.datasets import tabarena_datasets, dataset_info
b=os.environ['BUCKET']
print(sum(1 for d in tabarena_datasets() if dataset_info(d)['bucket']==b))
")
  [ "$N_DS" -eq 0 ] && { echo "bucket $BUCKET: empty"; continue; }
  eval "CONF=\$CONF_$BUCKET"; read -r TIME MEM CONC <<<"$CONF"
  MAX=$(( N_DS * N_COAL - 1 ))
  CMD=(sbatch --job-name="tfm-lofo-$BUCKET"
       --array="0-${MAX}%${CONC}" --time="$TIME" --mem="$MEM"
       --export=ALL,BUCKET="$BUCKET",CODE_ROOT="$CODE_ROOT"
       slurm/06_lofo_array.sh)
  echo "bucket $BUCKET: $N_DS datasets -> array 0-${MAX}%${CONC}, time=$TIME"
  if [ -n "${DRY:-}" ]; then printf '  DRY: %s\n' "${CMD[*]}"; else "${CMD[@]}"; fi
done

cat <<'EOF'

The question this answers: our fraction sweep found net gain rising monotonically
as f -> 1, with A5_f90 the best point. LOFO pushes f higher still (0.9375 ->
0.984) while raising M. If net gain keeps rising, the optimum is at the f -> 1
limit and context perturbation is simply a weak version of doing nothing. If it
turns over somewhere in M=16..64, there is an interior optimum and the earlier
sweep missed it.
EOF
