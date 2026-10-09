#!/bin/bash
# Re-run everything that is missing or provisional after the 2026-10-09 fixes.
#
#     cd ~/tabicl-ens   (or /scratch/j/joslin/tabicl-ens)
#     DRY=1 bash slurm/rerun_provisional.sh     # list what moves + the sbatch lines
#     bash slurm/rerun_provisional.sh           # quarantine + submit
#
# 1. QUARANTINE (moved to $CACHE_DIR/_stale/<stamp>/, never deleted)
#      TabICLv2  every <dataset>/S_*   (S_ctl, S_A7, S_A5_f*, S_A4_g*): written before
#                                       the MemberView fix, so they lack A1/A2 diversity
#      TabFM     classification plus_* splits with no agg.npy (plus_full, plus_noexpand,
#                                       plus_nocal): scoring needs upstream's own aggregate
# 2. SUBMIT (skip-complete, so only the moved pairs + anything never finished run)
#      TabICLv2  SET=shipplus  both tasks, all buckets
#      TabFM     SET=plus      classification (regression plus_* is replayed from meta)
#      TabFM     SET=shipplus  only with RETRY_SA7=1 (S_A7 produced no cells last time --
#                              read the diagnosis below before burning GPU hours on it)
# 3. Prints what is still incomplete.
#
# Knobs: MODELS="tabicl tabfm"  BUCKETS="small medium large"  RETRY_SA7=1  DRY=1
#        ACCOUNT=... GRES=...  CONF_TABICL_LARGE="03:00:00 96G 6"  (passed through)
# Afterwards:  python -m experiments.score_cells $CACHE_DIR/tabiclv2 OUT/tabicl --model tabicl
#              python -m experiments.score_cells $CACHE_DIR/tabfm    OUT/tabfm  --model tabfm
#              python -m experiments.report OUT REPORT_DIR
set -euo pipefail
cd "$(dirname "$0")/.."
source "slurm/_common.sh"
cc_activate

MODELS="${MODELS:-tabicl tabfm}"
BUCKETS="${BUCKETS:-small medium large}"
export BUCKETS
DRYFLAG=(); [ -n "${DRY:-}" ] && DRYFLAG=(--dry)

echo "=== 0. why did TabFM S_A7 produce no cells? ==="
if compgen -G "logs/*.out" > /dev/null; then
  echo "-- coalition lines mentioning S_A7 (count):"
  grep -h "S_A7" logs/*.out 2>/dev/null | sed -E 's/[0-9]+/N/g' | sort | uniq -c | sort -rn | head -5 || true
  echo "-- first error in an S_A7 TabFM log:"
  f=$(grep -l "S_A7" logs/tfm-tabf*.err 2>/dev/null | head -1 || true)
  [ -n "$f" ] && { echo "   $f"; grep -m3 -E "Error|Traceback|n/a" "$f" | sed 's/^/   /'; } || echo "   (none found)"
else
  echo "(no logs/ here -- run this from the directory you submitted from)"
fi
echo

echo "=== 1. quarantine provisional cells ==="
python -m experiments.quarantine "$CACHE_DIR" --models $MODELS ${DRYFLAG[@]+"${DRYFLAG[@]}"}
echo

echo "=== 2. submit ==="
export DRY="${DRY:-}"
if [[ " $MODELS " == *" tabicl "* ]]; then
  MODELS=tabicl SET=shipplus bash slurm/submit_all.sh
fi
if [[ " $MODELS " == *" tabfm "* ]]; then
  MODELS=tabfm TASKS=classification SET=plus bash slurm/submit_all.sh
  if [ "${RETRY_SA7:-0}" = 1 ]; then
    MODELS=tabfm SET=shipplus bash slurm/submit_all.sh
  else
    echo "[skip] TabFM S_A7 (RETRY_SA7=1 to resubmit once the cause is fixed)"
  fi
fi
echo

echo "=== 3. still incomplete after this submission (pairs; DRY shows the pre-run state) ==="
for m in $MODELS; do
  python -m experiments.grid plan --models $m --tasks classification regression \
      --set all --skip-complete --cache "$CACHE_DIR" 2>/dev/null | tail -n 15 || true
done
