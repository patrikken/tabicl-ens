#!/bin/bash
# Submit EVERYTHING in one go: TabICLv2 and TabFM, classification and regression,
# every coalition, every size bucket.
#
#     cd ~/tabicl-ens
#     bash slurm/prefetch_all.sh            # login node, once
#     DRY=1 bash slurm/submit_all.sh        # preview: plan table + sbatch lines
#     bash slurm/submit_all.sh              # go
#
# Narrow it with env vars (all optional):
#     MODELS="tabfm"                  tabicl | tabfm | both (default both)
#     TASKS="regression"              classification | regression | both
#     SET=native                      coalition set: TabICL native|a4|a5|lofo|all,
#                                     TabFM axes|withhold|expand|plus|new|all
#                                     (default all; each model keeps its own names)
#     BUCKETS="small medium"          size buckets (default all three)
#     SKIP_COMPLETE=0                 also resubmit pairs whose splits all exist
#     ACCOUNT=def-xxx GRES=gpu:h100:1 override the #SBATCH account / gres
#     TABFM_M=32                      TabFM members (forward passes) per cell
#     CONF_TABICL_MEDIUM="02:00:00 64G 10"   override one "time mem concurrent"
#
# IDEMPOTENT. A pair whose splits all have a meta.json is skipped at submit time
# and a half-done pair resumes at its first missing split, so after any time-out
# just run this script again. The array-index -> pair mapping is frozen in a plan
# file under $PROJECT_ROOT/plans (see experiments/grid.py), so cells finishing
# while the array runs cannot shift it.
#
# Same-time submission does NOT mean same-time execution: --array=0-N%K caps how
# many tasks of each array run at once, so the 12 arrays share the queue
# politely. Raise K per bucket via CONF_* if the queue is empty.

set -euo pipefail
source "slurm/_common.sh"
cc_activate

MODELS_ARG="${MODELS:-tabicl tabfm}"; [ "$MODELS_ARG" = both ] && MODELS_ARG="tabicl tabfm"
TASKS_ARG="${TASKS:-classification regression}"; [ "$TASKS_ARG" = both ] && TASKS_ARG="classification regression"
BUCKETS_ARG="${BUCKETS:-small medium large}"
SET="${SET:-all}"
SKIP_COMPLETE="${SKIP_COMPLETE:-1}"
export TABFM_CKPT="${TABFM_CKPT:-$PROJECT_ROOT/tabfm-ckpt}"
export TABFM_M="${TABFM_M:-32}"
export SPLIT_BACKEND=tabarena

#                         time      mem  concurrent   (edit freely; most cells are 20-40 min)
declare -A CONF=(
  [tabicl_small]="00:40:00 48G 16"
  [tabicl_medium]="01:30:00 64G 12"
  [tabicl_large]="03:00:00 96G 6"
  [tabfm_small]="00:25:00 64G 20"
  [tabfm_medium]="00:50:00 96G 15"
  [tabfm_large]="01:00:00 128G 10"
)

mkdir -p logs "$PROJECT_ROOT/plans"
STAMP="$(date +%Y%m%d-%H%M%S)"
SKIP_FLAG=(); [ "$SKIP_COMPLETE" = 1 ] && SKIP_FLAG=(--skip-complete)

# ---- preflight: fail here, not 30 minutes into a job ------------------------
FAIL=0
need_reg=0; for t in $TASKS_ARG; do [ "$t" = regression ] && need_reg=1; done
if ! python - <<PY
from experiments.datasets import tabarena_datasets
import sys
n_reg = len(tabarena_datasets("regression")); n_cls = len(tabarena_datasets("classification"))
print(f"  manifest: {n_cls} classification + {n_reg} regression datasets")
sys.exit(1 if ($need_reg and n_reg == 0) else 0)
PY
then
  echo "FATAL: the manifest has no regression datasets. On a login node run:" >&2
  echo "    python -m experiments.prepare_tabarena --out \$TABARENA_DIR --include-regression" >&2
  echo "  (or: bash slurm/prefetch_all.sh)" >&2; FAIL=1
fi
for m in $MODELS_ARG; do
  if [ "$m" = tabfm ]; then
    for head in $( [[ " $TASKS_ARG " == *classification* ]] && echo classification; [ $need_reg = 1 ] && echo regression ); do
      [ -d "$TABFM_CKPT/$head" ] || { echo "FATAL: TabFM $head checkpoint missing: $TABFM_CKPT/$head (bash slurm/prefetch_all.sh)" >&2; FAIL=1; }
    done
  fi
  if [ "$m" = tabicl ] && [ $need_reg = 1 ]; then
    python - <<'PY' || { echo "FATAL: TabICLv2 regressor checkpoint not in the HF cache (bash slurm/prefetch_all.sh)" >&2; FAIL=1; }
from huggingface_hub import try_to_load_from_cache
import sys
p = try_to_load_from_cache("jingang/TabICL", "tabicl-regressor-v2-20260212.ckpt")
sys.exit(0 if isinstance(p, str) else 1)
PY
  fi
done
[ "$FAIL" = 1 ] && { [ -z "${DRY:-}" ] && exit 1 || echo "(DRY run: continuing despite the above)"; }

echo "=== campaign plan (set=$SET, skip_complete=$SKIP_COMPLETE) ==="
python -m experiments.grid plan --models $MODELS_ARG --tasks $TASKS_ARG \
  --set "$SET" ${SKIP_FLAG[@]+"${SKIP_FLAG[@]}"} --cache "$CACHE_DIR"
echo

N_ARRAYS=0; N_TASKS=0
for MODEL in $MODELS_ARG; do
  for TASK in $TASKS_ARG; do
    for BUCKET in $BUCKETS_ARG; do
      PLAN="$PROJECT_ROOT/plans/${STAMP}_${MODEL}_${TASK}_${BUCKET}.txt"
      python -m experiments.grid list --model "$MODEL" --task "$TASK" --bucket "$BUCKET" \
        --set "$SET" ${SKIP_FLAG[@]+"${SKIP_FLAG[@]}"} --cache "$CACHE_DIR" > "$PLAN"
      N=$(wc -l < "$PLAN")
      if [ "$N" -eq 0 ]; then rm -f "$PLAN"; echo "[skip] $MODEL/$TASK/$BUCKET: nothing to do"; continue; fi

      OVR="CONF_$(echo "${MODEL}_${BUCKET}" | tr '[:lower:]' '[:upper:]')"
      read -r TIME MEM CONC <<<"${!OVR:-${CONF[${MODEL}_${BUCKET}]}}"
      CMD=(sbatch --job-name="tfm-${MODEL:0:4}-${TASK:0:3}-${BUCKET}"
           --array="0-$((N - 1))%${CONC}" --time="$TIME" --mem="$MEM"
           --export=ALL,MODEL="$MODEL",TASK_KIND="$TASK",BUCKET="$BUCKET",PLAN="$PLAN",CODE_ROOT="$CODE_ROOT",TABFM_CKPT="$TABFM_CKPT",TABFM_M="$TABFM_M")
      [ -n "${ACCOUNT:-}" ] && CMD+=(--account="$ACCOUNT")
      [ -n "${GRES:-}" ] && CMD+=(--gres="$GRES")
      CMD+=(slurm/09_all_array.sh)

      printf '%-7s %-15s %-7s %5d tasks  time=%s mem=%s max-parallel=%s\n' "$MODEL" "$TASK" "$BUCKET" "$N" "$TIME" "$MEM" "$CONC"
      if [ -n "${DRY:-}" ]; then printf '   DRY: %s\n' "${CMD[*]}"; else "${CMD[@]}"; fi
      N_ARRAYS=$((N_ARRAYS + 1)); N_TASKS=$((N_TASKS + N))
    done
  done
done

cat <<EOF

$N_ARRAYS arrays, $N_TASKS array tasks${DRY:+ (DRY: nothing submitted)}.

Monitor:     squeue -u \$USER
Progress:    python -m experiments.grid plan --cache \$CACHE_DIR --skip-complete
Resume:      bash slurm/submit_all.sh          (skips finished pairs; time-outs resume)
Failures:    grep -L "task .* done" logs/tfm-*.out
Plan files:  \$PROJECT_ROOT/plans/${STAMP}_*.txt  (array index -> pair, frozen)

Caches:      TabICLv2 -> \$CACHE_DIR            TabFM -> \$CACHE_DIR/tabfm
EOF
