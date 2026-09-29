#!/bin/bash
#SBATCH --job-name=tfm-ens-pilot
#SBATCH --account=def-CHANGEME            # <-- your CC allocation (def-/rrg-)
#SBATCH --gres=gpu:h100:1                 # some clusters want --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00                   # short jobs schedule far sooner
#SBATCH --array=0-71%12                   # 8 datasets x 9 coalitions, 12 concurrent
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# One array task = one (dataset, coalition) cell, looping over splits inside.
# run_cell.py skips any split whose meta.json exists, so this is idempotent:
# resubmitting after a preemption or timeout resumes at the first missing split.
#
# Submit from the checkout root so SLURM_SUBMIT_DIR resolves CODE_ROOT:
#     cd ~/tabicl-ens && mkdir -p logs && sbatch slurm/01_pilot_array.sh
# Or name it explicitly:
#     sbatch --export=ALL,CODE_ROOT=$HOME/tabicl-ens slurm/01_pilot_array.sh
#
# Requeue only the tasks that did not finish:
#     sbatch --array=$(sacct -j <JOBID> -n -X -o JobID,State \
#            | awk '$2!="COMPLETED"{split($1,a,"_"); printf "%s,", a[2]}' \
#            | sed 's/,$//') slurm/01_pilot_array.sh

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cc_activate
cc_banner

DATASETS=(blood-transfusion diabetes credit-g maternal_health_risk \
          MIC students_dropout Bioresponse Amazon_employee_access)
COALITIONS=(base A1 A2 A3 A1A2 A1A3 A2A3 A1A2A3 shipped)

N_COAL=${#COALITIONS[@]}
DATASET=${DATASETS[$(( SLURM_ARRAY_TASK_ID / N_COAL ))]}
COALITION=${COALITIONS[$(( SLURM_ARRAY_TASK_ID % N_COAL ))]}

echo "=== task ${SLURM_ARRAY_TASK_ID}: $DATASET / $COALITION ==="

# --verify on the first cell only: asserts captured members re-aggregate to
# predict_proba bitwise. Cheap, and it gates the whole campaign.
VERIFY=""
[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ] && VERIFY="--verify"

srun python -m experiments.run_cell \
  --dataset      "$DATASET" \
  --coalition    "$COALITION" \
  --n-estimators 32 \
  --out          "$CACHE_DIR" \
  --seed         0 \
  --device       cuda \
  $VERIFY

echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
