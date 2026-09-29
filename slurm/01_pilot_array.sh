#!/bin/bash
#SBATCH --job-name=tfm-ens-pilot
#SBATCH --account=aip-ebrahimi            # <-- your CC allocation (def-/rrg-)
#SBATCH --gpus=h100:4                # Fir/Trillium; some clusters want
                                          #   --gpus-per-node=h100:1 instead
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00                   # short jobs schedule far sooner
#SBATCH --array=0-71%12                   # 8 datasets x 9 coalitions, 12 at once
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# Each array task = one (dataset, coalition) cell, looping over splits.
# run_cell.py skips any split whose meta.json exists, so a preempted or
# timed-out task can simply be resubmitted and resumes where it stopped.
#
# Submit:   sbatch slurm/01_pilot_array.sh
# Resume:   sbatch slurm/01_pilot_array.sh          (same command; idempotent)
# Requeue only the failures:
#   sbatch --array=$(sacct -j <JOBID> -n -X -o JobID,State \
#          | awk '$2!="COMPLETED"{split($1,a,"_"); printf "%s,", a[2]}' \
#          | sed 's/,$//') slurm/01_pilot_array.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SCRATCH/tabicl-ens}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf}"
export OPENML_CACHE_DIR="${OPENML_CACHE_DIR:-$PROJECT_ROOT/openml}"
export TABICL_REPO="${TABICL_REPO:-$PROJECT_ROOT/tabicl}"

# Fail loudly instead of hanging if something still tries to reach the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Determinism: the A6 control asserts exactly-zero spread, which only means
# something if the numerics are pinned. Keep these identical across every run.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 arrow/17.0.0
source "$PROJECT_ROOT/venv/bin/activate"

cd "$TABICL_REPO"

DATASETS=(blood-transfusion diabetes credit-g maternal_health_risk \
          MIC students_dropout Bioresponse Amazon_employee_access)
COALITIONS=(base A1 A2 A3 A1A2 A1A3 A2A3 A1A2A3 shipped)

# set SLURM_ARRAY_TASK_ID to 1 if slurm didn't provide it (e.g. for testing on login node)
SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

N_COAL=${#COALITIONS[@]}
D_IDX=$(( SLURM_ARRAY_TASK_ID / N_COAL ))
C_IDX=$(( SLURM_ARRAY_TASK_ID % N_COAL ))
DATASET=${DATASETS[$D_IDX]}
COALITION=${COALITIONS[$C_IDX]}

echo "=== task $SLURM_ARRAY_TASK_ID : $DATASET / $COALITION on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --verify on the first cell only: asserts captured members re-aggregate to
# predict_proba bitwise. Cheap, and it gates the whole campaign.
VERIFY=""
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then VERIFY="--verify"; fi

srun python -m experiments.run_cell \
  --dataset   "$DATASET" \
  --coalition "$COALITION" \
  --n-estimators 32 \
  --out       "$PROJECT_ROOT/cache" \
  --seed      0 \
  --device    cuda \
  $VERIFY

echo "=== task $SLURM_ARRAY_TASK_ID done ==="
