#!/bin/bash
#SBATCH --job-name=tfm-ens-a6
#SBATCH --account=def-CHANGEME
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
#
# Determinism gate. Run this BEFORE the pilot array and do not proceed until it
# passes.
#
# With every axis set to 'none' and a single normalisation method, tabicl's
# Shuffler returns one identity pattern, so all members are the same view and
# the forward pass is deterministic. Spread across seeds must therefore be
# EXACTLY zero. Any nonzero spread means numerical nondeterminism (TF32, AMP,
# cuBLAS reduction order) is leaking in -- which would show up as spurious
# "diversity" in every axis measurement and silently inflate the results.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SCRATCH/tfm-ens}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf}"
export OPENML_CACHE_DIR="${OPENML_CACHE_DIR:-$PROJECT_ROOT/openml}"
export TABICL_REPO="${TABICL_REPO:-$PROJECT_ROOT/tabicl}"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 arrow/17.0.0
source "$PROJECT_ROOT/venv/bin/activate"
cd "$TABICL_REPO"

srun python - <<'PY'
import numpy as np
from experiments.capture import MemberCapturingTabICLClassifier, verify_equivalence
from experiments.coalitions import COALITIONS
from experiments.datasets import load_splits

kw = COALITIONS["A6_control"]
X_tr, y_tr, X_te, y_te = next(load_splits("diabetes"))

# 1) capture must reproduce predict_proba bitwise
clf = MemberCapturingTabICLClassifier(n_estimators=8, random_state=0, device="cuda", **kw)
clf.fit(X_tr, y_tr)
verify_equivalence(clf, X_te)
print("[ok] capture == predict_proba (bitwise)")

# 2) with every axis off there must be exactly one realised member
print(f"[info] m_realised = {clf.m_realised_} (expected 1)")

# 3) seeds must not move the prediction at all
ref = None
for seed in range(4):
    c = MemberCapturingTabICLClassifier(n_estimators=8, random_state=seed,
                                        device="cuda", **kw)
    c.fit(X_tr, y_tr)
    p = c.predict_members(X_te)
    if ref is None:
        ref = p
    else:
        d = np.abs(p - ref).max()
        print(f"[info] seed {seed}: max|delta| = {d:.3e}")
        assert d == 0.0, (
            f"NONDETERMINISM: seed {seed} moved predictions by {d:.3e}. "
            "Diversity diagnostics would be inflated. Pin TF32/AMP/cuBLAS "
            "before running the campaign."
        )
print("[PASS] A6 determinism control")
PY
