#!/bin/bash
#SBATCH --job-name=tfm-ens-a6
#SBATCH --account=def-CHANGEME
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Phase 0 gates. Run this BEFORE the pilot array; do not proceed until it passes.
#
#     cd ~/tabicl-ens && mkdir -p logs && sbatch slurm/02_a6_control.sh
#
# With every axis 'none' and one normalisation method, tabicl's Shuffler returns
# a single identity pattern, so all members are the same view and the forward
# pass is deterministic. Seed spread must therefore be EXACTLY zero. Anything
# else means numerical nondeterminism (TF32, AMP, cuBLAS reduction order) is
# leaking in, and it would surface as spurious "diversity" in every axis
# measurement in the study.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cc_activate
cc_banner

srun python - <<'PY'
import numpy as np
from experiments.capture import MemberCapturingTabICLClassifier, verify_equivalence
from experiments.coalitions import COALITIONS
from experiments.datasets import load_splits

kw = COALITIONS["A6_control"]
X_tr, y_tr, X_te, y_te = next(load_splits("diabetes"))

# Gate 0.1 - capture must reproduce predict_proba bitwise
clf = MemberCapturingTabICLClassifier(n_estimators=8, random_state=0, device="cuda", **kw)
clf.fit(X_tr, y_tr)
verify_equivalence(clf, X_te)
print("[ok] gate 0.1  capture == predict_proba (bitwise)")

# Gate 0.3 - every axis off => exactly one realised member
m = clf.m_realised_
print(f"[{'ok' if m == 1 else 'FAIL'}] gate 0.3  m_realised = {m} (expected 1)")
assert m == 1, "all axes off should yield exactly one member; check coalitions.py"

# Gate 0.2 - seeds must not move predictions at all
ref = None
for seed in range(4):
    c = MemberCapturingTabICLClassifier(n_estimators=8, random_state=seed,
                                        device="cuda", **kw)
    c.fit(X_tr, y_tr)
    p = c.predict_members(X_te)
    if ref is None:
        ref = p
        continue
    d = float(np.abs(p - ref).max())
    print(f"        seed {seed}: max|delta| = {d:.3e}")
    assert d == 0.0, (
        f"NONDETERMINISM: seed {seed} moved predictions by {d:.3e}. Diversity "
        "diagnostics would be inflated by numerics. Pin TF32/AMP/cuBLAS before "
        "running the campaign."
    )
print("[ok] gate 0.2  determinism")

# Gate 0.4 - float16 cache round-trip must not change decisions
p32 = ref
p16 = p32.astype(np.float16).astype(np.float32)
assert np.array_equal(p32.argmax(-1), p16.argmax(-1)), "float16 flips argmax"
print(f"[ok] gate 0.4  float16 round-trip, max|delta| = {np.abs(p32-p16).max():.3e}")

print("\n[PASS] Phase 0 gates")
PY
