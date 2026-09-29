"""Worker: compute and cache the member tensor for one (dataset, coalition).

One SLURM array task = one cell. Splits are looped inside and each is written
independently, so a preempted task resumes at the first missing split instead
of restarting.

Usage:
    python -m experiments.run_cell --dataset credit-g --coalition A1 \
        --n-estimators 32 --out $SCRATCH/tfm-ens/cache

Outputs, per split:
    <out>/<dataset>/<coalition>/split<k>/members.npy   float16 (M, n_test, C)
    <out>/<dataset>/<coalition>/split<k>/meta.json
    <out>/<dataset>/<coalition>/split<k>/y_test.npy
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from experiments.capture import (
    MemberCapturingTabICLClassifier,
    verify_equivalence,
)
from experiments.coalitions import COALITIONS
from experiments.datasets import load_splits

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


def _git_sha(path: str = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def run_cell(
    dataset: str,
    coalition: str,
    n_estimators: int,
    out_root: Path,
    seed: int = 0,
    verify: bool = False,
    device: str | None = None,
) -> None:
    kwargs = COALITIONS[coalition]
    # Two repos now: the upstream clone that produced the numbers, and the
    # experiment code that orchestrated them. Record both.
    tabicl_sha = _git_sha(os.environ.get("TABICL_REPO", "."))
    code_sha = _git_sha(os.environ.get("CODE_ROOT", "."))

    for k, (X_tr, y_tr, X_te, y_te) in enumerate(load_splits(dataset)):
        cell = out_root / dataset / coalition / f"split{k}"
        if (cell / "meta.json").exists():
            print(f"[skip] {cell} already complete", flush=True)
            continue
        cell.mkdir(parents=True, exist_ok=True)

        clf = MemberCapturingTabICLClassifier(
            n_estimators=n_estimators,
            checkpoint_version=CHECKPOINT,
            random_state=seed,
            device=device,
            # Left at defaults on purpose: average_logits=True means members are
            # LOGITS, which keeps every aggregator available post hoc.
            # kv_cache stays off: the cached path does not take feature_shuffles,
            # so it may not preserve member semantics (EXPERIMENTS.md §5).
            **kwargs,
        )

        t_fit = time.perf_counter()
        clf.fit(X_tr, y_tr)
        t_fit = time.perf_counter() - t_fit

        if verify:
            verify_equivalence(clf, X_te[:64])

        t_pred = time.perf_counter()
        members = clf.predict_members(X_te)
        t_pred = time.perf_counter() - t_pred

        m_realised = int(members.shape[0])
        np.save(cell / "members.npy", members.astype(np.float16))
        np.save(cell / "y_test.npy", np.asarray(y_te))

        meta = {
            "dataset": dataset,
            "split": k,
            "coalition": coalition,
            "axis_kwargs": kwargs,
            "n_estimators_requested": n_estimators,
            "m_realised": m_realised,
            "truncated": m_realised < n_estimators,
            "average_logits": bool(clf.average_logits),
            "softmax_temperature": float(clf.softmax_temperature),
            "space": "logits" if clf.average_logits else "probabilities",
            "n_train": int(np.asarray(X_tr).shape[0]),
            "n_test": int(np.asarray(X_te).shape[0]),
            "n_features": int(np.asarray(X_tr).shape[1]),
            "n_classes": int(len(np.unique(y_tr))),
            "seed": seed,
            "checkpoint": CHECKPOINT,
            "tabicl_sha": tabicl_sha,
            "code_sha": code_sha,
            "fit_seconds": t_fit,
            "predict_seconds": t_pred,
            "seconds_per_member": t_pred / max(1, m_realised),
            "device": str(device or ("cuda" if torch.cuda.is_available() else "cpu")),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_mem_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
        }
        (cell / "meta.json").write_text(json.dumps(meta, indent=2))
        print(
            f"[done] {dataset}/{coalition}/split{k} "
            f"M={m_realised}/{n_estimators} "
            f"fit={t_fit:.1f}s pred={t_pred:.1f}s",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--coalition", required=True, choices=sorted(COALITIONS))
    p.add_argument("--n-estimators", type=int, default=32)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--verify", action="store_true",
                   help="assert capture == predict_proba before caching")
    a = p.parse_args()
    run_cell(a.dataset, a.coalition, a.n_estimators, a.out, a.seed, a.verify, a.device)


if __name__ == "__main__":
    main()
