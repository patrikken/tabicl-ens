"""Worker: cache the member tensor for one (dataset, coalition) on TabFM.

Mirrors ``experiments/run_cell.py`` and writes the SAME cache layout, so the
whole downstream analysis -- scoring, M_eff, the recovery decomposition, Elo --
runs unchanged over both models and both tasks:

    <out>/<dataset>/<coalition>/split<k>/members.npy   float32
                                  classification: (M, n_test, C) LOGITS
                                  regression:     (M, n_test)    original-scale predictions
    <out>/<dataset>/<coalition>/split<k>/meta.json
    <out>/<dataset>/<coalition>/split<k>/y_test.npy

float32, NOT float16: TabFM logits reach ~1e3, where float16 spacing is 0.5-1 and
ties the ranking.

Usage (compute node, offline checkpoint):

    python -m experiments.run_cell_tabfm --dataset credit-g --coalition shipped \\
        --out $CACHE_DIR/tabfm --checkpoint $PROJECT_ROOT/tabfm-ckpt --device cuda

The task (classification | regression) comes from the dataset's manifest entry:
it picks the checkpoint subfolder, the estimator, and which coalition table
applies. The 6.5 GB checkpoint is loaded ONCE per process and reused across
every split, so an array task should cover a whole (dataset, coalition) cell.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from experiments.datasets import dataset_task, load_splits, split_severity
from experiments.tabfm_capture import (CHECKPOINT_REPO, TABFM_COALITIONS,
                                       TABFM_DEFAULT_M, TABFM_REG_COALITIONS,
                                       axis_side, destroys_information,
                                       is_expansion, load_model,
                                       make_capturing_classifier,
                                       make_capturing_regressor,
                                       member_metadata, resolve_fracs,
                                       tabfm_coalitions, verify_equivalence,
                                       verify_equivalence_regression)


def _git_sha(path: str = ".") -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:                                        # noqa: BLE001
        return "unknown"


def run_cell(dataset: str, coalition: str, n_estimators: int, out_root: Path,
             checkpoint: str | None, seed: int = 0, device: str = "cuda",
             verify_first: bool = True) -> None:
    task = dataset_task(dataset)
    reg = task == "regression"
    table = tabfm_coalitions(task)
    if coalition not in table:
        print(f"[n/a] {coalition} is not defined for TabFM {task} ({dataset}); "
              "nothing to do", flush=True)
        return
    kwargs_tpl = table[coalition]
    Cls = make_capturing_regressor() if reg else make_capturing_classifier()
    model = load_model(task, checkpoint_path=checkpoint, device=device)
    tabfm_sha = _git_sha(os.environ.get("TABFM_REPO", "."))
    code_sha = _git_sha(os.environ.get("CODE_ROOT", "."))
    verified = not verify_first

    for k, (X_tr, y_tr, X_te, y_te) in enumerate(load_splits(dataset)):
        cell = out_root / dataset / coalition / f"split{k}"
        if (cell / "meta.json").exists():
            print(f"[skip] {cell} already complete", flush=True)
            continue
        cell.mkdir(parents=True, exist_ok=True)

        n_rows, n_features = np.asarray(X_tr).shape
        kwargs = resolve_fracs(kwargs_tpl, n_rows, n_features)
        # 'base' pins its own member count; everything else takes the budget.
        n_est = int(kwargs.pop("n_estimators", n_estimators))

        clf = Cls(model=model, n_estimators=n_est, random_state=seed, **kwargs)

        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        t_fit = time.perf_counter() - t0

        if not verified:
            # Upstream owns the member/aggregate boundary; assert it has not
            # moved before trusting a whole campaign to this cache.
            Xv = X_te[:64] if hasattr(X_te, "__getitem__") else X_te
            (verify_equivalence_regression if reg else verify_equivalence)(clf, Xv)
            print("[verify] member capture == predict"
                  + (" (rtol 1e-6)" if reg else " (bitwise)"), flush=True)
            verified = True

        t0 = time.perf_counter()
        members = clf.predict_members(X_te)
        t_pred = time.perf_counter() - t0

        m_realised = int(members.shape[0])
        np.save(cell / "members.npy", members.astype(np.float32))
        np.save(cell / "y_test.npy", np.asarray(y_te))

        sev = split_severity(dataset, k)
        meta = {
            "dataset": dataset, "split": k, "coalition": coalition,
            "model": "TabFM", "backend": "pytorch", "task": task,
            "checkpoint": checkpoint or CHECKPOINT_REPO,
            "split_severity": sev, "split_verified": sev == "ok",
            "axis_kwargs": {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist())
                            for kk, vv in kwargs.items()},
            "n_estimators_requested": n_est, "m_realised": m_realised,
            "truncated": m_realised < n_est,
            "enable_nnls": bool(getattr(clf, "enable_nnls", False)),
            "nnls_weights": (np.asarray(clf.ensemble_weights_).tolist()
                             if getattr(clf, "ensemble_weights_", None) is not None
                             else None),
            "axis_side": axis_side(coalition, task),
            "information": ("expanding" if is_expansion(coalition, task)
                            else "destroying" if destroys_information(coalition, task)
                            else "preserving"),
            "members": member_metadata(clf),
            "n_train": int(n_rows), "n_test": int(np.asarray(X_te).shape[0]),
            "n_features": int(n_features),
            "seed": seed,
            "tabfm_sha": tabfm_sha, "code_sha": code_sha,
            "fit_seconds": t_fit, "predict_seconds": t_pred,
            "seconds_per_member": t_pred / max(1, m_realised),
            "device": device,
        }
        if reg:
            meta.update({
                "space": "predictions", "average_logits": False,
                "softmax_temperature": None, "calibration": None,
                "n_classes": None,
                "y_train_std": float(np.std(np.asarray(y_tr, dtype=float))),
            })
        else:
            meta.update({
                # TabFM aggregates by log pooling unless NNLS is on; record which,
                # because the cached tensor is logits either way and the analysis
                # has to know what the shipped rule would have done with them.
                "average_logits": bool(getattr(clf, "average_logits", True)),
                "calibration": getattr(clf, "active_calibration_method_", None),
                "softmax_temperature": float(getattr(clf, "softmax_temperature", 0.9)),
                "space": "logits",
                "n_classes": int(len(np.unique(y_tr))),
            })
        try:
            import torch                                     # noqa: PLC0415
            meta["gpu"] = (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None)
            meta["peak_mem_bytes"] = (int(torch.cuda.max_memory_allocated())
                                      if torch.cuda.is_available() else None)
        except Exception:                                    # noqa: BLE001
            pass

        (cell / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[done] {dataset}/{coalition}/split{k} M={m_realised}/{n_est} "
              f"fit={t_fit:.1f}s pred={t_pred:.1f}s", flush=True)
        del clf


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--coalition", required=True,
                   choices=sorted(set(TABFM_COALITIONS) | set(TABFM_REG_COALITIONS)))
    p.add_argument("--n-estimators", type=int, default=TABFM_DEFAULT_M)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--checkpoint", default=os.environ.get("TABFM_CKPT"),
                   help="local HF snapshot dir; compute nodes have no internet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args(argv)
    run_cell(a.dataset, a.coalition, a.n_estimators, a.out, a.checkpoint,
             a.seed, a.device, verify_first=not a.no_verify)


if __name__ == "__main__":
    main()
