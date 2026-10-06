"""Dataset + split adapter.

CONTRACT: ``load_splits(name)`` yields ``(X_train, y_train, X_test, y_test)``
per evaluation split, in a stable order.

Backends, chosen by ``$SPLIT_BACKEND``:

* ``tabarena`` (default for the full campaign) - the official TabArena v0.1
  splits, read from the manifest written by ``prepare_tabarena.py`` on a login
  node. The ``tabarena`` PACKAGE is never installed or imported anywhere in this
  project; the splits are reproduced from OpenML directly and verified against
  TabArena's own recorded train/test sizes. Run time needs numpy + the OpenML
  cache, nothing else.
* ``openml`` - the pilot backend: 8 datasets, own stratified splits. Fine for
  variance/cost questions; NOT comparable to published TabArena numbers.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

Split = Tuple[object, object, object, object]

BACKEND = os.environ.get("SPLIT_BACKEND", "tabarena")
TABARENA_DIR = Path(os.environ.get("TABARENA_DIR", "")) if os.environ.get("TABARENA_DIR") else None
N_SPLITS = int(os.environ.get("N_SPLITS", "9"))

# --- pilot set (OpenML backend) ------------------------------------------------
#   name                      openml  n      d     C
PILOT = {
    "blood-transfusion":      (46913,   748,    5, 2),
    "diabetes":               (46921,   768,    9, 2),
    "credit-g":               (46918,  1000,   21, 2),
    "maternal_health_risk":   (46941,  1014,    7, 3),
    "MIC":                    (46980,  1699,  112, 8),
    "students_dropout":       (46960,  4424,   37, 3),
    "Bioresponse":            (46912,  3751, 1777, 2),
    "Amazon_employee_access": (46905, 32769,   10, 2),
}


def load_splits(name: str) -> Iterator[Split]:
    if BACKEND == "openml":
        yield from _load_openml(name)
    else:
        yield from _load_tabarena(name)


# --- TabArena ------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _manifest() -> dict:
    if TABARENA_DIR is None:
        raise RuntimeError(
            "TABARENA_DIR is unset. Run prepare_tabarena.py on a login node and "
            "export TABARENA_DIR=$PROJECT_ROOT/tabarena."
        )
    f = TABARENA_DIR / "manifest.json"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found. Run:  python -m experiments.prepare_tabarena "
            f"--out {TABARENA_DIR}   (login node, needs internet)"
        )
    return json.loads(f.read_text())


def tabarena_datasets(task: str | None = "classification") -> list[str]:
    """Dataset names in the manifest, sorted. Used by the SLURM arrays.

    ``task`` filters on the manifest's ``problem_type``: ``"classification"``
    (default, so every pre-regression script keeps its behaviour once the
    manifest also holds regression datasets), ``"regression"``, or ``None`` for
    everything.
    """
    m = _manifest()
    names = sorted(m)
    if task is None:
        return names
    return [n for n in names if _task_of(m[n].get("problem_type")) == task]


def _task_of(problem_type) -> str:
    return "regression" if str(problem_type).lower() == "regression" else "classification"


def dataset_task(name: str) -> str:
    """'classification' | 'regression'. The OpenML pilot backend is classification."""
    try:
        return _task_of(_manifest()[name].get("problem_type"))
    except Exception:                                      # noqa: BLE001
        return "classification"


def dataset_info(name: str) -> dict:
    return _manifest()[name]


def split_severity(name: str, k: int) -> str:
    """Provenance of split ``k`` vs TabArena's published record.

    ``"ok"``       sizes match exactly.
    ``"resized"``  sizes differ - internally valid, NOT leaderboard-comparable.
    ``"overlap"``  train/test intersect - leakage, unusable for any claim.
    ``"unknown"``  no manifest (e.g. the OpenML pilot backend).

    Recorded into every cell's meta.json by run_cell.py so the analysis can
    filter or stratify without re-joining the audit table.
    """
    try:
        splits = _manifest()[name]["splits"]
    except Exception:  # noqa: BLE001 - pilot backend, or manifest absent
        return "unknown"
    for s in splits:
        if s["k"] == k:
            return s.get("severity", "unknown")
    return "unknown"


def _load_tabarena(name: str) -> Iterator[Split]:
    import openml

    info = _manifest()[name]
    ds = openml.datasets.get_dataset(info["openml_dataset_id"], download_data=True)
    X, y, _, _ = ds.get_data(target=info["label"], dataset_format="dataframe")
    y = np.asarray(y, dtype=np.float64) if _task_of(info.get("problem_type")) == "regression" \
        else np.asarray(y)

    z = np.load(TABARENA_DIR / "splits" / f"{name}.npz")
    for s in info["splits"]:
        k = s["k"]
        tr, te = z[f"train_{k}"], z[f"test_{k}"]
        yield X.iloc[tr], y[tr], X.iloc[te], y[te]


# --- OpenML (pilot) ------------------------------------------------------------
def _load_openml(name: str) -> Iterator[Split]:
    import openml
    from sklearn.model_selection import RepeatedStratifiedKFold

    did = PILOT[name][0]
    ds = openml.datasets.get_dataset(did, download_data=True)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    y = np.asarray(y)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=max(1, N_SPLITS // 3), random_state=0)
    for tr, te in cv.split(np.zeros(len(y)), y):
        yield X.iloc[tr], y[tr], X.iloc[te], y[te]
