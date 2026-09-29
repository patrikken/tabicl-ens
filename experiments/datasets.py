"""Dataset + split adapter.

CONTRACT: ``load_splits(name)`` yields ``(X_train, y_train, X_test, y_test)``
per evaluation split, in a stable order.

Two backends:

* ``openml``  (default) - downloads by OpenML ID and makes its own repeated
  stratified splits. Lets the pilot run immediately, without waiting on the
  TabArena API.
* ``tabarena`` - the official splits. REQUIRED for the full campaign: the
  paper compares against published TabArena numbers, and that comparison is
  only valid on their splits. Fill in ``_load_tabarena`` before Phase 1.

The OpenML IDs below were transcribed from a secondary source (the dataset
inventory in arXiv:2605.18696). VERIFY each against the official TabArena
registry before citing any number produced from them.
"""

from __future__ import annotations

import os
from typing import Iterator, Tuple

import numpy as np

Split = Tuple[object, object, object, object]

# Pilot set: 8 datasets chosen to span the structural caps of EXPERIMENTS.md §1.3
# and the cost regimes, not to be representative of TabArena.
#
#   name                         openml  n       d      C   why it is here
PILOT = {
    "blood-transfusion":         (46913,  748,     5,   2),  # d=5 -> A1 'random' hits the all_perms path
    "diabetes":                  (46921,  768,     9,   2),  # small binary reference
    "credit-g":                  (46918, 1000,    21,   2),  # small binary, more columns
    "maternal_health_risk":      (46941, 1014,     7,   3),  # small multiclass
    "MIC":                       (46980, 1699,   112,   8),  # C=8 -> A2 has real headroom
    "students_dropout":          (46960, 4424,    37,   3),  # medium multiclass
    "Bioresponse":               (46912, 3751,  1777,   2),  # d >> n -> where A4 should matter
    "Amazon_employee_access":    (46905, 32769,   10,   2),  # large n -> cost scaling
}

N_SPLITS = int(os.environ.get("N_SPLITS", "9"))
BACKEND = os.environ.get("SPLIT_BACKEND", "openml")


def load_splits(name: str) -> Iterator[Split]:
    if BACKEND == "tabarena":
        yield from _load_tabarena(name)
    else:
        yield from _load_openml(name)


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


def _load_tabarena(name: str) -> Iterator[Split]:
    """Official TabArena splits. REQUIRED for Phase 1 onward.

    TODO: wire to the tabarena / tabrepo API and yield its outer
    repeat/fold splits in their canonical order. Until then the campaign
    is not comparable to published TabArena results.
    """
    raise NotImplementedError(
        "TabArena backend not wired. Pilot may run with SPLIT_BACKEND=openml; "
        "the full campaign may not."
    )
