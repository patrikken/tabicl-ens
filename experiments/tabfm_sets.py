"""Which TabFM coalitions to run, and on which datasets.

Kept apart from ``tabfm_capture`` so the SLURM scripts can enumerate the grid
without importing torch or the 6.5 GB checkpoint on a login node.

The sets, in the order worth spending compute on:

``axes``       base + A1/A2/A3 and their mixes + shipped. The TabICLv2 campaign
               replayed on a second model -- the transfer test, and the only
               thing that turns our claims from "about TabICLv2" into "about
               in-context learners".
``withhold``   A4 / A5 fraction sweeps, matched to the TabICLv2 fractions. Does
               the recovery curve transfer?
``expand``     A8 feature expansion. The regime the recovery law has NO data on,
               because members gain columns rather than lose them.
``plus``       the TabFM+ stack and its ablations. The paper reports the bundle
               (+69.4 Elo classification) and attributes none of it; these four
               coalitions take it apart.
``new``        A7 categorical relabelling -- an information-preserving
               feature-side axis neither model's wrapper documents.
"""

from __future__ import annotations

from experiments.tabfm_capture import TABFM_COALITIONS, tabfm_coalitions

#: a 25% column draw of 6 columns is 2 columns; the collapse there is about the
#: dataset, not about ensembling. Same gate as the TabICLv2 A4 campaign.
A4_MIN_FEATURES = 8

SETS: dict[str, list[str]] = {
    "axes":     ["base", "A1", "A2", "A3", "A1A2", "A1A3", "A2A3", "A1A2A3",
                 "shipped"],
    "withhold": [f"A4_g{g:02d}" for g in (25, 50, 75, 90)]
                + [f"A5_f{f:02d}" for f in (25, 50, 75, 90)],
    "expand":   ["A8cross", "A8svd", "A8both"],
    "plus":     ["plus_full", "plus_noexpand", "plus_nonnls", "plus_nocal"],
    "new":      ["A7"],
}
SETS["all"] = [c for s in ("axes", "withhold", "expand", "plus", "new")
               for c in SETS[s]]


def coalition_set(name: str, n_features: int | None = None,
                  task: str = "classification") -> list[str]:
    """Coalitions in set ``name`` that exist for ``task`` and fit this table width.

    Regression drops everything that needs classes or logits (A2 and its mixes,
    the calibration ablation); ``SETS`` stays the classification vocabulary and
    the task filter is applied here so the two never drift.
    """
    if name not in SETS:
        raise KeyError(f"unknown set {name!r}; have {sorted(SETS)}")
    avail = tabfm_coalitions(task)
    out = [c for c in SETS[name] if c in avail]
    if n_features is not None and n_features < A4_MIN_FEATURES:
        out = [c for c in out if not c.startswith("A4_")]
    if task == "classification":
        missing = [c for c in SETS[name] if c not in TABFM_COALITIONS]
        if missing:
            raise KeyError(f"coalitions not defined: {missing}")
    return out
