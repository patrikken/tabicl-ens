"""Axis coalitions as TabICLClassifier constructor kwargs.

Axis vocabulary (see EXPERIMENTS.md §3):
  A1  feature-order permutation   -> feat_shuffle_method
  A2  class-order permutation     -> class_shuffle_method   (classification only)
  A3  preprocessing / transform   -> norm_methods
  A4  feature subsampling         -> NOT native; wrapper, see subsample.py
  A5  context construction        -> NOT native; wrapper, see subsample.py
  A6  stochastic replication      -> structurally empty; determinism control

Structural caps, from tabicl v2.2.0 ``Shuffler.shuffle``:
  A1  'shift'/'latin' capped by n_features; 'random' uncapped when n_features>5
  A2  'shift' yields exactly n_classes patterns  -> only 2 on binary tasks
  A3  at most 5 distinct normalisation methods
Only A1='random' reaches M=64. Cap the budget grid per axis and draw the cap on
every scaling figure, or structural ceilings will be misread as saturation.
"""

from __future__ import annotations

from typing import Any, Dict

ALL_NORM_METHODS = ["none", "power", "quantile", "quantile_rtdl", "robust"]

#: Coalitions over the native axes, plus the shipped default and the A6 control.
COALITIONS: Dict[str, Dict[str, Any]] = {
    # Baseline: exactly one member, no perturbation. v(empty set).
    "base": dict(
        feat_shuffle_method="none", class_shuffle_method="none", norm_methods=["none"]
    ),
    # Single axes.
    "A1": dict(
        feat_shuffle_method="random", class_shuffle_method="none", norm_methods=["none"]
    ),
    "A2": dict(
        feat_shuffle_method="none", class_shuffle_method="shift", norm_methods=["none"]
    ),
    "A3": dict(
        feat_shuffle_method="none", class_shuffle_method="none",
        norm_methods=ALL_NORM_METHODS,
    ),
    # Pairs.
    "A1A2": dict(
        feat_shuffle_method="random", class_shuffle_method="shift", norm_methods=["none"]
    ),
    "A1A3": dict(
        feat_shuffle_method="random", class_shuffle_method="none",
        norm_methods=ALL_NORM_METHODS,
    ),
    "A2A3": dict(
        feat_shuffle_method="none", class_shuffle_method="shift",
        norm_methods=ALL_NORM_METHODS,
    ),
    # Triple.
    "A1A2A3": dict(
        feat_shuffle_method="random", class_shuffle_method="shift",
        norm_methods=ALL_NORM_METHODS,
    ),
    # The configuration tabicl actually ships. Not equal to any clean coalition:
    # members are a randomly-permuted truncation of the product space.
    "shipped": dict(
        feat_shuffle_method="latin", class_shuffle_method="shift",
        norm_methods=["none", "power"],
    ),
    # Determinism control. Every axis off => exactly one unique member, and the
    # forward pass is deterministic, so spread across seeds MUST be zero.
    # Nonzero spread means nondeterminism (TF32 / AMP / reduction order) leaked
    # in and would inflate every diversity diagnostic in the study.
    "A6_control": dict(
        feat_shuffle_method="none", class_shuffle_method="none", norm_methods=["none"]
    ),
}

#: Per-axis ceiling on distinct members, given dataset shape.
def max_members(coalition: str, n_features: int, n_classes: int) -> int:
    """Upper bound on distinct members for a coalition on a given dataset."""
    kw = COALITIONS[coalition]
    f = kw["feat_shuffle_method"]
    c = kw["class_shuffle_method"]
    n_norm = len(kw["norm_methods"])

    if f == "none":
        n_feat = 1
    elif f == "random":
        n_feat = 10**6 if n_features > 5 else _factorial(n_features)
    else:  # shift, latin
        n_feat = max(1, n_features)

    n_cls = max(1, n_classes) if c != "none" else 1
    return n_feat * n_cls * n_norm


def _factorial(n: int) -> int:
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out

# ---------------------------------------------------------------------------
# A5 - context-side perturbation (see experiments/context.py)
#
# NOT native to TabICLv2: every shipped member sees all context rows. These are
# a wrapper intervention, so a positive result is a recommendation to change the
# model's member-generating process, not a description of it.
#
# Each coalition holds the context SIZE fixed and varies only which rows are
# drawn, so gain vs the coalition's own M=1 isolates diversity from the
# information loss that plain subsampling would introduce. Sweeping frac is the
# diagnostic: as frac -> 1 members converge, so diversity -> 0 while member
# quality -> max.
CONTEXT_COALITIONS: Dict[str, Dict[str, Any]] = {
    "A5_f25":    dict(frac=0.25, mode="random"),
    "A5_f50":    dict(frac=0.50, mode="random"),
    "A5_f75":    dict(frac=0.75, mode="random"),
    "A5_f90":    dict(frac=0.90, mode="random"),
    # class-balanced draws at matched size: the group-balanced construction
    # from the context-selection literature, as an axis rather than a fix.
    "A5bal_f50": dict(frac=0.50, mode="balanced"),
}

#: A5 needs M *fits* (the context is the training data), not M forward passes
#: through one fit, so it is an order of magnitude dearer per member. Default
#: lower than the feature-side budget; raise only with the timings in hand.
CONTEXT_DEFAULT_M = 16


def is_context(coalition: str) -> bool:
    return coalition in CONTEXT_COALITIONS


def all_coalitions() -> Dict[str, Dict[str, Any]]:
    return {**COALITIONS, **CONTEXT_COALITIONS}

# ---------------------------------------------------------------------------
# A5-LOFO - leave-one-fold-out context partitioning.
#
# The fraction sweep above varied f at FIXED M=16, which cannot separate "how
# much context is dropped" from "how many members there are". LOFO ties them,
#
#     f = 1 - 1/M,   pairwise overlap = (M-2)/(M-1),
#
# so sweeping M traces a different path through the (M, f) plane: as M grows the
# quality cost falls AND the diversity falls, and the product may have an
# interior optimum the fixed-M sweep was structurally unable to see.
#
# Cost warning: M members = M fits, and at large M each fit is nearly
# full-sized, so A5lofo_M64 is ~64 near-full fits per cell. Small/medium only.
LOFO_COALITIONS: Dict[str, Dict[str, Any]] = {
    "A5lofo_M8":  dict(mode="lofo", n_folds=8),    # f = 0.875
    "A5lofo_M16": dict(mode="lofo", n_folds=16),   # f = 0.9375
    "A5lofo_M32": dict(mode="lofo", n_folds=32),   # f = 0.969
    "A5lofo_M64": dict(mode="lofo", n_folds=64),   # f = 0.984
}
CONTEXT_COALITIONS.update(LOFO_COALITIONS)


def lofo_folds(coalition: str) -> int | None:
    """Member count implied by a LOFO coalition (M is not free here)."""
    return LOFO_COALITIONS.get(coalition, {}).get("n_folds")
