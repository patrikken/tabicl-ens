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
