"""Shipped-style member views, and categorical relabelling (A7), for the wrappers.

The A4/A5 wrappers fit one single-view estimator per member, with every native
axis off. To ask "does withholding rows/columns ADD to what the shipped recipe
already does?" each member must also carry a shipped-style view. TabICLv2's
shipped schedule is a truncated product of (feature order) x (class order) x
(norm in {none, power}); inside a one-view member that becomes

    feature order  random permutation, seeded by the member
    class order    random permutation, seeded by the member   (classification)
    norm method    none for even members, power for odd members

This is a faithful DRAW from the shipped distribution (random vs 'latin'
permutations differ only in covering the pattern space evenly), not the exact
shipped schedule -- the ``S_ctl`` coalition (full context + this view) is the
matched control, so every "S_A*" gain is read against it, never against the
native ``shipped`` cell.

No torch / tabicl import here: the SLURM tooling and tests import this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHIPPED_NORMS = ("none", "power")


def shipped_view(task: str, m: int) -> dict:
    """Estimator kwargs for member ``m`` carrying a shipped-style view."""
    kw = dict(feat_shuffle_method="random", norm_methods=[SHIPPED_NORMS[m % 2]])
    if task != "regression":
        kw["class_shuffle_method"] = "random"
    return kw


def categorical_columns(X) -> list:
    """Columns the model treats as categorical (pandas category / object / bool)."""
    if not hasattr(X, "dtypes"):
        return []
    return [c for c, t in X.dtypes.items()
            if isinstance(t, pd.CategoricalDtype) or t == object or t == bool
            or pd.api.types.is_string_dtype(t)]


def relabel_categories(X_tr, X_te, rng: np.random.Generator):
    """A7: permute the integer coding of every categorical column, consistently
    across train and test. Information-preserving -- a bijection on the values
    seen in training -- but it changes the ordinal order the encoder sees.

    Each training category becomes a zero-padded rank string drawn from a random
    permutation, so any lexicographic/ordinal encoder downstream sees a new order.
    Categories first seen at test time have no training code and become missing
    (the same thing an ordinal encoder does with an unseen value).
    Returns copies; the inputs are untouched.
    """
    cols = categorical_columns(X_tr)
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for c in cols:
        cats = pd.unique(X_tr[c].dropna())
        perm = rng.permutation(len(cats))
        mapping = {v: f"{int(p):06d}" for v, p in zip(cats, perm)}
        X_tr[c] = X_tr[c].astype(object).map(mapping)
        X_te[c] = X_te[c].astype(object).map(mapping)
    return X_tr, X_te, cols
