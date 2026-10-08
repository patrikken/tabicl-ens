"""Shipped-style member views, and categorical relabelling (A7), for the wrappers.

The A4/A5 wrappers fit one single-view estimator per member, with every native
axis off. To ask "does withholding rows/columns ADD to what the shipped recipe
already does?" each member must also carry a shipped-style view. TabICLv2's
shipped schedule is a truncated product of (feature order) x (class order) x
(norm in {none, power}); inside a one-view member that becomes

    feature order  random permutation, seeded by the member   (MemberView)
    class order    random permutation, seeded by the member   (MemberView;
                   classification only, undone on the logits)
    norm method    none for even members, power for odd members (estimator kwarg)

Feature and class order are applied OUTSIDE the estimator because a one-member
estimator ignores them (upstream Shuffler returns the identity for n_estimators=1).
A first version passed them as kwargs and silently produced members that differed
only in normalisation; test_shipplus guards against that.

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
    """Estimator kwargs for member ``m``: the normalisation half of the shipped view.

    Feature and class order CANNOT be set here: upstream's ``Shuffler`` returns the
    identity pattern whenever ``n_estimators == 1`` (``if method == "none" or
    n_estimators == 1``), so a single-view estimator ignores both kwargs. Those two
    axes are applied by :class:`MemberView` instead, outside the estimator."""
    return dict(feat_shuffle_method="none", norm_methods=[SHIPPED_NORMS[m % 2]],
                **({} if task == "regression" else {"class_shuffle_method": "none"}))


class MemberView:
    """The feature-order (A1) and class-order (A2) half of a shipped-style view,
    applied around a single-view estimator.

    ``transform`` permutes the columns of the member's train and test tables and,
    for classification, replaces the labels by a permuted integer coding, so the
    model sees a different feature order and a different class order. ``restore``
    undoes the class permutation on the member's logits, so every cached member is
    aligned to the ORIGINAL class order and can be pooled with any other.
    Pass ``None`` for ``classes`` on regression.
    """

    def __init__(self, n_features: int, classes, rng: np.random.Generator):
        self.feat_perm = rng.permutation(n_features)
        self.classes = None if classes is None else np.asarray(classes)
        self.class_perm = None if classes is None else rng.permutation(len(self.classes))

    @staticmethod
    def _cols(X, perm):
        return X.iloc[:, perm] if hasattr(X, "iloc") else np.asarray(X)[:, perm]

    def transform(self, X_tr, X_te, y_tr):
        X_tr, X_te = self._cols(X_tr, self.feat_perm), self._cols(X_te, self.feat_perm)
        if self.class_perm is None:
            return X_tr, X_te, y_tr
        code = np.searchsorted(self.classes, np.asarray(y_tr))
        return X_tr, X_te, self.class_perm[code]

    def restore(self, logits: np.ndarray) -> np.ndarray:
        """Member logits over permuted class codes -> original class order."""
        return logits if self.class_perm is None else logits[..., self.class_perm]

    def proxy(self, clf):
        """``clf`` as seen by ``capture.align_member``: original class labels."""
        from types import SimpleNamespace
        return SimpleNamespace(
            classes_=self.classes,
            average_logits=getattr(clf, "average_logits", True),
            softmax_temperature=getattr(clf, "softmax_temperature", 0.9))


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
