"""Member-level capture for TabICLv2.

Mirrors ``TabICLClassifier.predict_proba`` (tabicl v2.2.0,
``src/tabicl/_sklearn/classifier.py``) up to the aggregation step and returns
the per-member outputs instead of collapsing them.

Two invariants this module depends on, both read from v2.2.0:

1. ``predict_proba`` builds ``outputs`` of shape ``(M, n_test, n_classes)`` and
   a list ``class_shuffles`` of length ``M``, then aggregates via
   ``avg += outputs[i][..., class_shuffles[i]]``.
2. With ``average_logits=True`` (the default) ``outputs`` holds LOGITS, and the
   tempered softmax is applied only after averaging.

``verify_equivalence`` asserts that re-aggregating the captured members
reproduces ``predict_proba`` exactly. Run it before trusting any cache.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

try:  # sklearn >= 1.6
    from sklearn.utils.validation import validate_data
except ImportError:  # pragma: no cover
    validate_data = None

from tabicl import TabICLClassifier, TabICLRegressor


class MemberCapturingTabICLClassifier(TabICLClassifier):
    """TabICLClassifier that can return per-ensemble-member outputs."""

    def predict_members(self, X) -> np.ndarray:
        """Return aligned per-member outputs.

        Returns
        -------
        np.ndarray
            Shape ``(M_realised, n_test, n_classes)``. LOGITS when
            ``self.average_logits`` is True (the default), else probabilities.
            Class axes are already unshuffled, so members are directly
            comparable and ``members.mean(axis=0)`` is the ensemble
            pre-activation.
        """
        check_is_fitted(self)

        if validate_data is not None:
            X = validate_data(self, X, reset=False, dtype=None, skip_check_array=True)
        X = self.X_encoder_.transform(X)

        has_kv_cache = getattr(self, "model_kv_cache_", None) is not None

        outputs = []
        if has_kv_cache:
            test_data = self.ensemble_generator_.transform(X, mode="test")
            for norm_method, (Xs_test,) in test_data.items():
                outputs.append(
                    self._batch_forward_with_cache(Xs_test, self.model_kv_cache_[norm_method])
                )
        else:
            data = self.ensemble_generator_.transform(X, mode="both")
            for norm_method, (Xs, ys) in data.items():
                feature_shuffles = self.ensemble_generator_.feature_shuffles_[norm_method]
                outputs.append(self._batch_forward(Xs, ys, feature_shuffles))
        outputs = np.concatenate(outputs, axis=0)

        class_shuffles = []
        for shuffles in self.ensemble_generator_.class_shuffles_.values():
            class_shuffles.extend(shuffles)

        return np.stack(
            [outputs[i][..., sh] for i, sh in enumerate(class_shuffles)], axis=0
        )

    @property
    def m_realised_(self) -> int:
        """Actual member count, which may be below ``n_estimators``.

        tabicl truncates the (feature x class) x norm product to
        ``n_estimators``, and caps each axis by ``n_features`` / ``n_classes``,
        so the realised count is dataset-dependent. Always log this rather than
        the requested ``n_estimators``.
        """
        check_is_fitted(self)
        return sum(len(v) for v in self.ensemble_generator_.class_shuffles_.values())


def aggregate_like_tabicl(members: np.ndarray, clf) -> np.ndarray:
    """Reproduce tabicl's shipped aggregation from captured members.

    Shipped rule: arithmetic mean over members (in logit space when
    ``average_logits``), then softmax at ``softmax_temperature``, then
    renormalise. This is LOGARITHMIC pooling of probabilities, not linear.
    """
    avg = members.mean(axis=0)
    if clf.average_logits:
        avg = clf.softmax(avg, axis=-1, temperature=clf.softmax_temperature)
    return avg / avg.sum(axis=1, keepdims=True)


def verify_equivalence(clf, X, atol: float = 0.0) -> None:
    """Assert captured-then-aggregated == predict_proba.

    Gate every campaign on this. ``atol=0.0`` demands bitwise equality, which
    should hold since both paths run the same kernels in the same order.
    """
    members = clf.predict_members(X)
    ours = aggregate_like_tabicl(members, clf)
    theirs = clf.predict_proba(X)
    if atol == 0.0:
        assert np.array_equal(ours, theirs), (
            "Member capture diverges from predict_proba. Upstream aggregation "
            "likely changed; re-read classifier.py::predict_proba."
        )
    else:
        np.testing.assert_allclose(ours, theirs, atol=atol)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
class MemberCapturingTabICLRegressor(TabICLRegressor):
    """TabICLRegressor that can return per-ensemble-member predictions.

    Mirrors ``TabICLRegressor.predict`` (tabicl v2.2.0,
    ``src/tabicl/_sklearn/regressor.py``) up to the aggregation step. predict():

        data = ensemble_generator_.transform(X, mode="both")
        for Xs, ys in data.values():  results.append(_batch_forward(Xs, ys, "mean"))
        arr = concatenate(results)                                # (E, T) scaled
        arr = y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(E, T)
        return arr.mean(axis=0)                                   # the ONLY aggregator

    so a member here is one view's predictive MEAN on the ORIGINAL target scale,
    and the shipped aggregator is the arithmetic mean over members. The kv-cache
    path is not mirrored (it is off in every run; see run_cell.py).
    """

    def predict_members(self, X) -> np.ndarray:
        """Return ``(M_realised, n_test)`` per-member predictions, original scale."""
        check_is_fitted(self)
        if getattr(self, "model_kv_cache_", None) is not None:
            raise RuntimeError("kv_cache is not mirrored by predict_members; "
                               "construct with kv_cache=False")
        if validate_data is not None:
            X = validate_data(self, X, reset=False, dtype=None, skip_check_array=True)
        X = self.X_encoder_.transform(X)

        data = self.ensemble_generator_.transform(X, mode="both")
        outs = [self._batch_forward(Xs, ys, output_type="mean")
                for Xs, ys in data.values()]
        arr = np.concatenate(outs, axis=0)                     # (E, T), scaled
        E, T = arr.shape
        return self.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(E, T)

    @property
    def m_realised_(self) -> int:
        """Actual member count (tabicl truncates the view product to n_estimators)."""
        check_is_fitted(self)
        return int(sum(len(v) for v in self.ensemble_generator_.feature_shuffles_.values()))


def aggregate_regression(members: np.ndarray) -> np.ndarray:
    """tabicl's shipped regression aggregation: arithmetic mean over members."""
    return np.mean(members, axis=0)


def verify_equivalence_regression(reg, X, atol: float = 0.0) -> None:
    """Assert captured-then-averaged == ``predict`` (bitwise by default)."""
    ours = aggregate_regression(reg.predict_members(X))
    theirs = reg.predict(X)
    if atol == 0.0:
        if not np.array_equal(ours, theirs):
            d = float(np.abs(ours - theirs).max())
            raise AssertionError(
                f"Regression member capture diverges from predict (max |delta| {d:.3e}). "
                "Upstream aggregation likely changed; re-read regressor.py::predict.")
    else:
        np.testing.assert_allclose(ours, theirs, atol=atol)


# ---------------------------------------------------------------------------
# Shared by the A4 / A5 wrappers (subsample.py, context.py), which fit one
# single-member estimator per draw and must work for both tasks.
# ---------------------------------------------------------------------------
CKPT_CLASSIFIER = "tabicl-classifier-v2-20260212.ckpt"
CKPT_REGRESSOR = "tabicl-regressor-v2-20260212.ckpt"


def default_checkpoint(task: str) -> str:
    return CKPT_REGRESSOR if task == "regression" else CKPT_CLASSIFIER


def member_estimator(task: str, checkpoint: str, seed: int, device):
    """One unperturbed single-member estimator: every native axis off."""
    if task == "regression":
        return MemberCapturingTabICLRegressor(
            n_estimators=1, feat_shuffle_method="none", norm_methods=["none"],
            checkpoint_version=checkpoint, random_state=seed, device=device)
    return MemberCapturingTabICLClassifier(
        n_estimators=1, feat_shuffle_method="none", class_shuffle_method="none",
        norm_methods=["none"], checkpoint_version=checkpoint,
        random_state=seed, device=device)


def align_member(owner, clf, m, ref_classes, all_classes):
    """Record aggregation metadata from member 0 and check class alignment.

    Regression has no class axis, so there is nothing to align and the cached
    members are original-scale predictions (``average_logits`` is False).
    """
    if getattr(owner, "task", "classification") == "regression":
        if ref_classes is None:
            owner.average_logits = False
            owner.softmax_temperature = float("nan")
            return np.array([])
        return ref_classes
    cls = np.asarray(clf.classes_)
    if ref_classes is None:
        owner.average_logits = bool(clf.average_logits)
        owner.softmax_temperature = float(clf.softmax_temperature)
        if len(cls) != len(all_classes):
            raise RuntimeError(
                f"member 0 fitted {len(cls)} classes but the pool has "
                f"{len(all_classes)}; raise min_per_class or frac.")
        return cls
    if not np.array_equal(cls, ref_classes):
        raise RuntimeError(
            f"member {m} fitted a different class set ({cls} vs {ref_classes}); "
            "member columns would not align.")
    return ref_classes
