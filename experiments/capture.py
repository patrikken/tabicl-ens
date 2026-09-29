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

from tabicl import TabICLClassifier


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
