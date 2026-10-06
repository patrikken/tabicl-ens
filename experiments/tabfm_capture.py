"""Member-level capture for TabFM (PyTorch backend).

TabFM already factors the member/aggregate boundary, so unlike TabICLv2 -- where
we had to mirror ``predict_proba`` up to the aggregation step and assert bitwise
equality -- the capture here is a three-line override. From
``tabfm/src/classifier_and_regressor.py`` @ fbb6655::

    def predict_proba(self, X):
        check_is_fitted(self)
        logits = self._predict_proba_internal(X)    # (E, T, K) raw member logits
        return self._process_logits(logits)         # the ONLY aggregator

so stashing the intermediate is all that is required. ``verify_equivalence``
still asserts the round trip, because the contract is upstream's to change.

WHY TabFM IS THE BETTER SECOND MODEL
------------------------------------
Every axis in our taxonomy is a constructor kwarg here, and two axes we never
had are too:

    A1 feature-order permutation    feat_shuffle_method="random" | "none"
    A2 class-order permutation      class_shift=True | False
    A3 preprocessing/transform      norm_methods=[...]
    A4 feature subsampling          max_num_features   (NATIVE -- we had to
                                    write subsample.py for TabICLv2)
    A5 context subsampling          max_num_rows       (NATIVE -- ditto
                                    context.py)
    A7 categorical relabelling      permute_categorical=True   <- NEW
    A8 feature EXPANSION            n_feature_crosses, n_svd_features  <- NEW,
                                    and the one regime our recovery law has no
                                    data on, because it ADDS information
                                    instead of withholding it.

Aggregation is a kwarg too: ``average_logits`` (log pooling, our shipped
TabICLv2 rule), ``enable_nnls`` + ``nnls_beta`` (validation-fit stacking weights
shrunk toward uniform), and Platt/vector calibration. That covers the learned
aggregator our τ/family sweep never tested.

CAVEATS
-------
* ``average_logits`` and ``enable_nnls`` are mutually exclusive (upstream
  raises); ``TABFM_COALITIONS`` respects that.
* Members are grouped by norm method and shuffled before the groups are
  concatenated, so the ``E`` axis is NOT "view i" in any natural order. Use
  :func:`member_metadata` to label members before attributing anything to an
  axis.
* The shipped ``results/*.parquet`` baselines are JAX/TPU. We run PyTorch, so
  ours are not bitwise comparable -- keep them under distinct method names.
* Pretrained weights are ``tabfm-non-commercial-v1.0``: research only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CHECKPOINT_REPO = "google/tabfm-1.0.0-pytorch"   # NB: 1.0.0, not 1.1.0


def load_model(model_type: str = "classification", checkpoint_path: str | None = None,
               device: str = "cuda", dtype: Any = None):
    """Load the PyTorch checkpoint, offline when ``checkpoint_path`` is given.

    Compute nodes have no internet: snapshot the HF repo on a login node and
    pass the directory. ``load`` auto-descends into ``classification/`` or
    ``regression/``. Defaults to CPU upstream if ``device`` is not passed, which
    silently costs an order of magnitude -- so it is explicit here.
    """
    import torch                                   # noqa: PLC0415
    from tabfm import tabfm_v1_0_0_pytorch         # noqa: PLC0415

    return tabfm_v1_0_0_pytorch.load(
        model_type=model_type, checkpoint_path=checkpoint_path,
        device=device, dtype=torch.bfloat16 if dtype is None else dtype)


def _classifier_base():
    from tabfm import TabFMClassifier              # noqa: PLC0415
    return TabFMClassifier


def make_capturing_classifier():
    """Build the capture subclass lazily, so importing this module needs no tabfm."""
    TabFMClassifier = _classifier_base()

    class MemberCapturingTabFMClassifier(TabFMClassifier):
        """TabFMClassifier that can return per-member outputs.

        ``predict_members`` returns ``(E, n_test, n_classes)`` **logits** --
        pre-softmax, pre-weighting, pre-calibration -- which keeps every
        aggregator available post hoc on the cache, exactly as the TabICLv2
        harness does.
        """

        def predict_members(self, X) -> np.ndarray:
            # Check the attribute fit() actually sets rather than
            # sklearn's check_is_fitted, which couples to the estimator-tag
            # protocol and breaks across sklearn versions for no benefit here.
            if not hasattr(self, "ensemble_generator_"):
                raise RuntimeError("call fit() before predict_members()")
            return np.asarray(self._predict_proba_internal(X))

        def aggregate_members(self, members: np.ndarray) -> np.ndarray:
            """Upstream's own aggregation, applied to a captured tensor."""
            return np.asarray(self._process_logits(np.asarray(members)))

    return MemberCapturingTabFMClassifier


def verify_equivalence(clf, X, atol: float = 0.0) -> None:
    """Assert captured-then-aggregated == ``predict_proba``.

    Gate every campaign on this. ``atol=0.0`` demands bitwise equality: both
    paths run the same kernels in the same order, so anything else means the
    member/aggregate boundary moved upstream and the cache would be silently
    wrong.
    """
    members = clf.predict_members(X)
    ours = clf.aggregate_members(members)
    theirs = np.asarray(clf.predict_proba(X))
    if atol == 0.0:
        if not np.array_equal(ours, theirs):
            d = float(np.abs(ours - theirs).max())
            raise AssertionError(
                f"Member capture diverges from predict_proba (max |delta| {d:.3e}). "
                f"Re-read classifier_and_regressor.py::_process_logits.")
    else:
        np.testing.assert_allclose(ours, theirs, atol=atol)


def member_metadata(clf) -> list[dict]:
    """One record per member, in ``E``-axis order.

    Members are built per norm method and the groups are concatenated, so the
    ``E`` index does not correspond to any natural view ordering -- attributing
    gain to an axis without this mapping silently mislabels members.
    """
    gen = getattr(clf, "ensemble_generator_", None)
    if gen is None:
        raise RuntimeError("fit the classifier first")
    configs = getattr(gen, "ensemble_configs_", None)
    if configs is None:
        return []
    out: list[dict] = []
    for norm_method, cfgs in configs.items():
        for i, cfg in enumerate(cfgs):
            rec: dict[str, Any] = {"norm_method": norm_method, "within_group": i}
            # cfg is a tuple whose arity has changed across versions; record
            # what is there rather than unpacking positionally.
            for name, val in zip(
                    ("shuffle_pattern", "shift_offset", "cat_perm", "row_sub_pattern"),
                    cfg if isinstance(cfg, (tuple, list)) else (cfg,)):
                if val is None:
                    rec[name] = None
                elif np.ndim(val) == 0:
                    rec[name] = int(val)
                else:
                    rec[name] = int(np.size(val))      # store size, not the array
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Coalitions. Same vocabulary as experiments/coalitions.py so the two models'
# results line up column for column in the analysis.
#
# ``frac_features`` / ``frac_rows`` are resolved against the dataset shape at
# fit time (see resolve_fracs) because TabFM takes absolute caps, not fractions.
# ---------------------------------------------------------------------------
ALL_NORMS = ["none", "power", "quantile", "quantile_rtdl", "robust"]

TABFM_COALITIONS: dict[str, dict[str, Any]] = {
    # --- baseline: one member, nothing perturbed ---
    "base": dict(n_estimators=1, feat_shuffle_method="none", class_shift=False,
                 norm_methods=["none"]),
    # --- single native axes ---
    "A1": dict(feat_shuffle_method="random", class_shift=False, norm_methods=["none"]),
    "A2": dict(feat_shuffle_method="none", class_shift=True, norm_methods=["none"]),
    "A3": dict(feat_shuffle_method="none", class_shift=False, norm_methods=ALL_NORMS),
    # --- pairs and the triple ---
    "A1A2": dict(feat_shuffle_method="random", class_shift=True, norm_methods=["none"]),
    "A1A3": dict(feat_shuffle_method="random", class_shift=False, norm_methods=ALL_NORMS),
    "A2A3": dict(feat_shuffle_method="none", class_shift=True, norm_methods=ALL_NORMS),
    "A1A2A3": dict(feat_shuffle_method="random", class_shift=True, norm_methods=ALL_NORMS),
    # --- what TabFM actually ships ---
    "shipped": dict(feat_shuffle_method="random", class_shift=True,
                    norm_methods=["none", "power"]),
    # --- A7: categorical relabelling. New axis, information-preserving,
    #     feature-side -- it belongs in the 2x2 cell with A1-A3 and we have
    #     never measured it. Off by default upstream, even in .ensemble().
    "A7": dict(feat_shuffle_method="none", class_shift=False, norm_methods=["none"],
               permute_categorical=True),
    # --- A4 / A5: native here. Fractions mirror the TabICLv2 sweeps exactly so
    #     the recovery curves are comparable point for point.
    **{f"A4_g{int(g*100):02d}": dict(feat_shuffle_method="none", class_shift=False,
                                     norm_methods=["none"], frac_features=g)
       for g in (0.25, 0.50, 0.75, 0.90)},
    **{f"A5_f{int(f*100):02d}": dict(feat_shuffle_method="none", class_shift=False,
                                     norm_methods=["none"], frac_rows=f)
       for f in (0.25, 0.50, 0.75, 0.90)},
    # --- A8: feature EXPANSION. The regime the recovery law has no data on:
    #     members gain columns instead of losing them, so member quality can be
    #     POSITIVE and net = quality + diversity leaves the measured regime.
    "A8cross": dict(feat_shuffle_method="none", class_shift=False,
                    norm_methods=["none"], n_feature_crosses="sqrt"),
    "A8svd": dict(feat_shuffle_method="none", class_shift=False,
                  norm_methods=["none"], n_svd_features="sqrt"),
    "A8both": dict(feat_shuffle_method="none", class_shift=False,
                   norm_methods=["none"], n_feature_crosses="sqrt",
                   n_svd_features="sqrt"),
    # --- the full TabFM+ stack, and its parts, for attribution. The paper
    #     reports the bundle (+69.4 Elo classification) and ablates none of it.
    "plus_full": dict(feat_shuffle_method="random", class_shift=True,
                      norm_methods=["none", "power"], n_feature_crosses="sqrt",
                      n_svd_features="sqrt", average_logits=False,
                      enable_nnls=True, binary_calibration_method="platt",
                      multiclass_calibration_method="vector"),
    "plus_noexpand": dict(feat_shuffle_method="random", class_shift=True,
                          norm_methods=["none", "power"], average_logits=False,
                          enable_nnls=True, binary_calibration_method="platt",
                          multiclass_calibration_method="vector"),
    "plus_nonnls": dict(feat_shuffle_method="random", class_shift=True,
                        norm_methods=["none", "power"], n_feature_crosses="sqrt",
                        n_svd_features="sqrt"),
    "plus_nocal": dict(feat_shuffle_method="random", class_shift=True,
                       norm_methods=["none", "power"], n_feature_crosses="sqrt",
                       n_svd_features="sqrt", average_logits=False,
                       enable_nnls=True),
}

#: M members = M forward passes here (one fit, cached context), so the budget can
#: match TabICLv2's 32 rather than A4/A5's 16.
TABFM_DEFAULT_M = 32

FRAC_KEYS = ("frac_features", "frac_rows")


def resolve_fracs(kwargs: dict, n_rows: int, n_features: int) -> dict:
    """Turn ``frac_features`` / ``frac_rows`` into TabFM's absolute caps.

    TabFM subsamples per member when ``max_num_features`` / ``max_num_rows`` is
    below the table's size, which is exactly our A4 / A5 construction -- but it
    takes counts, not fractions. Resolving here keeps the coalition table
    dataset-independent.

    Note ``max_num_features`` defaults to 500 upstream, so a table wider than
    that is ALREADY being subsampled before we touch it; we clamp to the true
    width so the fraction means what it says.
    """
    out = {k: v for k, v in kwargs.items() if k not in FRAC_KEYS}
    if "frac_features" in kwargs:
        out["max_num_features"] = max(2, int(round(kwargs["frac_features"] * n_features)))
    if "frac_rows" in kwargs:
        out["max_num_rows"] = max(16, int(round(kwargs["frac_rows"] * n_rows)))
    return out


def is_expansion(coalition: str) -> bool:
    """Does this coalition ADD columns rather than withhold them?"""
    kw = TABFM_COALITIONS[coalition]
    return bool(kw.get("n_feature_crosses") or kw.get("n_svd_features"))


def destroys_information(coalition: str) -> bool:
    kw = TABFM_COALITIONS[coalition]
    return any(k in kw for k in FRAC_KEYS)


def axis_side(coalition: str) -> str:
    kw = TABFM_COALITIONS[coalition]
    return "context" if "frac_rows" in kw else "feature"
