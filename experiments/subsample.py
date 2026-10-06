"""A4 — feature-side subsampling. The missing cell of the perturbation 2x2.

Our axes divide two ways: *where* the perturbation acts (features vs context)
and *whether it destroys information*:

                      information-preserving   information-destroying
    feature-side      A1, A2, A3               **A4  (this module)**
    context-side      -- (impossible)          A5  (context.py)

A1--A3 re-present the same data (column order, class order, normalisation) and
cost the member almost nothing: measured recovery rate -- diversity gained per
unit of member quality destroyed -- is 1.0--1.7 on binary and 2.1--3.0 on
multiclass, i.e. they always pay. A5 removes rows and recovers only 0.48--1.0
of what it destroys, so it never pays until it is nearly trivial.

A4 removes *columns*. It destroys information like A5 but leaves every row in
the context like A1--A3, so it is the one construction that separates the two
candidate explanations:

* recovery < 1  -> the law is about information destruction in general.
* recovery > 1  -> rows are special (the retrieval account), and feature
  bagging is a wrapper improvement no TFM currently ships.

Either outcome is informative, which is the point of running it. Note also that
feature subsampling (``mtry``) is *the* diversity mechanism in random forests,
the most successful tabular ensemble family there is, and every TFM inference
wrapper omits it.

DESIGN, MIRRORING A5
--------------------
Every coalition holds the subset SIZE fixed at a fraction ``frac`` of the
columns and varies only *which* columns are drawn, so one run yields both
comparisons:

* **matched**  -- vs the same coalition at M=1, i.e. one member at the same
  column count. Isolates diversity from the information loss.
* **absolute** -- vs ``base``, the full-column single pass. Is it worth doing.

Fractions mirror A5's exactly (0.25/0.50/0.75/0.90) so the two recovery curves
are directly comparable point for point.

COST
----
Changing the column set changes the fit, so like A5 this needs **M fits**, not
M forward passes through one fit. Budget accordingly; default M=16.

CLASS COVERAGE
--------------
Dropping columns cannot change the label set, so unlike A5 there is no risk of
a member fitting a different class vector. We assert it anyway -- the check is
free and a silent column misalignment would corrupt every diagnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# column importance
# ---------------------------------------------------------------------------
def column_importance(X, y, seed: int = 0, max_rows: int = 10_000,
                      n_trees: int = 100, task: str = "classification") -> np.ndarray:
    """Model-free-ish per-column importance, computed on TRAIN only.

    An ExtraTrees fit on an ordinal encoding of the columns. Cheap (CPU,
    seconds), handles mixed dtypes, and is the standard "importance criterion"
    a practitioner would reach for.

    Returns a non-negative vector summing to 1. Falls back to uniform if
    anything goes wrong, so a scoring failure degrades the mode to ``random``
    rather than killing the cell -- the caller records which happened.

    NOTE for the write-up: this uses training labels, so importance-weighted
    members are biased toward the same informative columns. That should *raise*
    member quality and *lower* diversity relative to uniform draws, which is
    precisely the contrast the mode is there to measure. There is no test-set
    leakage -- the test split is never touched.
    """
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder

    d = np.asarray(X).shape[1]
    try:
        rng = np.random.default_rng(seed)
        n = np.asarray(X).shape[0]
        idx = (rng.choice(n, max_rows, replace=False) if n > max_rows
               else np.arange(n))

        if hasattr(X, "iloc"):
            sub = X.iloc[idx]
            num = sub.select_dtypes(include=[np.number])
            cat = sub.drop(columns=num.columns)
            parts, order = [], []
            if num.shape[1]:
                parts.append(SimpleImputer(strategy="median")
                             .fit_transform(num.to_numpy(dtype=float)))
                order.extend(sub.columns.get_indexer(num.columns))
            if cat.shape[1]:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                     unknown_value=-1, encoded_missing_value=-1)
                parts.append(enc.fit_transform(cat.astype(object)))
                order.extend(sub.columns.get_indexer(cat.columns))
            Z = np.hstack(parts)
        else:
            Z = np.asarray(X, dtype=float)[idx]
            Z = SimpleImputer(strategy="median").fit_transform(Z)
            order = list(range(d))

        Trees = ExtraTreesRegressor if task == "regression" else ExtraTreesClassifier
        et = Trees(n_estimators=n_trees, random_state=seed, n_jobs=-1)
        et.fit(Z, np.asarray(y)[idx])

        imp = np.zeros(d, dtype=float)
        imp[np.asarray(order, dtype=int)] = et.feature_importances_
        if not np.isfinite(imp).all() or imp.sum() <= 0:
            raise ValueError("degenerate importances")
        return imp / imp.sum()
    except Exception:                                   # noqa: BLE001
        return np.full(d, 1.0 / max(1, d))


# ---------------------------------------------------------------------------
# subset construction
# ---------------------------------------------------------------------------
def feature_subsets(n_features: int, n_members: int, frac: float,
                    rng: np.random.Generator, mode: str = "random",
                    importance: np.ndarray | None = None,
                    min_features: int = 2) -> list[np.ndarray]:
    """``n_members`` column-index arrays, one per member.

    ``random``      independent draws of k columns. Column usage is Binomial,
                    so some columns land in many members and some in none.
    ``roundrobin``  balanced dealing: every column is used either
                    floor(Mk/d) or ceil(Mk/d) times. Same k as ``random``, so
                    the pair isolates *coverage balance* from subset size --
                    if random loses, it is because it wastes draws, not
                    because it drops columns.
    ``importance``  draws of k columns without replacement, with probability
                    proportional to ``importance``. Biases members toward the
                    informative columns.
    ``lofo``        partition the columns into M folds; member m holds every
                    column except fold m. Ties the fraction to the member
                    count, g = 1 - 1/M, exactly as LOFO does for rows.

    All modes except ``lofo`` produce subsets of identical size, which is what
    makes the mode comparison a clean one.
    """
    d = int(n_features)
    if d < min_features:
        raise ValueError(f"A4 needs >= {min_features} columns, dataset has {d}")

    if mode == "lofo":
        fold = rng.permutation(d) % n_members
        out = [np.sort(np.where(fold != m)[0]) for m in range(n_members)]
        # Two ways this degenerates, and the dangerous one is the EMPTY fold.
        # With M > d some folds hold no columns at all, so those members see
        # every column and are identical to the unperturbed fit -- which
        # inflates member quality and deflates diversity without erroring.
        # Guard it here as well as in coalitions.a4_applicable, because a
        # silent pass would look like a real measurement.
        empty = n_members - len(np.unique(fold))
        if empty:
            raise ValueError(
                f"lofo with M={n_members} on d={d} columns leaves {empty} "
                f"empty fold(s); those members would see every column. "
                f"Use M <= d/2 or skip this dataset.")
        short = [m for m, s in enumerate(out) if len(s) < min_features]
        if short:
            raise ValueError(
                f"lofo with M={n_members} on d={d} leaves members with "
                f"< {min_features} columns; lower M or skip this dataset.")
        return out

    k = int(np.clip(round(frac * d), min_features, d))

    if mode == "random":
        return [np.sort(rng.choice(d, k, replace=False))
                for _ in range(n_members)]

    if mode == "importance":
        if importance is None:
            raise ValueError("mode='importance' requires an importance vector")
        p = np.asarray(importance, dtype=float)
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        return [np.sort(rng.choice(d, k, replace=False, p=p))
                for _ in range(n_members)]

    if mode == "roundrobin":
        # Deal from a reshuffled permutation stream so every column is used
        # as near-equally as possible, rejecting duplicates within a member.
        out, stream = [], list(rng.permutation(d))
        for _ in range(n_members):
            take: list[int] = []
            seen: set[int] = set()
            guard = 0
            while len(take) < k:
                if not stream:
                    stream = list(rng.permutation(d))
                    guard += 1
                    if guard > n_members + 2:           # pathological; bail out
                        rest = [c for c in range(d) if c not in seen]
                        take.extend(rest[: k - len(take)])
                        break
                c = stream.pop(0)
                if c in seen:
                    continue                            # defer to a later member
                seen.add(c)
                take.append(c)
            out.append(np.sort(np.asarray(take, dtype=np.int64)))
        return out

    raise ValueError(f"unknown mode {mode!r}")


def expected_overlap(frac: float) -> float:
    """Expected pairwise column overlap between two INDEPENDENT draws.

    Two independent k-subsets of d columns share ~k^2/d columns, i.e. a
    fraction k/d = frac of each. Identical in form to the row case, which is
    what makes the A4/A5 recovery curves comparable on the same x-axis.
    """
    return float(frac)


def balanced_overlap(n_members: int, n_features: int, k: float) -> float:
    """Expected pairwise overlap under a *balanced* design.

    Round-robin and LOFO do not draw independently: every column is used by
    the same number of members, u = Mk/d. Counting the pairs that share each
    column, C(u, 2) per column over C(M, 2) pairs in total, the mean shared
    fraction is

        (u - 1) / (M - 1).

    For LOFO k = d(1 - 1/M), so u = M - 1 and this reduces to the familiar
    (M - 2)/(M - 1). Balanced dealing sits slightly BELOW the independent-draw
    value at the same k -- spreading the columns evenly buys a little extra
    disjointness for free, which is the thing A4rr_g50 is there to price.
    """
    if n_members < 2:
        return 0.0
    u = n_members * k / max(1, n_features)
    return float((u - 1.0) / (n_members - 1.0))


def mean_pairwise_overlap(subsets: list[np.ndarray]) -> float:
    """Mean |A n B| / |A| over ALL pairs of members.

    Over all pairs, not consecutive ones. Consecutive pairs are exchangeable
    for independent draws, but under round-robin dealing adjacent members are
    carved from the same permutation sweep and are therefore the MOST disjoint
    pair in the pool -- a consecutive-pairs estimate reads ~0.24 where the true
    all-pairs value is ~0.47. M <= 64 here, so the full O(M^2) sweep is free.
    """
    M = len(subsets)
    if M < 2:
        return 0.0
    vals = [len(np.intersect1d(subsets[i], subsets[j])) / max(1, len(subsets[i]))
            for i in range(M) for j in range(i + 1, M)]
    return float(np.mean(vals))


def _take_columns(X, cols):
    return X.iloc[:, cols] if hasattr(X, "iloc") else X[:, cols]


# ---------------------------------------------------------------------------
# the ensemble
# ---------------------------------------------------------------------------
@dataclass
class FeatureSubsampleEnsemble:
    """M members that differ ONLY in which columns they see.

    Every other axis is switched off (no column shuffle, no class shuffle, one
    normalisation), so the member spread is attributable to the column subset
    alone. Each member is a fresh fit, because the column set changes the fit.
    """
    n_estimators: int = 16
    frac: float = 0.5
    mode: str = "random"            # random | roundrobin | importance | lofo
    n_folds: int | None = None      # lofo only; M is forced to equal it
    min_features: int = 2
    device: str | None = None
    checkpoint: str = "tabicl-classifier-v2-20260212.ckpt"
    task: str = "classification"    # classification | regression
    seed: int = 0
    timings: dict = field(default_factory=dict)
    #: mirrored from the fitted members so run_cell records how the cached
    #: logits should be aggregated (log pooling at this temperature).
    average_logits: bool = True
    softmax_temperature: float = 0.9

    def __post_init__(self):
        if self.task == "regression" and "classifier" in self.checkpoint:
            self.checkpoint = self.checkpoint.replace("classifier", "regressor")

    def fit_predict_members(self, X_tr, y_tr, X_te) -> np.ndarray:
        """Return aligned member outputs.

        Classification: ``(M, n_test, n_classes)`` LOGITS. Regression:
        ``(M, n_test)`` predictions on the original target scale.
        """
        from experiments.capture import align_member, member_estimator

        y_tr = np.asarray(y_tr)
        rng = np.random.default_rng(self.seed)
        all_classes = None if self.task == "regression" else np.unique(y_tr)
        d = np.asarray(X_tr).shape[1]

        if self.mode == "lofo":
            # M is a consequence of the partition, not a free parameter.
            self.n_estimators = int(self.n_folds or self.n_estimators)
            self.frac = 1.0 - 1.0 / self.n_estimators

        imp = None
        t_imp = 0.0
        if self.mode == "importance":
            t0 = time.perf_counter()
            imp = column_importance(X_tr, y_tr, seed=self.seed, task=self.task)
            t_imp = time.perf_counter() - t0
        # a flat vector means the scorer fell back; record it rather than
        # silently reporting a uniform draw as importance-weighted.
        imp_ok = bool(imp is not None and np.ptp(imp) > 1e-9)

        subsets = feature_subsets(d, self.n_estimators, self.frac, rng,
                                  self.mode, imp, self.min_features)

        members, ref_classes = [], None
        t_fit = t_pred = 0.0
        sizes = [int(len(c)) for c in subsets]
        usage = np.zeros(d, dtype=np.int64)
        for cols in subsets:
            usage[cols] += 1

        for m, cols in enumerate(subsets):
            clf = member_estimator(self.task, self.checkpoint, self.seed + m, self.device)
            t0 = time.perf_counter()
            clf.fit(_take_columns(X_tr, cols), y_tr)
            t_fit += time.perf_counter() - t0

            ref_classes = align_member(self, clf, m, ref_classes, all_classes)

            t0 = time.perf_counter()
            p = clf.predict_members(_take_columns(X_te, cols))   # (1, n_test, C)
            t_pred += time.perf_counter() - t0
            members.append(p[0])
            del clf

        self.timings = dict(
            fit_seconds=t_fit,
            predict_seconds=t_pred,
            importance_seconds=t_imp,
            n_features_total=int(d),
            subset_size_mean=float(np.mean(sizes)),
            subset_size_min=int(np.min(sizes)),
            # measured over ALL pairs (see mean_pairwise_overlap); the expected
            # value depends on whether the design is balanced, so record both
            # references rather than one that is wrong for half the modes.
            pairwise_overlap_mean=mean_pairwise_overlap(subsets),
            pairwise_overlap_expected=(
                balanced_overlap(self.n_estimators, d, float(np.mean(sizes)))
                if self.mode in ("lofo", "roundrobin")
                else expected_overlap(float(np.mean(sizes)) / d)),
            pairwise_overlap_independent=expected_overlap(float(np.mean(sizes)) / d),
            mode=self.mode,
            frac_effective=float(np.mean(sizes)) / d,
            # coverage diagnostics: how evenly the columns were spread, and
            # whether any column was never shown to any member at all.
            n_unused_columns=int((usage == 0).sum()),
            usage_cv=float(np.std(usage) / max(1e-12, np.mean(usage))),
            importance_used=imp_ok,
        )
        return np.stack(members, axis=0)
