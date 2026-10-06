"""A5 — context-side perturbation, size-preserving.

Every other axis re-presents the SAME rows (different column order, class order,
normalisation). A5 changes *which rows the model conditions on*, which is the
only axis that touches what an in-context learner retrieves over. That makes it
the axis with a mechanistic prediction attached (H1, the retrieval account of
tabular ICL).

THE CONFOUND, AND THE DESIGN THAT AVOIDS IT
-------------------------------------------
Subsampling the context does not only add diversity, it removes information: a
member holding 50% of the rows is a strictly weaker predictor. Comparing an A5
ensemble against the FULL-context single pass therefore mixes two effects with
opposite signs and can report a negative result for an axis that is in fact
decorrelating well.

So every A5 coalition holds the context size FIXED at a fraction ``f`` and
varies only *which* rows are drawn. That yields two comparisons from one run:

* **matched** — vs the same coalition at M=1, i.e. one member at the same
  context size. Isolates the diversity effect. This is the H1 test, and it is
  directly comparable to the feature-side axes' own M=1 baselines.
* **absolute** — vs ``base``, the full-context single pass. The practical
  question: is this worth doing at all.

Sweeping ``f`` is the diagnostic. As f→1 members converge on the same rows, so
diversity→0 while member quality→max. If matched gain *rises* with f, the gain
is not coming from context diversity.

CLASS COVERAGE
--------------
A draw that misses a class changes the model's fitted class set, so member
outputs would no longer be column-aligned and the ensemble would be silently
wrong. Every draw is therefore stratified with a floor of ``min_per_class``, and
``ContextEnsemble`` asserts that all members fitted the identical class vector.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def strata(y, task: str = "classification", max_bins: int = 10,
           min_bin: int = 20) -> np.ndarray:
    """Stratification labels: the classes themselves, or quantile bins of y.

    Regression has no classes, but the context draws still need to keep the
    target distribution (a random draw can miss the tails, and then a member is
    a different problem, not a different view). Quantile bins give the same
    stratified-with-floor and class-balanced constructions a meaning there.
    """
    y = np.asarray(y)
    if task != "regression":
        return y
    n = len(y)
    nb = int(np.clip(n // min_bin, 2, max_bins))
    edges = np.unique(np.quantile(y.astype(float), np.linspace(0, 1, nb + 1)[1:-1]))
    return np.digitize(y.astype(float), edges)


def subsample_indices(y, frac: float, rng: np.random.Generator,
                      mode: str = "random", min_per_class: int = 4) -> np.ndarray:
    """Row indices for one context draw of size ~``frac`` of the pool.

    ``random``   stratified only by the ``min_per_class`` floor; the remainder
                 is drawn uniformly, so class proportions follow the data.
    ``balanced`` equal count per class -- the group-balanced construction from
                 the fair-ICL context literature, included so the axis covers
                 more than uniform resampling.
    """
    y = np.asarray(y)
    classes, inv = np.unique(y, return_inverse=True)
    n, C = len(y), len(classes)
    pools = [np.where(inv == c)[0] for c in range(C)]
    target = max(min_per_class * C, int(round(frac * n)))

    if mode == "balanced":
        per = max(min_per_class, target // C)
        take = [rng.choice(p, min(per, len(p)), replace=False) for p in pools]
        return np.sort(np.concatenate(take))

    if mode != "random":
        raise ValueError(f"unknown mode {mode!r}")

    forced = np.concatenate(
        [rng.choice(p, min(min_per_class, len(p)), replace=False) for p in pools])
    rest = np.setdiff1d(np.arange(n), forced, assume_unique=False)
    need = min(max(0, target - len(forced)), len(rest))
    extra = rng.choice(rest, need, replace=False) if need else np.array([], int)
    return np.sort(np.concatenate([forced, extra]).astype(np.int64))


def expected_overlap(frac: float) -> float:
    """Expected pairwise row overlap between two independent draws.

    Two independent draws of fraction f share ~f of their rows, so f is also the
    knob that trades member quality against member diversity.
    """
    return float(frac)


def lofo_partition(y, n_folds: int, rng: np.random.Generator,
                   pin_below: int = 2) -> tuple[list[np.ndarray], np.ndarray]:
    """Leave-one-fold-out partition: member m holds every row EXCEPT fold m.

    Unlike the independent draws above, this ties the context fraction to the
    member count:

        f = 1 - 1/M,  pairwise overlap = (M-2)/(M-1)

    which is the one-parameter family our fixed-M fraction sweep could not
    reach. It is also the maximum-f construction for a given M, and it
    guarantees every row is used by exactly M-1 members.

    Folds are assigned round-robin WITHIN each class, so every fold is
    stratified and each member's context keeps the class proportions. A class
    with fewer than ``pin_below`` rows cannot survive being held out at all, so
    those rows are pinned into every member rather than dropped -- without this
    a singleton class vanishes from one member and the member outputs stop
    being column-aligned.

    Returns (member index arrays, pinned row indices).
    """
    y = np.asarray(y)
    classes, inv = np.unique(y, return_inverse=True)
    n = len(y)
    counts = np.bincount(inv, minlength=len(classes))

    fold = np.full(n, -1, dtype=np.int64)        # -1 = pinned, never held out
    for c in range(len(classes)):
        idx = np.where(inv == c)[0]
        if counts[c] < pin_below:
            continue                              # leave at -1
        idx = rng.permutation(idx)
        fold[idx] = np.arange(len(idx)) % n_folds

    pinned = np.where(fold == -1)[0]
    members = [np.sort(np.where(fold != m)[0]) for m in range(n_folds)]
    return members, pinned


@dataclass
class ContextEnsemble:
    """M members that differ ONLY in which context rows they see.

    Each member is a fresh fit with every other axis switched off
    (no column shuffle, no class shuffle, one normalisation), so the member
    spread is attributable to the context draw alone.

    Unlike the feature-side axes -- one fit, M cheap forward passes -- this
    needs **M fits**, because the context IS the training data for an in-context
    learner. That cost is real and is what the per-member timings will show.
    """
    n_estimators: int = 16
    frac: float = 0.5
    mode: str = "random"          # random | balanced | lofo
    n_folds: int | None = None    # lofo only; M is forced to equal it
    min_per_class: int = 4
    device: str | None = None
    checkpoint: str = "tabicl-classifier-v2-20260212.ckpt"
    task: str = "classification"   # classification | regression
    seed: int = 0
    timings: dict = field(default_factory=dict)
    #: mirrored from the fitted members so run_cell can record how the cached
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
        reg = self.task == "regression"
        all_classes = None if reg else np.unique(y_tr)
        strat = strata(y_tr, self.task)

        lofo_members = None
        if self.mode == "lofo":
            # M is a consequence of the partition, not a free parameter.
            self.n_estimators = int(self.n_folds or self.n_estimators)
            self.frac = 1.0 - 1.0 / self.n_estimators
            lofo_members, pinned = lofo_partition(strat, self.n_estimators, rng)
            self._n_pinned = int(len(pinned))

        members, ref_classes = [], None
        t_fit = t_pred = 0.0
        sizes, overlaps, prev = [], [], None

        for m in range(self.n_estimators):
            idx = (lofo_members[m] if lofo_members is not None
                   else subsample_indices(strat, self.frac, rng, self.mode,
                                          self.min_per_class))
            sizes.append(int(len(idx)))
            if prev is not None:
                overlaps.append(len(np.intersect1d(idx, prev)) / max(1, len(idx)))
            prev = idx

            clf = member_estimator(self.task, self.checkpoint, self.seed + m, self.device)
            t0 = time.perf_counter()
            clf.fit(X_tr.iloc[idx] if hasattr(X_tr, "iloc") else X_tr[idx],
                    y_tr[idx])
            t_fit += time.perf_counter() - t0

            ref_classes = align_member(self, clf, m, ref_classes, all_classes)

            t0 = time.perf_counter()
            p = clf.predict_members(X_te)          # (1, n_test[, C])
            t_pred += time.perf_counter() - t0
            members.append(p[0])
            del clf

        self.timings = dict(
            fit_seconds=t_fit, predict_seconds=t_pred,
            context_size_mean=float(np.mean(sizes)),
            context_size_min=int(np.min(sizes)),
            pairwise_overlap_mean=float(np.mean(overlaps)) if overlaps else 0.0,
            pairwise_overlap_expected=(
                (self.n_estimators - 2) / (self.n_estimators - 1)
                if self.mode == "lofo" and self.n_estimators > 1
                else expected_overlap(self.frac)),
            mode=self.mode,
            frac_effective=self.frac,
            n_pinned=getattr(self, "_n_pinned", 0),
        )
        return np.stack(members, axis=0)
