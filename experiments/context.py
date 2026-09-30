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
    mode: str = "random"
    min_per_class: int = 4
    device: str | None = None
    checkpoint: str = "tabicl-classifier-v2-20260212.ckpt"
    seed: int = 0
    timings: dict = field(default_factory=dict)
    #: mirrored from the fitted members so run_cell can record how the cached
    #: logits should be aggregated (log pooling at this temperature).
    average_logits: bool = True
    softmax_temperature: float = 0.9

    def fit_predict_members(self, X_tr, y_tr, X_te) -> np.ndarray:
        """Return aligned ``(M, n_test, n_classes)`` member LOGITS."""
        from experiments.capture import MemberCapturingTabICLClassifier

        y_tr = np.asarray(y_tr)
        rng = np.random.default_rng(self.seed)
        all_classes = np.unique(y_tr)

        members, ref_classes = [], None
        t_fit = t_pred = 0.0
        sizes, overlaps, prev = [], [], None

        for m in range(self.n_estimators):
            idx = subsample_indices(y_tr, self.frac, rng, self.mode,
                                    self.min_per_class)
            sizes.append(int(len(idx)))
            if prev is not None:
                overlaps.append(len(np.intersect1d(idx, prev)) / max(1, len(idx)))
            prev = idx

            clf = MemberCapturingTabICLClassifier(
                n_estimators=1,
                feat_shuffle_method="none",
                class_shuffle_method="none",
                norm_methods=["none"],
                checkpoint_version=self.checkpoint,
                random_state=self.seed + m,
                device=self.device,
            )
            t0 = time.perf_counter()
            clf.fit(X_tr.iloc[idx] if hasattr(X_tr, "iloc") else X_tr[idx],
                    y_tr[idx])
            t_fit += time.perf_counter() - t0

            cls = np.asarray(clf.classes_)
            if ref_classes is None:
                ref_classes = cls
                self.average_logits = bool(clf.average_logits)
                self.softmax_temperature = float(clf.softmax_temperature)
                if len(cls) != len(all_classes):
                    raise RuntimeError(
                        f"member 0 fitted {len(cls)} classes but the pool has "
                        f"{len(all_classes)}; raise min_per_class or frac.")
            elif not np.array_equal(cls, ref_classes):
                raise RuntimeError(
                    f"member {m} fitted a different class set ({cls} vs "
                    f"{ref_classes}); member columns would not align.")

            t0 = time.perf_counter()
            p = clf.predict_members(X_te)          # (1, n_test, C)
            t_pred += time.perf_counter() - t0
            members.append(p[0])
            del clf

        self.timings = dict(
            fit_seconds=t_fit, predict_seconds=t_pred,
            context_size_mean=float(np.mean(sizes)),
            context_size_min=int(np.min(sizes)),
            pairwise_overlap_mean=float(np.mean(overlaps)) if overlaps else 0.0,
            pairwise_overlap_expected=expected_overlap(self.frac),
        )
        return np.stack(members, axis=0)
