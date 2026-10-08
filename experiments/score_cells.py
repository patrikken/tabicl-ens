"""Score every cached cell, for either model and either task, in one pass.

    python -m experiments.score_cells CACHE OUT_DIR [--model tabicl|tabfm]
        CACHE = $CACHE_DIR (reads tabiclv2/ or tabfm/ by --model) or the model dir itself
        -> OUT_DIR/scores.csv   one row per (dataset, coalition, split) at full M,
                                with gain/rel_gain against that dataset's 'base'
        -> OUT_DIR/sweep.csv    the same cells at member budgets 1,2,4,...,32,
                                drawn post hoc from the stored members

Score convention (higher is better, never pooled across tasks):
    binary      ROC-AUC
    multiclass  mean log-likelihood (-log_loss)
    regression  -RMSE on the original target scale
:func:`elo.to_metric_error` converts these to TabArena's lower-is-better error.

Classification members are logits; TabICLv2 logits are clipped at +-80
(saturated, loses nothing), TabFM logits reach ~1e3 and are NEVER clipped (it
turns the top classes into argmax ties). Regression members are per-view
predictive means; the aggregator is the arithmetic mean over members, which is
what both regressors do with their views (TabFM with NNLS weights excepted --
that recipe is replayed only for the full-M row via its own recorded weights if
present, otherwise the plain mean; see ``agg`` below).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BUDGETS = (1, 2, 4, 8, 16, 32)
REPS = 5
CLIP = {"tabicl": 80.0, "tabfm": None}


def _softmax(x, t):
    z = np.asarray(x, dtype=np.float64) / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    s = e.sum(-1, keepdims=True)
    return np.divide(e, s, out=np.full_like(e, 1.0 / e.shape[-1]), where=s > 0)


def _repair(m: np.ndarray, clip: float | None) -> tuple[np.ndarray, dict]:
    n_nan, n_pos, n_neg = (int(np.isnan(m).sum()), int(np.isposinf(m).sum()),
                           int(np.isneginf(m).sum()))
    if n_nan or n_pos or n_neg:
        big = clip if clip else 6e4
        m = np.nan_to_num(m, nan=0.0, posinf=big, neginf=-big)
    if clip:
        m = np.clip(m, -clip, clip)
    return m, dict(n_nan=n_nan, n_posinf=n_pos, n_neginf=n_neg,
                   frac_nonfinite=(n_nan + n_pos + n_neg) / max(1, m.size))


def cell_score(task: str, pred: np.ndarray, y: np.ndarray, n_classes: int = 0) -> float:
    """pred: probabilities (T,C) for classification, predictions (T,) for regression."""
    from sklearn.metrics import log_loss, roc_auc_score
    if not np.isfinite(pred).all():
        return float("nan")
    if task == "regression":
        return -float(np.sqrt(np.mean((pred - y) ** 2)))
    try:
        if task == "binary":
            return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, pred[:, 1]))
        return -float(log_loss(y, pred, labels=np.arange(n_classes)))
    except ValueError:
        return float("nan")


def _combine(task: str, mem: np.ndarray, tau: float) -> np.ndarray:
    """Members -> what the cell predicts. Mean over members, in logit space for
    classification (then softmax at the recorded temperature)."""
    if task == "regression":
        return mem.mean(0)
    p = _softmax(mem.mean(0), tau)
    return p / p.sum(-1, keepdims=True)


def _task_and_target(meta: dict, y: np.ndarray):
    if meta.get("task") == "regression" or y.dtype.kind == "f" and meta.get("n_classes") is None:
        return "regression", np.asarray(y, dtype=np.float64), 0
    classes = np.unique(y)
    yi = np.searchsorted(classes, y)
    return ("binary" if len(classes) == 2 else "multiclass"), yi, len(classes)


def _effective_members(task: str, mem: np.ndarray, y: np.ndarray) -> float:
    """Effective number of independent members from the correlation of per-member
    errors (0/1 error for classification, residual for regression)."""
    M = mem.shape[0]
    err = (mem - y[None]) if task == "regression" else (mem.argmax(-1) != y[None]).astype(float)
    if M < 2 or err.std() == 0:
        return 1.0
    R = np.nan_to_num(np.corrcoef(err), nan=0.0)
    np.fill_diagonal(R, 1.0)
    return float(M ** 2 / (R ** 2).sum())


def full_prediction(d: Path, meta: dict, task: str, mem: np.ndarray, tau: float,
                    model: str):
    """What the cell predicts at full M, under the model's OWN aggregation rule.

    Returns ``(prediction, rule, recipe_ok)``.
      upstream  ``agg.npy``: the model's own aggregate (NNLS, pooling,
                calibration included). Always preferred when present.
      nnls      regression, NNLS weights recorded in meta -> ``w @ members``.
      mean      plain mean (logits then softmax at ``tau``): exactly the
                shipped rule for TabICLv2 and for TabFM without TabFM+ options.
      mean      with ``recipe_ok=False``: the cell used TabFM+ options
                (NNLS / probability pooling / calibration) that cannot be
                replayed from members alone. Its score is the PLAIN-MEAN score
                and must not be read as the TabFM+ recipe.
    """
    f = d / "agg.npy"
    if model == "tabfm" and f.exists():
        return np.load(f).astype(np.float64), "upstream", True
    if model == "tabfm":
        w = meta.get("nnls_weights")
        if task == "regression" and meta.get("enable_nnls") and w is not None:
            w = np.asarray(w, dtype=np.float64)
            if w.shape[0] == mem.shape[0]:
                return w @ mem, "nnls", True
        plus = (meta.get("enable_nnls") or meta.get("average_logits") is False
                or meta.get("calibration"))
        if task != "regression" and plus:
            return _combine(task, mem, tau), "mean", False
        if task == "regression" and meta.get("enable_nnls"):
            return _combine(task, mem, tau), "mean", False
    return _combine(task, mem, tau), "mean", True


def score_cell(d: Path, model: str, budgets=BUDGETS, reps=REPS):
    meta = json.loads((d / "meta.json").read_text())
    mem = np.load(d / "members.npy").astype(np.float32 if meta.get("task") != "regression"
                                            else np.float64)
    y = np.load(d / "y_test.npy", allow_pickle=True)
    task, yt, C = _task_and_target(meta, y)
    mem, health = _repair(mem, CLIP[model] if task != "regression" else None)
    M = mem.shape[0]
    tau = meta.get("softmax_temperature") or 0.9
    key = f'{meta["dataset"]}|{meta["coalition"]}|{meta["split"]}'
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))

    base = dict(dataset=meta["dataset"], coalition=meta["coalition"], split=meta["split"],
                task=task, n_classes=C, n_train=meta.get("n_train"),
                n_test=meta.get("n_test"), n_features=meta.get("n_features"))
    pred, rule, recipe_ok = full_prediction(d, meta, task, mem, tau, model)
    full = dict(**base, M=M, m_req=meta.get("n_estimators_requested"),
                truncated=meta.get("truncated"), severity=meta.get("split_severity"),
                **health, meff=_effective_members(task, mem, yt),
                agg_rule=rule, recipe_ok=recipe_ok,
                score=cell_score(task, pred, yt, C),
                score_single=cell_score(task, _combine(task, mem[:1], tau), yt, C),
                fit_s=meta.get("fit_seconds"), pred_s=meta.get("predict_seconds"),
                y_train_std=meta.get("y_train_std"))
    sweep = []
    # budgets above the cell's own M collapse onto M; keep each budget once, or a
    # 16-member A4/A5 cell would emit M=16 three times and break every join
    for b in ([1] if M == 1 else sorted({min(B, M) for B in budgets})):
        n = 1 if b == M else reps
        s = []
        for _ in range(n):
            idx = np.arange(M) if b == M else rng.choice(M, b, replace=False)
            s.append(cell_score(task, _combine(task, mem[idx], tau), yt, C))
        sweep.append(dict(**base, M=b, score=float(np.nanmean(s)) if not np.all(np.isnan(s))
                          else float("nan")))
    return full, sweep


def reference_of(coalition: str, model: str) -> str | None:
    """Coalition a "shipped + one axis" ablation is read against.

    TabICLv2 S_A4/S_A5/S_A7 -> ``S_ctl`` (same one-fit-per-member machinery and
    shipped-style views, nothing withheld), so wrapper overhead cancels.
    TabFM S_* -> ``shipped`` (the same fit with exactly one extra kwarg).
    ``S_ctl`` itself -> ``shipped``: that gain is the wrapper/schedule overhead
    and should be ~0; a large value means the S_* gains are not trustworthy.
    Everything else has no ablation reference (gain is vs ``base``)."""
    if coalition == "S_ctl":
        return "shipped"
    if coalition.startswith("S_"):
        return "S_ctl" if model == "tabicl" else "shipped"
    return None


def add_ablation_gain(df: pd.DataFrame, model: str, keys=("dataset", "split"),
                      extra=()) -> pd.DataFrame:
    """Add ``ref`` and ``gain_ref`` (= score - the ablation reference's score on the
    same dataset/split[/M]); NaN where a coalition has no reference or its
    reference was not run on that cell. Implemented as a merge on unique keys, so
    it can neither multiply rows nor fall back to a slow per-row lookup."""
    k = list(keys) + list(extra)
    out = df.copy()
    out["ref"] = [reference_of(c, model) for c in out.coalition]
    ref = (out.drop_duplicates(subset=k + ["coalition"])[k + ["coalition", "score"]]
           .rename(columns={"coalition": "ref", "score": "_ref_score"}))
    n = len(out)
    out = out.merge(ref, on=k + ["ref"], how="left")
    assert len(out) == n, "reference join changed the row count"
    out["gain_ref"] = out["score"] - out["_ref_score"]
    return out.drop(columns="_ref_score")


def add_gain(df: pd.DataFrame, keys=("dataset", "split"), extra=()) -> pd.DataFrame:
    """Gain over the same dataset/split 'base' coalition (same M budget for the sweep).
    ``rel_gain`` is gain / |base score| for regression (relative RMSE reduction)
    and NaN otherwise -- AUC and log-loss differences are already comparable."""
    k = list(keys) + list(extra)
    b = (df[df.coalition == "base"].drop_duplicates(subset=k)
         .set_index(k)["score"].rename("base"))
    out = df.join(b, on=k)
    out["gain"] = out["score"] - out["base"]
    out["rel_gain"] = np.where(out["task"] == "regression",
                               out["gain"] / out["base"].abs().replace(0, np.nan), np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("out_dir")
    ap.add_argument("--model", choices=list(CLIP), default="tabicl")
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full, sweep = [], []
    # CACHE may be $CACHE_DIR itself or the model's own directory
    sub = {"tabicl": "tabiclv2", "tabfm": "tabfm"}[a.model]
    if (cache / sub).is_dir():
        cache = cache / sub
    metas = sorted(cache.rglob("meta.json"))
    # a legacy flat cache (cells at the root) can sit beside the model dirs
    metas = [m for m in metas if not {"tabfm", "tabiclv2"} & set(m.relative_to(cache).parts)
             or cache.name in ("tabfm", "tabiclv2")]
    for i, mf in enumerate(metas):
        f, s = score_cell(mf.parent, a.model)
        full.append(f)
        sweep += s
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(metas)}", flush=True)
    df, sw = pd.DataFrame(full), pd.DataFrame(sweep)
    cell = ["dataset", "coalition", "split"]
    dup = df.duplicated(subset=cell, keep="first")
    if dup.any():
        print(f"WARNING: {int(dup.sum())} duplicate (dataset, coalition, split) cells "
              f"(same cell under two directories?); keeping the first of each")
        df = df[~dup].reset_index(drop=True)
    sw = sw.drop_duplicates(subset=cell + ["M"]).reset_index(drop=True)
    df = add_ablation_gain(add_gain(df), a.model)
    sw = add_ablation_gain(add_gain(sw, extra=("M",)), a.model, extra=("M",))
    df.to_csv(out / "scores.csv", index=False)
    sw.to_csv(out / "sweep.csv", index=False)
    print(f"{len(df)} cells, {len(sw)} sweep rows -> {out}")
    print(df.groupby(["task", "coalition"])["gain"].mean().unstack(0).round(4).to_string())
    ab = df[df.ref.notna()]
    if len(ab):
        print("\nablations (gain over their reference):")
        print(ab.groupby(["task", "coalition", "ref"])["gain_ref"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
