"""Bradley-Terry ("Elo") ratings, matching TabArena's protocol exactly.

WHY REIMPLEMENT RATHER THAN IMPORT
----------------------------------
``bencheval`` is standalone (scipy/numpy/pandas/sklearn only) and *is* importable
on a cluster, so importing it is the right default -- :func:`elo_ratings` does
exactly that when it can. But the official package pulls the *results pool*
through ``tabarena``, which hard-depends on ``autogluon.tabular``, and that is
the install that fails on Compute Canada. This module therefore carries a
faithful reimplementation of the rank path so the analysis runs with nothing but
numpy/scipy/pandas, and :func:`crosscheck_against_bencheval` asserts the two
agree whenever both are available. Never trust the reimplementation on its own
in the paper -- run the cross-check and report that it passed.

THE PROTOCOL, as read from
``packages/bencheval/src/bencheval/elo_utils.py`` @ tabarena e8cd3ca5:

* unit of comparison = ``(dataset, fold)``; a method's wins on a unit are
  ``n_methods - rank``, ranks averaged so **ties count half**;
* each unit is weighted ``1 / n_splits(dataset)`` and summed within its dataset,
  so **every dataset carries total weight 1** however many folds it ran;
* with a complete balanced schedule the win totals are a sufficient statistic, so
  the likelihood is ``sum_i W_i t_i - n_tasks * sum_{i<j} log(e^{t_i}+e^{t_j})``,
  maximised in log-strengths by L-BFGS-B with an analytic gradient;
* a ridge ``BT_RIDGE`` matching ``LogisticRegression(C=1e6)`` keeps a separable
  field finite;
* ratings are ``400 * log10(strength)`` centred on 1000, then shifted so the
  anchor method lands on ``calibration_elo``;
* the headline rating is the **single-fit MLE**, not the bootstrap median
  (``use_bootstrap_median=False`` is bencheval's default, despite what the
  TabArena paper's wording suggests). The bootstrap, over **datasets** with
  replacement, supplies the CI only.

``metric_error`` is always lower-is-better and >= 0. Feeding a
higher-is-better score ranks the field backwards in silence, so
:func:`to_metric_error` is the only sanctioned way to convert our scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCALE = 400.0
INIT_RATING = 1000.0
#: matches the L2 that ``LogisticRegression(C=1e6)`` applies in the battle path
BT_RIDGE = 0.5 / (1e6 * math.log(10) ** 2)
SOLVER_TOL = 1e-10
ANCHOR = "RF (default)"
BOOTSTRAP_ROUNDS = 200          # TabArenaContext.evaluate_all's default


# ---------------------------------------------------------------- conversion --
def to_metric_error(score: np.ndarray | pd.Series, task: np.ndarray | pd.Series
                    ) -> np.ndarray:
    """Our score convention -> TabArena's ``metric_error`` (lower is better).

    Ours (both higher-is-better, never pooled):
        binary      ROC-AUC
        multiclass  mean log-likelihood, i.e. -log_loss
    TabArena's, per its task metadata ``eval_metric``:
        binary      roc_auc     -> metric_error = 1 - AUC
        multiclass  log_loss    -> metric_error = log_loss = -score

    Verified against the shipped TabFM parquet, whose ``roc_auc`` rows carry
    1 - AUC (e.g. Amazon_employee_access fold 0 at 0.1493).
    """
    score = np.asarray(score, dtype=float)
    task = np.asarray(task)
    out = np.where(task == "binary", 1.0 - score, -score)
    if not np.isfinite(out).all():
        raise ValueError("non-finite metric_error; sanitize the scores first")
    if (out < -1e-12).any():
        bad = out.min()
        raise ValueError(
            f"negative metric_error ({bad:.3e}). TabArena requires >= 0 -- check "
            f"the task labels, or that multiclass scores really are -log_loss.")
    return np.clip(out, 0.0, None)


# ------------------------------------------------------------------ the fit --
def _win_totals(df: pd.DataFrame, method_col: str, task_col: str,
                split_col: str, error_col: str) -> tuple[pd.Index, np.ndarray]:
    """Per-dataset weighted win totals, shape ``(n_methods, n_tasks)``.

    Mirrors ``EloHelper._rank_win_matrix``. Raises on a ragged schedule rather
    than imputing, because win totals are a sufficient statistic only when every
    method ran every unit -- silently filling would give a different answer that
    still looks like an Elo.
    """
    wide = df.pivot_table(index=method_col, columns=[task_col, split_col],
                          values=error_col)
    if wide.isna().to_numpy().any():
        n = int(wide.isna().to_numpy().sum())
        miss = wide.isna().sum(axis=1).sort_values(ascending=False)
        raise ValueError(
            f"ragged schedule: {n} (method, unit) pairs missing. Worst offenders:\n"
            f"{miss[miss > 0].head(10).to_string()}\n"
            f"Restrict to the units every method ran (see complete_subgrid).")
    if len(wide) < 2:
        raise ValueError(f"need >= 2 methods, got {len(wide)}")

    n_methods = len(wide)
    wins_per_unit = n_methods - wide.rank(axis=0, method="average",
                                          ascending=True).to_numpy()
    tasks = wide.columns.get_level_values(task_col).to_numpy()
    _, task_of_unit = np.unique(tasks, return_inverse=True)
    n_tasks = int(task_of_unit.max()) + 1
    weights = 1.0 / np.bincount(task_of_unit)[task_of_unit]
    onehot = np.zeros((len(task_of_unit), n_tasks))
    onehot[np.arange(len(task_of_unit)), task_of_unit] = 1.0
    return wide.index, (wins_per_unit * weights) @ onehot


def _bradley_terry(wins: np.ndarray, n_tasks: int, init: np.ndarray | None = None,
                   max_iter: int = 10_000, tol: float = 1e-12) -> np.ndarray:
    """Maximise the BT likelihood in log-strengths. Mirrors
    ``EloHelper._bradley_terry_from_win_totals``."""
    n = len(wins)
    off_diag = ~np.eye(n, dtype=bool)
    upper = np.triu_indices(n, 1)

    def nll(t):
        largest = np.maximum(t[:, None], t[None, :])
        log_sum = largest + np.log(np.exp(t[:, None] - largest)
                                   + np.exp(t[None, :] - largest))
        ll = wins @ t - n_tasks * log_sum[upper].sum()
        p_win = np.exp(t[:, None] - log_sum)
        grad = wins - n_tasks * np.where(off_diag, p_win, 0.0).sum(axis=1)
        return -ll + BT_RIDGE * (t @ t), -grad + 2 * BT_RIDGE * t

    res = minimize(nll, np.zeros(n) if init is None else np.asarray(init, float),
                   jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": tol * 1e-3, "gtol": tol})
    t = res.x - res.x.mean()
    return SCALE * (t / np.log(10)) + INIT_RATING


@dataclass
class EloResult:
    ratings: pd.DataFrame          # index method; columns elo, elo_lo, elo_hi
    n_methods: int
    n_tasks: int
    n_units: int
    anchor: str | None


def elo_ratings(df: pd.DataFrame, *, method_col: str = "method",
                task_col: str = "dataset", split_col: str = "fold",
                error_col: str = "metric_error", anchor: str | None = ANCHOR,
                anchor_elo: float = INIT_RATING,
                bootstrap_rounds: int = BOOTSTRAP_ROUNDS, seed: int = 0,
                prefer_bencheval: bool = True) -> EloResult:
    """Elo for every method in ``df``, with a bootstrap CI over datasets.

    ``df`` is long form: one row per (method, dataset, fold) with a
    lower-is-better ``metric_error``. Uses ``bencheval`` when importable and
    ``prefer_bencheval``; otherwise the reimplementation above.
    """
    if prefer_bencheval:
        try:
            return _elo_via_bencheval(df, method_col, task_col, split_col,
                                      error_col, anchor, anchor_elo,
                                      bootstrap_rounds)
        except ImportError:
            pass

    methods, W = _win_totals(df, method_col, task_col, split_col, error_col)
    n_tasks = W.shape[1]
    point = _bradley_terry(W.sum(axis=1), n_tasks)

    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap_rounds, len(methods)))
    log_init = (point - INIT_RATING) * np.log(10) / SCALE   # warm start
    for b in range(bootstrap_rounds):
        counts = np.bincount(rng.choice(n_tasks, size=n_tasks, replace=True),
                             minlength=n_tasks)
        boot[b] = _bradley_terry(W @ counts, n_tasks, init=log_init)

    out = pd.DataFrame({"elo": point,
                        "elo_lo": np.quantile(boot, 0.025, axis=0),
                        "elo_hi": np.quantile(boot, 0.975, axis=0)},
                       index=methods)
    # post_calibrate: the CI is taken uncalibrated, then everything is shifted,
    # so the +/- bars do not depend on which method anchors the scale.
    if anchor is not None:
        if anchor not in out.index:
            raise KeyError(f"anchor {anchor!r} absent; have {list(out.index)[:8]}...")
        out += anchor_elo - out.loc[anchor, "elo"]
    return EloResult(out.sort_values("elo", ascending=False), len(methods),
                     n_tasks, int(df.groupby([task_col, split_col]).ngroups),
                     anchor)


def _elo_via_bencheval(df, method_col, task_col, split_col, error_col,
                       anchor, anchor_elo, bootstrap_rounds) -> EloResult:
    from bencheval.evaluator import BenchmarkEvaluator      # noqa: PLC0415

    ev = BenchmarkEvaluator(method_col=method_col, task_col=task_col,
                            seed_column=split_col, error_col=error_col,
                            columns_to_agg_extra=[])
    per_task = ev.compute_results_per_task(data=df, include_seed_col=True)
    bars = ev.compute_elo(per_task, calibration_framework=anchor,
                          calibration_elo=anchor_elo,
                          BOOTSTRAP_ROUNDS=bootstrap_rounds,
                          include_quantiles=True, round_decimals=None)
    out = pd.DataFrame({"elo": bars["elo"],
                        "elo_lo": bars["elo"] - bars.get("elo-", 0.0),
                        "elo_hi": bars["elo"] + bars.get("elo+", 0.0)})
    return EloResult(out.sort_values("elo", ascending=False), df[method_col].nunique(),
                     df[task_col].nunique(),
                     int(df.groupby([task_col, split_col]).ngroups), anchor)


def crosscheck_against_bencheval(df: pd.DataFrame, *, tol: float = 1e-6, **kw
                                 ) -> pd.DataFrame | None:
    """Run both implementations and return their per-method difference.

    Returns ``None`` when ``bencheval`` is not installed. Raises when the point
    estimates disagree by more than ``tol`` Elo -- at which point the
    reimplementation, not the official package, is what is wrong.
    """
    try:
        import bencheval  # noqa: F401,PLC0415
    except ImportError:
        return None
    a = elo_ratings(df, prefer_bencheval=False, **kw).ratings["elo"]
    b = elo_ratings(df, prefer_bencheval=True, **kw).ratings["elo"]
    d = pd.DataFrame({"ours": a, "bencheval": b.reindex(a.index)})
    d["diff"] = d.ours - d.bencheval
    worst = d["diff"].abs().max()
    if worst > tol:
        raise AssertionError(
            f"Elo reimplementation disagrees with bencheval by {worst:.4g} Elo "
            f"(tol {tol:g}):\n{d.sort_values('diff', key=abs, ascending=False).head()}")
    return d


# -------------------------------------------------------------- bookkeeping --
def complete_subgrid(df: pd.DataFrame, *, method_col: str = "method",
                     task_col: str = "dataset", split_col: str = "fold",
                     verbose: bool = True) -> pd.DataFrame:
    """Restrict to the ``(dataset, fold)`` units that EVERY method ran.

    The rank path requires a complete grid. Our coalitions cover fewer datasets
    than the arena pool (A4 skips 3 for having < 8 columns, column-LOFO more
    still), so this is the honest intersection rather than an imputation. Elo is
    pool- and task-dependent, so always report what was dropped.
    """
    units = df.groupby([task_col, split_col])[method_col].nunique()
    full = units[units == df[method_col].nunique()].index
    out = df.set_index([task_col, split_col]).loc[full].reset_index()
    if verbose:
        kept, tot = len(full), len(units)
        print(f"[complete_subgrid] {kept}/{tot} units kept "
              f"({out[task_col].nunique()}/{df[task_col].nunique()} datasets, "
              f"{out[method_col].nunique()} methods)")
        lost = sorted(set(df[task_col]) - set(out[task_col]))
        if lost:
            print(f"[complete_subgrid] datasets dropped entirely: {', '.join(lost)}")
    return out


def win_rate(elo_a: float, elo_b: float) -> float:
    """Expected win rate of A over B, the BT link. Useful for stating what an
    Elo gap *means* -- +100 Elo is a 64% win rate, not 'a hundred better'."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / SCALE))
