"""Fetch and assemble the real TabArena comparison pool.

TabArena's leaderboard is a Bradley-Terry fit over a *pool* of method
configurations, so an Elo number is meaningless without saying which pool it was
computed in. This module gets the published pool so our coalitions can be rated
against the same field the papers quote.

WHY NOT JUST `pip install tabarena`
-----------------------------------
``tabarena`` hard-depends on ``autogluon.tabular``, which is the install that
fails on the cluster -- the same reason we never installed it for the splits.
But the results tier is just parquet files behind a public Cloudflare R2 domain,
addressed by a key the repo ships a checksum manifest for, so plain HTTP plus
pandas is enough. ``bencheval`` (standalone: scipy/numpy/pandas/sklearn) can
still be installed separately and is what ``elo.py`` cross-checks against.

    local  <out>/<suite>/<method>/<artifact>.parquet
    remote https://data.tabarena.ai/cache/artifacts/<suite>/methods/<method>/results/<artifact>.parquet

read from ``MethodDownloader.local_to_key`` and ``MethodDownloaderPublicR2``
(tabarena @ e8cd3ca5) and cross-checked against
``models/_artifacts/results_checksums.json``, which lists every published
artifact with its md5.

LOGIN NODE ONLY -- compute nodes have no internet. Fetch once, commit the
directory to $PROJECT_ROOT, and every later run is offline.

THE FROZEN v0.1 POOL is suite ``tabarena-2025-06-12``: 42 artifacts over 22
methods, each expanding to (default) / (tuned) / (tuned + ensemble) rows, which
is where the ~67 configurations in the TabArena and TabFM papers come from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://data.tabarena.ai/cache/"
FROZEN_SUITE = "tabarena-2025-06-12"       # the NeurIPS-2025 / v0.1 leaderboard
#: shipped with the repo; `arena.py checksums` refreshes it from a tabarena clone
CHECKSUMS = Path(__file__).with_name("data") / "results_checksums.json"

#: columns we need; the published frames carry more and the set has drifted
#: between suites, so select rather than assume.
NEEDED = ["method", "dataset", "fold", "metric_error", "metric", "problem_type"]


# ------------------------------------------------------------------- fetch --
def artifact_keys(suite: str = FROZEN_SUITE,
                  checksums: Path = CHECKSUMS) -> dict[str, str]:
    """``{key: md5}`` for every published results artifact in ``suite``."""
    if not checksums.exists():
        raise FileNotFoundError(
            f"{checksums} missing. Copy it from a tabarena clone:\n"
            f"  cp <tabarena>/packages/tabarena/src/tabarena/models/_artifacts/"
            f"results_checksums.json {checksums}")
    all_keys = json.loads(checksums.read_text())
    return {k: v for k, v in all_keys.items() if k.split("/")[1] == suite}


def fetch_pool(out: Path, suite: str = FROZEN_SUITE, *, timeout: float = 120.0,
               force: bool = False) -> Path:
    """Download every results parquet for ``suite``. Verifies md5. Resumable.

    Returns the suite directory. Total is well under 100 MB -- this is the
    results tier (per-(method, dataset, fold) metric values), not the processed
    tier (10 GB/method, carries predictions) or raw (~100 GB/method).
    """
    keys = artifact_keys(suite)
    if not keys:
        raise ValueError(f"no artifacts for suite {suite!r}; "
                         f"known: {sorted({k.split('/')[1] for k in json.loads(CHECKSUMS.read_text())})}")
    root = out / suite
    root.mkdir(parents=True, exist_ok=True)
    ok = skipped = failed = 0

    for key, want_md5 in sorted(keys.items()):
        method, artifact = key.split("/")[3], key.split("/")[-1]
        dest = root / method / artifact
        if dest.exists() and not force:
            if hashlib.md5(dest.read_bytes()).hexdigest() == want_md5:
                skipped += 1
                continue
            print(f"  [stale] {method}/{artifact}, re-downloading")
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = BASE_URL + key
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310
                blob = r.read()
        except Exception as e:                                        # noqa: BLE001
            print(f"  [FAIL] {method}/{artifact}: {type(e).__name__}: {e}")
            failed += 1
            continue
        got = hashlib.md5(blob).hexdigest()
        if got != want_md5:
            print(f"  [FAIL] {method}/{artifact}: md5 {got} != {want_md5}")
            failed += 1
            continue
        dest.write_bytes(blob)
        ok += 1
        print(f"  [ok]   {method}/{artifact}  {len(blob)/1e6:.2f} MB")

    print(f"\n{ok} downloaded, {skipped} already present, {failed} failed -> {root}")
    if failed:
        print("Failures are usually no-internet (compute node) or a changed key. "
              "Re-run on a login node; the download is resumable.")
    return root


# -------------------------------------------------------------------- load --
def load_pool(root: Path, *, problem_types: tuple[str, ...] = ("binary", "multiclass"),
              verbose: bool = True) -> pd.DataFrame:
    """Every downloaded artifact as one long frame in the leaderboard schema."""
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}; run `arena.py fetch` first")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        missing = [c for c in NEEDED if c not in df.columns]
        if missing:
            if verbose:
                print(f"  [skip] {f.parent.name}/{f.name}: missing {missing}")
            continue
        frames.append(df[NEEDED])
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool.problem_type.isin(problem_types)]
    pool = pool.drop_duplicates(subset=["method", "dataset", "fold"])
    if verbose:
        print(f"[pool] {len(pool):,} rows  {pool.method.nunique()} methods  "
              f"{pool.dataset.nunique()} datasets  "
              f"{pool.groupby(['dataset','fold']).ngroups} units")
    return pool


def load_tabfm_shipped(tabfm_repo: Path, *, verbose: bool = True) -> pd.DataFrame:
    """TabFM's own TabArena numbers, from ``results/`` in the google-research clone.

    Two methods: ``TabFM`` (single pass) and ``TabFM-Ensemble`` (the TabFM+
    stack). Already in the leaderboard schema, already ``metric_error``.

    CAVEAT for the write-up: these are **JAX/TPU** runs. Our own TabFM cells go
    through the PyTorch checkpoint, so the two are not bitwise comparable and
    should carry different method names in the pool.
    """
    files = sorted((tabfm_repo / "results").glob("*classification*.parquet"))
    if not files:
        raise FileNotFoundError(f"no classification parquet in {tabfm_repo/'results'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[[c for c in NEEDED if c in df.columns]]
    if verbose:
        print(f"[tabfm shipped] {len(df)} rows  methods={sorted(df.method.unique())}")
    return df


# ---------------------------------------------------------- our coalitions --
def our_results(scores: pd.DataFrame, *, model: str, budget: int | None = None,
                coalitions: list[str] | None = None,
                name: str = "{model} {coalition} (M={M})") -> pd.DataFrame:
    """Our per-cell scores -> the leaderboard schema.

    ``scores`` is any of our ``*_scores.csv`` frames: one row per
    (dataset, coalition, split, budget). Our ``split`` IS TabArena's ``fold`` --
    ``prepare_tabarena.py`` enumerates that dataset's (repeat, fold) rows from
    the official task table in order, which is the same flattening the published
    results use (26 datasets x 9 splits + 12 x 30 = 594 classification units).

    ``budget=None`` keeps each coalition at its own maximum budget; pass an int
    to compare every coalition at a matched member count.
    """
    from elo import to_metric_error                          # noqa: PLC0415

    df = scores.copy()
    if coalitions is not None:
        df = df[df.coalition.isin(coalitions)]
    if budget is None:
        df = df[df.budget == df.groupby("coalition").budget.transform("max")]
    else:
        df = df[df.budget == budget]
        if df.empty:
            raise ValueError(f"no cells at budget={budget}")

    out = pd.DataFrame({
        "method": [name.format(model=model, coalition=c, M=int(m))
                   for c, m in zip(df.coalition, df.budget)],
        "dataset": df.dataset.to_numpy(),
        "fold": df.split.to_numpy(),
        "metric_error": to_metric_error(df.score, df.task),
        "metric": np.where(df.task.to_numpy() == "binary", "roc_auc", "log_loss"),
        "problem_type": df.task.to_numpy(),
    })
    return out.drop_duplicates(subset=["method", "dataset", "fold"])


def effect_sizes(scores: pd.DataFrame, *, baseline: str = "base",
                 budget: int | None = None) -> pd.DataFrame:
    """Per-method mean gain over ``baseline``, in the native metric.

    This is the column that goes NEXT TO Elo. Reported per task type, never
    pooled: binary is ROC-AUC, multiclass is -log-loss, and averaging them is
    the mistake that started this whole line of work.
    """
    base = (scores[scores.coalition == baseline]
            .groupby(["dataset", "split"]).score.mean().rename("base").reset_index())
    df = scores.copy()
    if budget is None:
        df = df[df.budget == df.groupby("coalition").budget.transform("max")]
    else:
        df = df[df.budget == budget]
    d = df.merge(base, on=["dataset", "split"])
    d["gain"] = d.score - d.base
    g = (d.groupby(["coalition", "task"])
           .agg(M=("budget", "max"), gain=("gain", "mean"),
                n_ds=("dataset", "nunique"))
           .reset_index())
    per_ds = (d.groupby(["coalition", "task", "dataset"]).gain.mean()
                .groupby(level=[0, 1]).apply(lambda s: (s > 0).sum()).rename("helps"))
    return g.merge(per_ds.reset_index(), on=["coalition", "task"])


# --------------------------------------------------------------------- CLI --
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download the pool (LOGIN NODE ONLY)")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--suite", default=FROZEN_SUITE)
    f.add_argument("--force", action="store_true")

    i = sub.add_parser("inspect", help="summarise a downloaded pool")
    i.add_argument("--pool", type=Path, required=True)

    c = sub.add_parser("checksums", help="copy the manifest out of a tabarena clone")
    c.add_argument("--tabarena", type=Path, required=True)

    a = p.parse_args(argv)
    if a.cmd == "fetch":
        fetch_pool(a.out, a.suite, force=a.force)
    elif a.cmd == "inspect":
        pool = load_pool(a.pool if a.pool.name.startswith("tabarena-")
                         else a.pool / FROZEN_SUITE)
        print("\nmethods:")
        for m, n in pool.method.value_counts().items():
            print(f"  {m:46s} {n:5d} rows")
        print(f"\nunits per dataset: "
              f"{pool.groupby('dataset').fold.nunique().value_counts().to_dict()}")
        print(f"metrics: {pool.metric.value_counts().to_dict()}")
    elif a.cmd == "checksums":
        src = (a.tabarena / "packages/tabarena/src/tabarena/models/_artifacts"
               / "results_checksums.json")
        CHECKSUMS.parent.mkdir(parents=True, exist_ok=True)
        CHECKSUMS.write_text(src.read_text())
        print(f"copied {src} -> {CHECKSUMS} "
              f"({len(json.loads(CHECKSUMS.read_text()))} artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
