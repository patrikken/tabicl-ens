"""Enumerate the campaign: which (dataset, coalition) pairs to run, per model/task/bucket.

One array task = one (dataset, coalition) pair, looping its splits internally.
This module is the single source of truth for that list; the SLURM scripts only
index into a plan file it writes, so the mapping from array index to pair can
never drift between submit time and run time (it would, if each task re-derived
the list while cells were completing underneath it).

    python -m experiments.grid plan   --models tabicl tabfm --tasks classification regression
    python -m experiments.grid list   --model tabfm --task regression --bucket small \\
                                      --skip-complete --cache $CACHE_DIR/tabfm > plan.txt

Completeness is judged from disk: a pair is complete iff every split of the
dataset has a meta.json. A pair with SOME splits done is kept (the worker skips
finished splits), so resubmitting after a time-out resumes instead of restarting.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from experiments.datasets import dataset_info, tabarena_datasets

MODELS = ("tabicl", "tabfm")
TASKS = ("classification", "regression")
BUCKETS = ("small", "medium", "large")


def coalitions_for(model: str, task: str, n_features: int, bucket: str,
                   set_name: str = "all") -> list[str]:
    if model == "tabicl":
        from experiments.coalitions import tabicl_coalition_set          # noqa: PLC0415
        return tabicl_coalition_set(set_name, task, n_features, bucket)
    if model == "tabfm":
        from experiments.tabfm_sets import coalition_set                 # noqa: PLC0415
        return coalition_set(set_name, n_features, task)
    raise ValueError(f"unknown model {model!r}")


def cache_root(model: str, cache: Path) -> Path:
    """TabICLv2 caches live at the cache root, TabFM under ``tabfm/``."""
    return cache / "tabfm" if model == "tabfm" and cache.name != "tabfm" else cache


def n_done(root: Path, dataset: str, coalition: str) -> int:
    d = root / dataset / coalition
    return sum(1 for _ in d.glob("split*/meta.json")) if d.exists() else 0


def pairs(model: str, task: str, bucket: str, set_name: str = "all",
          skip_complete: bool = False, cache: Path | None = None
          ) -> list[tuple[str, str, int, int]]:
    """``(dataset, coalition, n_splits, n_done)`` in a stable order."""
    out = []
    root = cache_root(model, cache) if cache is not None else None
    for ds in tabarena_datasets(task):
        info = dataset_info(ds)
        if bucket != "all" and info["bucket"] != bucket:
            continue
        for c in coalitions_for(model, task, info["n_features"], info["bucket"], set_name):
            done = n_done(root, ds, c) if root is not None else 0
            if skip_complete and done >= info["n_splits"]:
                continue
            out.append((ds, c, info["n_splits"], done))
    return out


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "plan"):
        s = sub.add_parser(name)
        s.add_argument("--set", default="all", dest="set_name")
        s.add_argument("--skip-complete", action="store_true")
        s.add_argument("--cache", type=Path,
                       default=Path(os.environ["CACHE_DIR"]) if os.environ.get("CACHE_DIR") else None)
    l = sub.choices["list"]
    l.add_argument("--model", required=True, choices=MODELS)
    l.add_argument("--task", required=True, choices=TASKS)
    l.add_argument("--bucket", default="all")
    pl = sub.choices["plan"]
    pl.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    pl.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = _parse(argv)
    if a.cmd == "list":
        for ds, c, _n, _d in pairs(a.model, a.task, a.bucket, a.set_name,
                                   a.skip_complete, a.cache):
            print(ds, c)
        return 0

    # plan: a table of what would run
    print(f"{'model':7s} {'task':15s} {'bucket':7s} {'datasets':>8s} {'tasks':>6s} "
          f"{'cells todo':>10s} {'cells done':>10s}")
    tot_t = tot_c = 0
    for m in a.models:
        for t in a.tasks:
            try:
                names = tabarena_datasets(t)
            except Exception as e:                           # noqa: BLE001
                print(f"{m:7s} {t:15s} (manifest unreadable: {e})")
                continue
            for b in BUCKETS:
                ps = pairs(m, t, b, a.set_name, a.skip_complete, a.cache)
                nd = len({p[0] for p in ps})
                todo = sum(n - d for _, _, n, d in ps)
                done = sum(d for _, _, _, d in ps)
                tot_t += len(ps); tot_c += todo
                print(f"{m:7s} {t:15s} {b:7s} {nd:8d} {len(ps):6d} {todo:10d} {done:10d}")
            if not names:
                print(f"{m:7s} {t:15s} {'-':7s} 0 datasets in manifest -- run prepare_tabarena"
                      + (" --include-regression" if t == "regression" else ""))
    print(f"{'TOTAL':23s} {'':8s} {tot_t:6d} {tot_c:10d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
