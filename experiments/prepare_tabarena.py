"""LOGIN NODE ONLY. Materialise TabArena's official splits into a portable manifest.

Why a manifest instead of importing tabarena in the worker:

  * `tabarena` depends on a PRE-RELEASE `autogluon.tabular` (>=1.6.3b20260917) plus
    ray. That is a heavy, fragile stack to stand up inside a Compute Canada venv,
    and compute nodes have no internet to repair it.
  * The splits are just index arrays. Resolving them once on a login node and
    writing them to disk makes the worker depend on nothing but numpy + openml
    cache, and makes the exact split set a versioned, auditable artifact.

Writes to $PROJECT_ROOT/tabarena/:
    manifest.json            per-dataset: tid, problem_type, n_classes, n_samples,
                             n_features, split list, size bucket
    splits/<dataset>.npz     train_<k> / test_<k> index arrays
and warms the OpenML cache so compute nodes never reach the network.

Usage (login node, after `pip install tabarena`):
    python -m experiments.prepare_tabarena --out $PROJECT_ROOT/tabarena
    python -m experiments.prepare_tabarena --out ... --lite     # 1 split/dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRESET = "TabArena-v0.1"

#: Size buckets drive SLURM time limits; TabArena spans ~700 to ~150k rows and a
#: single wall-clock limit either wastes the queue or kills the big datasets.
def size_bucket(n_samples: int) -> str:
    if n_samples <= 5_000:
        return "small"
    if n_samples <= 50_000:
        return "medium"
    return "large"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lite", action="store_true",
                    help="one split per dataset (smoke test, not for the paper)")
    ap.add_argument("--classification-only", action="store_true", default=True,
                    help="skip regression tasks (TabICLClassifier only)")
    ap.add_argument("--include-regression", dest="classification_only",
                    action="store_false")
    ap.add_argument("--skip-data", action="store_true",
                    help="manifest only; do not warm the OpenML cache")
    a = ap.parse_args()

    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
    from tabarena.benchmark.task.metadata.schema import tid_from_task_id_str
    from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper

    coll = TaskMetadataCollection.from_preset(PRESET)
    if a.lite:
        coll = coll.subset_tasks(subset="lite")
    by_ds = coll.task_metadata_by_dataset()
    print(f"{PRESET}: {len(by_ds)} datasets")

    out = a.out
    (out / "splits").mkdir(parents=True, exist_ok=True)

    manifest, skipped = {}, []
    for i, (name, md) in enumerate(sorted(by_ds.items()), 1):
        if a.classification_only and md.problem_type == "regression":
            skipped.append((name, "regression"))
            continue

        tid = tid_from_task_id_str(md.task_id_str)
        try:
            task = OpenMLTaskWrapper.from_task_id(tid, metadata=md)
        except Exception as e:  # noqa: BLE001
            skipped.append((name, f"load failed: {type(e).__name__}: {e}"))
            print(f"  [{i:2d}/{len(by_ds)}] {name:42s} SKIP ({e})")
            continue

        n_repeats, n_folds, n_samples_dim = task.get_split_dimensions()
        splits, arrays = [], {}
        k = 0
        for rep in range(n_repeats):
            for fold in range(n_folds):
                for smp in range(max(1, n_samples_dim)):
                    tr, te = task.get_split_indices(fold=fold, repeat=rep, sample=smp)
                    arrays[f"train_{k}"] = np.asarray(tr, dtype=np.int64)
                    arrays[f"test_{k}"] = np.asarray(te, dtype=np.int64)
                    splits.append(dict(k=k, repeat=rep, fold=fold, sample=smp,
                                       n_train=len(tr), n_test=len(te)))
                    k += 1
        np.savez_compressed(out / "splits" / f"{name}.npz", **arrays)

        X, y = task.X, task.y
        n_rows, n_feat = int(X.shape[0]), int(X.shape[1])
        manifest[name] = dict(
            dataset=name, tid=int(tid), openml_dataset_id=int(task.dataset_id),
            problem_type=md.problem_type, label=task.label,
            n_classes=int(getattr(md, "num_classes", 0) or 0),
            n_samples=n_rows, n_features=n_feat,
            n_splits=len(splits), splits=splits,
            split_dims=dict(repeats=n_repeats, folds=n_folds, samples=n_samples_dim),
            bucket=size_bucket(n_rows),
            eval_metric=task.eval_metric,
        )
        print(f"  [{i:2d}/{len(by_ds)}] {name:42s} tid={tid:<8d} "
              f"{md.problem_type:10s} n={n_rows:>7d} d={n_feat:>5d} splits={len(splits)}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    buckets = {}
    for m in manifest.values():
        buckets[m["bucket"]] = buckets.get(m["bucket"], 0) + 1
    print(f"\nwrote {out/'manifest.json'}: {len(manifest)} datasets")
    print(f"  buckets: {buckets}")
    print(f"  total cells (datasets x splits): "
          f"{sum(m['n_splits'] for m in manifest.values())}")
    if skipped:
        print(f"  skipped {len(skipped)}:")
        for n, why in skipped:
            print(f"    {n}: {why}")

    # --- self-check: the worker must be able to read this back without tabarena
    print("\nverifying manifest is readable without tabarena...")
    import importlib
    m = json.loads((out / "manifest.json").read_text())
    ds0 = next(iter(m))
    z = np.load(out / "splits" / f"{ds0}.npz")
    tr, te = z["train_0"], z["test_0"]
    assert len(set(tr) & set(te)) == 0, "train/test overlap"
    assert len(tr) > 0 and len(te) > 0
    print(f"  ok: {ds0} split 0 -> train {len(tr)}, test {len(te)}, no overlap")
    importlib.invalidate_caches()
    print("\nNext: re-run with --skip-data omitted so the OpenML cache is warm, "
          "then submit slurm/03_full_array.sh")


if __name__ == "__main__":
    main()
