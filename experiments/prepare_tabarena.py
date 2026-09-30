"""LOGIN NODE ONLY. Materialise TabArena v0.1's official splits — without tabarena.

The `tabarena` package is NOT imported here or anywhere else in this project: it
pulls a pre-release `autogluon.tabular` plus ray, which does not install cleanly
on the cluster. It is not needed, because TabArena's splits ARE OpenML task
splits. Their wrapper is a pure passthrough:

    # tabarena/benchmark/task/openml/task_wrapper.py
    def get_split_indices(self, fold=0, repeat=0, sample=0):
        return self.task.get_train_test_split_indices(fold=fold, repeat=repeat, sample=sample)

So we read their vendored task table (`data/TabArena-v0.1_tasks_metadata.csv`,
Apache-2.0, commit 52dac56) for the task ids and the official (repeat, fold)
list, and call `openml` directly for the indices.

Fidelity is not assumed, it is CHECKED: the CSV records `num_instances_train`
and `num_instances_test` for every one of the 816 units, and this script asserts
the split it computes matches on both counts. A mismatch aborts.

Writes $TABARENA_DIR/:
    manifest.json          per dataset: tid, problem_type, n_classes, splits, bucket
    splits/<ds>.npz        train_<k> / test_<k> index arrays
and warms the OpenML cache so compute nodes never touch the network.

    python -m experiments.prepare_tabarena --out $TABARENA_DIR
    python -m experiments.prepare_tabarena --out $TABARENA_DIR --include-regression
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

CSV_PATH = Path(__file__).parent / "data" / "TabArena-v0.1_tasks_metadata.csv"


def size_bucket(n: int) -> str:
    """Drives per-bucket SLURM time limits; TabArena spans 748 to 150k rows."""
    return "small" if n <= 5_000 else ("medium" if n <= 50_000 else "large")


def parse_tid(task_id_str: str) -> int:
    """TabArena's tid_from_task_id_str: '{Prefix}|{tid}|...' or a plain int."""
    parts = str(task_id_str).split("|")
    return int(parts[1]) if len(parts) > 1 else int(parts[0])


def load_task_table() -> "OrderedDict[str, list[dict]]":
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"vendored task table missing: {CSV_PATH}")
    by_ds: OrderedDict[str, list[dict]] = OrderedDict()
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            by_ds.setdefault(r["tabarena_task_name"], []).append(r)
    return by_ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--classification-only", action="store_true", default=True)
    ap.add_argument("--include-regression", dest="classification_only",
                    action="store_false")
    ap.add_argument("--only", nargs="*", default=None, help="subset of dataset names")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the train/test size assertions (not recommended)")
    a = ap.parse_args()

    import openml

    by_ds = load_task_table()
    print(f"TabArena v0.1 task table: {len(by_ds)} datasets, "
          f"{sum(len(v) for v in by_ds.values())} official splits")

    out: Path = a.out
    (out / "splits").mkdir(parents=True, exist_ok=True)

    manifest, skipped, mismatches = {}, [], []
    for i, (name, rows) in enumerate(by_ds.items(), 1):
        head = rows[0]
        if a.only and name not in a.only:
            continue
        if a.classification_only and head["is_classification"] != "True":
            skipped.append((name, "regression"))
            continue

        tid = parse_tid(head["task_id_str"])
        try:
            task = openml.tasks.get_task(
                tid, download_splits=True, download_data=True,
                download_qualities=False, download_features_meta_data=False,
            )
        except Exception as e:  # noqa: BLE001
            skipped.append((name, f"{type(e).__name__}: {e}"))
            print(f"  [{i:2d}/{len(by_ds)}] {name:42s} SKIP ({e})")
            continue

        arrays, splits, bad = {}, [], 0
        for k, r in enumerate(rows):
            rep, fold = int(r["repeat"]), int(r["fold"])
            tr, te = task.get_train_test_split_indices(fold=fold, repeat=rep, sample=0)
            tr = np.asarray(tr, dtype=np.int64)
            te = np.asarray(te, dtype=np.int64)

            if not a.no_verify:
                exp_tr = int(float(r["num_instances_train"]))
                exp_te = int(float(r["num_instances_test"]))
                if len(tr) != exp_tr or len(te) != exp_te:
                    bad += 1
                    mismatches.append(
                        f"{name} r{rep}f{fold}: got {len(tr)}/{len(te)}, "
                        f"TabArena records {exp_tr}/{exp_te}")
                assert len(set(tr.tolist()) & set(te.tolist())) == 0, \
                    f"{name} r{rep}f{fold}: train/test overlap"

            arrays[f"train_{k}"] = tr
            arrays[f"test_{k}"] = te
            splits.append(dict(k=k, repeat=rep, fold=fold,
                               split_index=r.get("split_index", f"r{rep}f{fold}"),
                               n_train=int(len(tr)), n_test=int(len(te))))

        np.savez_compressed(out / "splits" / f"{name}.npz", **arrays)

        n_inst = int(float(head["num_instances"]))
        n_cls_raw = int(float(head["num_classes"]))
        manifest[name] = dict(
            dataset=name,
            tid=tid,
            openml_dataset_id=int(task.dataset_id),
            problem_type=head["problem_type"],
            label=head["target_name"],
            eval_metric=head["eval_metric"],
            n_classes=max(0, n_cls_raw),        # -1 in the CSV for regression
            n_samples=n_inst,
            n_features=int(float(head["num_features"])),
            n_splits=len(splits),
            splits=splits,
            bucket=size_bucket(n_inst),
        )
        flag = "" if bad == 0 else f"  !! {bad} SIZE MISMATCH"
        print(f"  [{i:2d}/{len(by_ds)}] {name:42s} tid={tid:<8d} "
              f"{head['problem_type']:10s} n={n_inst:>7d} d={manifest[name]['n_features']:>5d} "
              f"splits={len(splits):>2d}{flag}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    buckets: dict[str, int] = {}
    for m in manifest.values():
        buckets[m["bucket"]] = buckets.get(m["bucket"], 0) + 1
    total_cells = sum(m["n_splits"] for m in manifest.values())
    print(f"\nwrote {out/'manifest.json'}")
    print(f"  datasets       : {len(manifest)}")
    print(f"  dataset-splits : {total_cells}")
    print(f"  buckets        : {buckets}")
    if skipped:
        print(f"  skipped        : {len(skipped)}")
        for n, why in skipped[:5]:
            print(f"      {n}: {why}")
        if len(skipped) > 5:
            print(f"      ... and {len(skipped)-5} more")

    if mismatches:
        print(f"\n!! {len(mismatches)} SPLIT SIZE MISMATCHES vs TabArena's record:")
        for m in mismatches[:10]:
            print("   ", m)
        raise SystemExit(
            "\nSplits do NOT match TabArena. Results would not be comparable to the "
            "published leaderboard. Investigate before running the campaign."
        )

    print("\n[OK] every split matches TabArena's recorded train/test sizes")

    # the worker must read this back with numpy alone
    ds0 = next(iter(manifest))
    z = np.load(out / "splits" / f"{ds0}.npz")
    assert len(set(z["train_0"]) & set(z["test_0"])) == 0
    print(f"[OK] manifest readable without tabarena "
          f"({ds0}: train {len(z['train_0'])}, test {len(z['test_0'])})")


if __name__ == "__main__":
    main()
