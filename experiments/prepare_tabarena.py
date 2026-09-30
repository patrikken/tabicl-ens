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

FIDELITY IS AUDITED, NOT ENFORCED. The CSV records `num_instances_train` /
`num_instances_test` for all 816 units. Every split is checked against those
numbers, but a mismatch is a WARNING, not a failure: the run proceeds so the
full evaluation can establish trends. Every check lands in
`$TABARENA_DIR/split_audit.csv`, and each split carries a `verified` flag in the
manifest so the analysis can filter or stratify later.

Severity, recorded per split:
    ok        sizes match TabArena's record exactly
    resized   sizes differ -> internally valid, but NOT comparable to the
              published leaderboard. Exclude before making a leaderboard claim.
    overlap   train and test indices intersect -> leakage, the split's numbers
              are not usable for any claim. Rare; investigate if it appears.

Writes $TABARENA_DIR/:
    manifest.json          per dataset: tid, problem_type, splits, bucket, verified
    splits/<ds>.npz        train_<k> / test_<k> index arrays
    split_audit.csv        one row per split, full audit trail
and warms the OpenML cache so compute nodes never touch the network.

    python -m experiments.prepare_tabarena --out $TABARENA_DIR
    python -m experiments.prepare_tabarena --out $TABARENA_DIR --include-regression
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

CSV_PATH = Path(__file__).parent / "data" / "TabArena-v0.1_tasks_metadata.csv"

AUDIT_FIELDS = [
    "dataset", "k", "repeat", "fold", "split_index",
    "n_train", "n_train_expected", "n_test", "n_test_expected",
    "size_match", "overlap", "severity",
]


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
    ap.add_argument("--strict", action="store_true",
                    help="abort on any mismatch instead of warning (off by default: "
                         "the full eval runs first, trends before leaderboard claims)")
    a = ap.parse_args()

    import openml

    by_ds = load_task_table()
    print(f"TabArena v0.1 task table: {len(by_ds)} datasets, "
          f"{sum(len(v) for v in by_ds.values())} official splits")

    out: Path = a.out
    (out / "splits").mkdir(parents=True, exist_ok=True)

    manifest: dict = {}
    skipped: list[tuple[str, str]] = []
    audit: list[dict] = []

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

        arrays: dict[str, np.ndarray] = {}
        splits: list[dict] = []
        sev_count: Counter = Counter()

        for k, r in enumerate(rows):
            rep, fold = int(r["repeat"]), int(r["fold"])
            tr, te = task.get_train_test_split_indices(fold=fold, repeat=rep, sample=0)
            tr = np.asarray(tr, dtype=np.int64)
            te = np.asarray(te, dtype=np.int64)

            exp_tr = int(float(r["num_instances_train"]))
            exp_te = int(float(r["num_instances_test"]))
            size_match = (len(tr) == exp_tr) and (len(te) == exp_te)
            overlap = int(len(np.intersect1d(tr, te, assume_unique=False)))

            severity = "overlap" if overlap else ("ok" if size_match else "resized")
            sev_count[severity] += 1

            audit.append(dict(
                dataset=name, k=k, repeat=rep, fold=fold,
                split_index=r.get("split_index", f"r{rep}f{fold}"),
                n_train=len(tr), n_train_expected=exp_tr,
                n_test=len(te), n_test_expected=exp_te,
                size_match=int(size_match), overlap=overlap, severity=severity,
            ))

            arrays[f"train_{k}"] = tr
            arrays[f"test_{k}"] = te
            splits.append(dict(
                k=k, repeat=rep, fold=fold,
                split_index=r.get("split_index", f"r{rep}f{fold}"),
                n_train=int(len(tr)), n_test=int(len(te)),
                verified=bool(size_match), severity=severity,
            ))

        np.savez_compressed(out / "splits" / f"{name}.npz", **arrays)

        n_inst = int(float(head["num_instances"]))
        manifest[name] = dict(
            dataset=name,
            tid=tid,
            openml_dataset_id=int(task.dataset_id),
            problem_type=head["problem_type"],
            label=head["target_name"],
            eval_metric=head["eval_metric"],
            n_classes=max(0, int(float(head["num_classes"]))),  # -1 for regression
            n_samples=n_inst,
            n_features=int(float(head["num_features"])),
            n_splits=len(splits),
            splits=splits,
            bucket=size_bucket(n_inst),
            all_splits_verified=all(s["verified"] for s in splits),
            severity_counts=dict(sev_count),
        )

        note = ""
        if sev_count.get("resized"):
            note += f"  [WARN {sev_count['resized']} resized]"
        if sev_count.get("overlap"):
            note += f"  [!! {sev_count['overlap']} OVERLAP]"
        print(f"  [{i:2d}/{len(by_ds)}] {name:42s} tid={tid:<8d} "
              f"{head['problem_type']:10s} n={n_inst:>7d} "
              f"d={manifest[name]['n_features']:>5d} splits={len(splits):>2d}{note}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with (out / "split_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        w.writerows(audit)

    # ---- summary ----------------------------------------------------------
    buckets: Counter = Counter(m["bucket"] for m in manifest.values())
    sev: Counter = Counter(r["severity"] for r in audit)
    total_cells = sum(m["n_splits"] for m in manifest.values())

    print(f"\nwrote {out/'manifest.json'}")
    print(f"  datasets       : {len(manifest)}")
    print(f"  dataset-splits : {total_cells}")
    print(f"  buckets        : {dict(buckets)}")
    if skipped:
        print(f"  skipped        : {len(skipped)}")
        for n, why in skipped[:5]:
            print(f"      {n}: {why}")
        if len(skipped) > 5:
            print(f"      ... and {len(skipped)-5} more")

    print(f"\nsplit audit -> {out/'split_audit.csv'}")
    print(f"  ok      : {sev.get('ok', 0):>4d}  matches TabArena exactly")
    print(f"  resized : {sev.get('resized', 0):>4d}  differs -> valid, but not "
          f"leaderboard-comparable")
    print(f"  overlap : {sev.get('overlap', 0):>4d}  train/test intersect -> unusable")

    bad_ds = [n for n, m in manifest.items() if not m["all_splits_verified"]]
    if bad_ds:
        print(f"\n  {len(bad_ds)} dataset(s) with at least one non-matching split:")
        for n in bad_ds[:10]:
            print(f"      {n}: {manifest[n]['severity_counts']}")
        if len(bad_ds) > 10:
            print(f"      ... and {len(bad_ds)-10} more")
        print("\n  Proceeding anyway (trends first). Before any claim against the")
        print("  published TabArena leaderboard, filter to severity == 'ok'.")
        if sev.get("overlap"):
            print("\n  !! OVERLAP rows leak test data into training. Drop those splits")
            print("     from every analysis, not just leaderboard comparisons.")
        if a.strict:
            raise SystemExit("--strict: aborting on split mismatches.")
    else:
        print("\n[OK] every split matches TabArena's recorded train/test sizes")

    ds0 = next(iter(manifest))
    z = np.load(out / "splits" / f"{ds0}.npz")
    assert len(set(z["train_0"]) & set(z["test_0"])) == 0 or sev.get("overlap")
    print(f"[OK] manifest readable without tabarena "
          f"({ds0}: train {len(z['train_0'])}, test {len(z['test_0'])})")


if __name__ == "__main__":
    main()
