"""Fetch TabArena's published per-fold results -- the real Elo pool.

LOGIN NODE ONLY for ``emit`` and ``fetch``; ``build`` and everything downstream
run offline.

WHY NOT JUST ``pip install tabarena``
-------------------------------------
Same reason we vendored the task table instead of importing the package: the
benchmark extra does not install on the cluster. But the results themselves are
plain public HTTP -- a method's leaderboard rows live at a deterministic key:

    https://data.tabarena.ai/cache/artifacts/<suite>/methods/<method>/results/<file>.parquet

with ``<file>`` determined by ``method_type``: ``hpo_results`` for a ``config``
method, ``model_results`` for a ``baseline``, ``portfolio_results`` for a
``portfolio``. So the only thing we need the package for is *enumerating* the
methods, and that step needs no network. We therefore split the job:

    emit   needs ``import tabarena``, no network. Writes pool_manifest.json.
    fetch  needs network, no tabarena. Downloads the parquets.
    build  needs neither. Concatenates to one tidy frame.

If ``import tabarena`` fails everywhere available, ``emit --from-source`` reads
the method list out of the cloned repo's own metadata modules by AST instead of
importing them, which has no third-party dependencies at all.

SCHEMA. ``build`` emits exactly what ``elo.py`` and ``bencheval`` expect:
``method``, ``task``, ``split``, ``metric_error`` (lower is better),
``problem_type``. TabArena names the fold column ``fold`` and the dataset column
``dataset``; both are renamed here so one vocabulary survives downstream.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

BASE_URL = "https://data.tabarena.ai/"
#: Methods built through ``MethodMetadata.tabarena_legacy_s3`` live in the older
#: public **S3** bucket, not the R2 custom domain. Different host, same key
#: layout. Most of the tuned baselines that anchor the leaderboard -- Random
#: Forest included -- are in here, so getting this wrong loses the anchor.
LEGACY_S3_BASE_URL = "https://tabarena.s3.amazonaws.com/"
LEGACY_S3_FACTORY = "tabarena_legacy_s3"
CACHE_PREFIX = "cache"
RESULTS_FILE = {
    "config": "hpo_results.parquet",
    "baseline": "model_results.parquet",
    "portfolio": "portfolio_results.parquet",
}


# ------------------------------------------------------------------ emit
def _emit_via_import(context: str) -> list[dict]:
    """Enumerate methods through tabarena's own context object."""
    from tabarena.contexts.tabarena.context import TabArenaContext  # noqa: PLC0415

    ctx = TabArenaContext() if context == "default" else TabArenaContext(context)
    out = []
    for method in ctx.methods:
        md = ctx.method_metadata(method=method)
        fname = RESULTS_FILE.get(md.method_type)
        if fname is None:
            print(f"  [skip] {method}: unknown method_type {md.method_type!r}", file=sys.stderr)
            continue
        ck = getattr(md, "cache_kwargs", None) or {}
        prefix = ck.get("prefix", CACHE_PREFIX)
        key = f"{prefix}/artifacts/{md.suite}/methods/{md.method}/results/{fname}"
        if getattr(md, "cache_type", None) == "s3":
            base = f"https://{ck.get('bucket', 'tabarena')}.s3.amazonaws.com/"
        else:
            base = ck.get("base_url", BASE_URL)
        out.append(dict(method=md.method, suite=md.suite, method_type=md.method_type,
                        method_class=getattr(md, "method_class", None),
                        key=key, url=base + key))
    return out


def _emit_from_source(repo: Path) -> list[dict]:
    """Enumerate methods by parsing the clone -- no imports, no dependencies.

    Walks every ``MethodMetadata(...)`` / ``MethodMetadata.<factory>(...)`` call
    in the tabarena package and reads the literal ``method`` / ``suite`` /
    ``method_type`` keywords. Calls whose values are not literals (built in a
    loop, or inheriting ``suite`` from a ``**_common_kwargs``) cannot be
    resolved this way and are reported, not guessed -- a silently short pool
    would quietly change every Elo number.
    """
    src_root = repo / "packages" / "tabarena" / "src" / "tabarena"
    if not src_root.is_dir():
        raise FileNotFoundError(f"{src_root} not found -- is --repo pointing at the tabarena clone?")

    found, unresolved = {}, []
    for py in sorted(src_root.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        # module-level literal bindings: `_common_kwargs = dict(suite=...)` and
        # `methods = ["RandomForest", ...]`. The 2025-06-12 catalogue builds the
        # original tuned baselines in a `for method in methods:` loop, so
        # without the list bindings the pool silently loses them.
        common: dict[str, object] = {}
        lists: dict[str, list[str]] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Name):
                continue
            val = node.value
            if isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id == "dict":
                for kw in val.keywords:
                    if kw.arg and isinstance(kw.value, ast.Constant):
                        common[kw.arg] = kw.value.value
            elif isinstance(val, (ast.List, ast.Tuple)):
                items = [e.value for e in val.elts if isinstance(e, ast.Constant)
                         and isinstance(e.value, str)]
                if items:
                    lists[tgt.id] = items

        # loop variable -> the literal list it iterates, for `for m in methods:`
        loop_var: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                    and isinstance(node.iter, ast.Name) and node.iter.id in lists):
                loop_var[node.target.id] = lists[node.iter.id]

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (fn.id if isinstance(fn, ast.Name) else
                    (fn.attr if isinstance(fn, ast.Attribute) else None))
            base_obj = (fn.value.id if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                        else None)
            if not (name == "MethodMetadata" or base_obj == "MethodMetadata"):
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            vals = {k: v.value for k, v in kw.items() if isinstance(v, ast.Constant)}
            suite = vals.get("suite", common.get("suite"))
            mtype = vals.get("method_type", "config")
            fname = RESULTS_FILE.get(mtype)

            # `method=` is either a literal or the loop variable; the latter
            # expands to every name in the list it iterates.
            mnode = kw.get("method")
            if isinstance(mnode, ast.Constant):
                methods = [mnode.value]
            elif isinstance(mnode, ast.Name) and mnode.id in loop_var:
                methods = loop_var[mnode.id]
            else:
                methods = []

            if not methods or not suite or fname is None:
                if fname is not None:
                    unresolved.append(f"{py.relative_to(repo)}:{node.lineno}")
                continue

            legacy = (base_obj == "MethodMetadata" and name == LEGACY_S3_FACTORY)
            bucket = vals.get("bucket", "tabarena")
            prefix = vals.get("prefix", CACHE_PREFIX)
            url_base = f"https://{bucket}.s3.amazonaws.com/" if legacy else BASE_URL
            for m in methods:
                key = f"{prefix}/artifacts/{suite}/methods/{m}/results/{fname}"
                found[(suite, m)] = dict(method=m, suite=suite, method_type=mtype,
                                         method_class=None, key=key, url=url_base + key)

    if unresolved:
        print(f"  [warn] {len(unresolved)} MethodMetadata call(s) had non-literal "
              f"method/suite and were skipped; the pool may be short. First few:",
              file=sys.stderr)
        for u in unresolved[:5]:
            print(f"         {u}", file=sys.stderr)
        print("  [warn] prefer `emit` without --from-source where tabarena imports.",
              file=sys.stderr)
    return sorted(found.values(), key=lambda d: (d["suite"], d["method"]))


def cmd_emit(a) -> None:
    entries = (_emit_from_source(Path(a.repo)) if a.from_source
               else _emit_via_import(a.context))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"base_url": BASE_URL, "methods": entries}, indent=2))
    print(f"{len(entries)} methods -> {out}")
    by_type: dict[str, int] = {}
    for e in entries:
        by_type[e["method_type"]] = by_type.get(e["method_type"], 0) + 1
    print(f"  by method_type: {by_type}")


# ------------------------------------------------------------------ fetch
def cmd_fetch(a) -> None:
    import requests  # noqa: PLC0415

    man = json.loads(Path(a.manifest).read_text())
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    ok = skipped = missing = 0
    failures = []

    for i, e in enumerate(man["methods"], 1):
        target = dest / f"{e['suite']}__{e['method']}.parquet"
        if target.exists() and not a.force:
            skipped += 1
            continue
        try:
            r = sess.get(e["url"], timeout=a.timeout, stream=True)
            if r.status_code == 404:
                missing += 1
                print(f"  [404] {e['method']}", flush=True)
                continue
            r.raise_for_status()
            tmp = target.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.rename(target)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - one bad method must not kill the run
            failures.append((e["method"], repr(exc)))
            print(f"  [fail] {e['method']}: {exc}", flush=True)
        if i % 10 == 0:
            print(f"  {i}/{len(man['methods'])}", flush=True)

    print(f"\ndownloaded {ok}, cached {skipped}, missing(404) {missing}, failed {len(failures)}")
    if failures:
        print("FAILURES -- rerun to retry (downloads are resumable per file):")
        for m, exc in failures[:10]:
            print(f"  {m}: {exc}")


# ------------------------------------------------------------------ build
#: TabArena's column names -> ours. ``metric_error`` is already its name.
RENAME = {"dataset": "task", "fold": "split", "framework": "method"}


def cmd_build(a) -> None:
    src = Path(a.dest)
    files = sorted(src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {src} -- run `fetch` first")

    frames, dropped = [], []
    for f in files:
        d = pd.read_parquet(f)
        d = d.rename(columns={k: v for k, v in RENAME.items() if k in d.columns})
        need = {"method", "task", "split", "metric_error"}
        if not need.issubset(d.columns):
            dropped.append((f.name, sorted(need - set(d.columns))))
            continue
        keep = ["method", "task", "split", "metric_error"]
        if "problem_type" in d.columns:
            keep.append("problem_type")
        frames.append(d[keep])

    if dropped:
        print(f"[warn] {len(dropped)} file(s) lacked the expected columns and were dropped:")
        for n, miss in dropped[:10]:
            print(f"  {n}: missing {miss}")

    pool = pd.concat(frames, ignore_index=True)
    pool = pool.dropna(subset=["metric_error"])
    before = len(pool)
    pool = pool.drop_duplicates(subset=["method", "task", "split"])
    if len(pool) != before:
        print(f"[warn] dropped {before - len(pool):,} duplicate (method, task, split) rows")

    if "problem_type" not in pool.columns:
        print("[warn] no problem_type column; binary and multiclass cannot be separated, "
              "and ROC-AUC must never be pooled with log-loss. Supply one before scoring.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out, index=False)
    print(f"\npool: {pool.method.nunique()} methods, {pool.task.nunique()} tasks, "
          f"{len(pool):,} rows -> {out}")
    if "problem_type" in pool.columns:
        print(pool.groupby("problem_type").agg(methods=("method", "nunique"),
                                               tasks=("task", "nunique"),
                                               rows=("task", "size")).to_string())


# ------------------------------------------------------------------ cli
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit", help="enumerate methods -> pool_manifest.json (no network)")
    e.add_argument("--out", default="results/pool_manifest.json")
    e.add_argument("--context", default="default")
    e.add_argument("--from-source", action="store_true",
                   help="parse the clone by AST instead of importing tabarena")
    e.add_argument("--repo", default="tabarena", help="path to the tabarena clone")
    e.set_defaults(func=cmd_emit)

    f = sub.add_parser("fetch", help="download the parquets (LOGIN NODE -- needs internet)")
    f.add_argument("--manifest", default="results/pool_manifest.json")
    f.add_argument("--dest", default="tabarena_pool")
    f.add_argument("--timeout", type=float, default=120.0)
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("build", help="concatenate to one tidy pool parquet (offline)")
    b.add_argument("--dest", default="tabarena_pool")
    b.add_argument("--out", default="results/tabarena_pool.parquet")
    b.set_defaults(func=cmd_build)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
