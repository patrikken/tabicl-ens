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
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://data.tabarena.ai/cache/"
#: data.tabarena.ai sits behind Cloudflare, which 403s stock `Python-urllib/3.x`
#: before the request ever reaches the bucket. Upstream never hits this because
#: `MethodDownloaderPublicR2` uses `requests` (a UA Cloudflare allows). We send a
#: browser UA so the transport is not what decides whether a public file is
#: public. A 403 that survives this is about the key or the bucket, not the
#: client -- `arena.py probe` tells the two apart.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
FROZEN_SUITE = "tabarena-2025-06-12"  # the NeurIPS-2025 / v0.1 leaderboard
#: shipped with the repo; `arena.py checksums` refreshes it from a tabarena clone
CHECKSUMS = Path(__file__).with_name("data") / "results_checksums.json"

#: columns we need; the published frames carry more and the set has drifted
#: between suites, so select rather than assume.
NEEDED = ["method", "dataset", "fold", "metric_error", "metric", "problem_type"]


# ------------------------------------------------------------------- fetch --
def artifact_keys(
    suite: str = FROZEN_SUITE, checksums: Path = CHECKSUMS
) -> dict[str, str]:
    """``{key: md5}`` for every published results artifact in ``suite``."""
    if not checksums.exists():
        raise FileNotFoundError(
            f"{checksums} missing. Copy it from a tabarena clone:\n"
            f"  cp <tabarena>/packages/tabarena/src/tabarena/models/_artifacts/"
            f"results_checksums.json {checksums}"
        )
    all_keys = json.loads(checksums.read_text())
    return {k: v for k, v in all_keys.items() if k.split("/")[1] == suite}


def _get(
    url: str, timeout: float, user_agent: str = USER_AGENT, retries: int = 3
) -> bytes:
    """GET with a real User-Agent, mirroring upstream's `requests` session.

    Prefers `requests` when installed (exactly what MethodDownloaderPublicR2
    uses) and falls back to urllib with the same headers.
    """
    headers = {"User-Agent": user_agent, "Accept": "*/*"}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            try:
                import requests  # noqa: PLC0415

                r = requests.get(url, headers=headers, timeout=timeout)
                r.raise_for_status()
                return r.content
            except ImportError:
                req = urllib.request.Request(url, headers=headers)  # noqa: S310
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                    return resp.read()
        except Exception as e:  # noqa: BLE001
            last = e
            code = getattr(
                getattr(e, "response", None), "status_code", getattr(e, "code", None)
            )
            if code in (403, 404):
                raise  # not transient
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def _describe_failure(url: str, timeout: float, user_agent: str) -> str:
    """Why did this 403? Cloudflare's edge and the bucket fail differently."""
    out = []
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": user_agent}
        )  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            out.append(f"HTTP {r.status}")
    except urllib.error.HTTPError as e:
        body = b""
        try:
            body = e.read()[:400]
        except Exception:  # noqa: BLE001
            pass
        hdr = {k.lower(): v for k, v in (e.headers or {}).items()}
        out.append(f"HTTP {e.code}")
        for k in ("server", "cf-ray", "cf-mitigated", "content-type"):
            if k in hdr:
                out.append(f"{k}: {hdr[k]}")
        if body:
            out.append("body[:400]: " + body.decode("utf-8", "replace").strip())
    except Exception as e:  # noqa: BLE001
        out.append(f"{type(e).__name__}: {e}")
    return "\n      ".join(out)


def probe(
    suite: str = FROZEN_SUITE, timeout: float = 60.0, user_agent: str = USER_AGENT
) -> int:
    """Diagnose a failing fetch in one command.

    Tries, in order: the domain root, a URL quoted verbatim from tabarena's own
    example (so a failure there is not our key construction), and the first
    results artifact for the suite. Prints enough of each response to tell
    Cloudflare bot-blocking (`server: cloudflare`, a `cf-ray`, an HTML body)
    from an S3-style AccessDenied (an XML body), which is what a WRONG KEY looks
    like on a bucket that denies ListBucket.
    """
    keys = artifact_keys(suite)
    first = BASE_URL + sorted(keys)[0] if keys else None
    targets = [
        ("domain root", "https://data.tabarena.ai/"),
        # verbatim from examples/plots/run_end_to_end_from_raw.py:34 -- this URL
        # is published by tabarena itself, so if it 403s too the problem is the
        # network or the client, never our key scheme.
        (
            "tabarena's own documented URL",
            "https://data.tabarena.ai/cache/artifacts/tabarena-2026-05-13/methods/TabPFN-3/raw.zip",
        ),
        (f"first {suite} artifact", first),
    ]
    print(f"User-Agent: {user_agent[:60]}...")
    try:
        import requests  # noqa: F401,PLC0415

        print("requests: installed (used by upstream's downloader)")
    except ImportError:
        print(
            "requests: NOT installed -- falling back to urllib. "
            "`pip install requests` is worth trying first."
        )
    import os

    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY"):
        if os.environ.get(v):
            print(f"{v}={os.environ[v]}   <- a proxy can 403 independently")
    print()
    bad = 0
    for label, url in targets:
        if url is None:
            continue
        print(f"  {label}\n    {url}")
        try:
            blob = _get(url, timeout, user_agent, retries=1)
            print(f"    OK  {len(blob)/1e6:.2f} MB")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"    FAIL {type(e).__name__}: {e}")
            print(f"      {_describe_failure(url, timeout, user_agent)}")
        print()
    if bad:
        print("""READING THE RESULT -- the three 403s are not the same failure.

  ProxyError / "Tunnel connection failed: 403"
      A proxy on THIS machine refused to open the tunnel; the request never
      reached Cloudflare. Check the *_PROXY vars printed above.

  HTTP 403 with an HTML body, server: cloudflare, a cf-ray header
      The edge refused this client. Either the User-Agent (stock
      `Python-urllib/3.x` is on every WAF blocklist -- this module now sends a
      browser UA, so if you still see it, it is the next line) or the source IP:
      Cloudflare commonly blocks datacentre ranges, which is what a cluster
      login node looks like. No client-side fix for that one.

  HTTP 403 with an XML <Error><Code>AccessDenied</Code> body
      That is the bucket, not the edge: the key is wrong, or the object is not
      public. Send me the body and I will fix the key construction.

  Only the suite artifact fails, the documented URL works
      That suite moved. Re-run `arena.py checksums --tabarena <clone>`.

IF THE EDGE IS BLOCKING THE CLUSTER, fetch from a machine it does not block --
your laptop reaches it in a browser -- and copy the tree up:

    python -m experiments.arena urls --out tabarena_urls.txt --script fetch_pool.sh
    bash fetch_pool.sh arena_pool          # run on the laptop
    rsync -av arena_pool/ <cluster>:$PROJECT_ROOT/arena_pool/

The layout is the one `load_pool` expects, and `fetch` re-verifies md5s, so a
later run on the cluster confirms the copy rather than re-downloading.""")
    return 1 if bad else 0


def write_urls(
    out: Path, suite: str = FROZEN_SUITE, script: Path | None = None
) -> Path:
    """Write a plain URL list -- and optionally a fetch script -- so the pool can
    be downloaded from a machine Cloudflare does not block and copied up.

    The script writes straight into the layout ``load_pool`` expects, so the
    result rsyncs to the cluster as-is and a later ``fetch`` run just re-verifies
    the md5s.
    """
    keys = sorted(artifact_keys(suite))
    out.write_text("\n".join(BASE_URL + k for k in keys) + "\n")
    print(f"wrote {len(keys)} URLs -> {out}")
    if script is not None:
        script.write_text(f"""#!/bin/bash
# Fetch the TabArena {suite} results pool into the layout load_pool expects.
#   bash {script.name} [outdir]        (default: arena_pool)
# Run this wherever https://data.tabarena.ai is reachable -- a laptop browser
# reaches it even when a cluster login node is refused -- then:
#   rsync -av <outdir>/ <cluster>:$PROJECT_ROOT/arena_pool/
set -euo pipefail
OUT="${{1:-arena_pool}}/{suite}"
mkdir -p "$OUT"
n=0; fail=0
while read -r u; do
  [ -z "$u" ] && continue
  m=$(echo "$u" | awk -F/ '{{print $(NF-2)}}')
  f=$(basename "$u")
  mkdir -p "$OUT/$m"
  if [ -s "$OUT/$m/$f" ]; then echo "  [have] $m/$f"; continue; fi
  if curl -fsSL --retry 3 "$u" -o "$OUT/$m/$f"; then
    echo "  [ok]   $m/$f  $(du -h "$OUT/$m/$f" | cut -f1)"; n=$((n+1))
  else
    echo "  [FAIL] $m/$f"; rm -f "$OUT/$m/$f"; fail=$((fail+1))
  fi
done < "{out.name}"
echo "$n downloaded, $fail failed -> $OUT"
""")
        script.chmod(0o755)
        print(f"wrote fetch script -> {script}  (run it where the domain resolves)")
    return out


def fetch_pool(
    out: Path,
    suite: str = FROZEN_SUITE,
    *,
    timeout: float = 120.0,
    force: bool = False,
    user_agent: str = USER_AGENT,
) -> Path:
    """Download every results parquet for ``suite``. Verifies md5. Resumable.

    Returns the suite directory. Total is well under 100 MB -- this is the
    results tier (per-(method, dataset, fold) metric values), not the processed
    tier (10 GB/method, carries predictions) or raw (~100 GB/method).
    """
    keys = artifact_keys(suite)
    if not keys:
        raise ValueError(
            f"no artifacts for suite {suite!r}; "
            f"known: {sorted({k.split('/')[1] for k in json.loads(CHECKSUMS.read_text())})}"
        )
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
            blob = _get(url, timeout, user_agent)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {method}/{artifact}: {type(e).__name__}: {e}")
            if failed == 0:  # diagnose the first failure only
                print(f"      {_describe_failure(url, timeout, user_agent)}")
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
        print(
            "\nRun `python -m experiments.arena probe` -- it separates "
            "Cloudflare refusing the client from a wrong key in one command. "
            "The download is resumable, so re-running only fetches what is missing."
        )
    return root


# -------------------------------------------------------------------- load --
def load_pool(
    root: Path,
    *,
    problem_types: tuple[str, ...] = ("binary", "multiclass"),
    verbose: bool = True,
) -> pd.DataFrame:
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
        print(
            f"[pool] {len(pool):,} rows  {pool.method.nunique()} methods  "
            f"{pool.dataset.nunique()} datasets  "
            f"{pool.groupby(['dataset','fold']).ngroups} units"
        )
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
def our_results(
    scores: pd.DataFrame,
    *,
    model: str,
    budget: int | None = None,
    coalitions: list[str] | None = None,
    name: str = "{model} {coalition} (M={M})",
) -> pd.DataFrame:
    """Our per-cell scores -> the leaderboard schema.

    ``scores`` is any of our ``*_scores.csv`` frames: one row per
    (dataset, coalition, split, budget). Our ``split`` IS TabArena's ``fold`` --
    ``prepare_tabarena.py`` enumerates that dataset's (repeat, fold) rows from
    the official task table in order, which is the same flattening the published
    results use (26 datasets x 9 splits + 12 x 30 = 594 classification units).

    ``budget=None`` keeps each coalition at its own maximum budget; pass an int
    to compare every coalition at a matched member count.
    """
    from elo import to_metric_error  # noqa: PLC0415

    df = scores.copy()
    if coalitions is not None:
        df = df[df.coalition.isin(coalitions)]
    if budget is None:
        df = df[df.budget == df.groupby("coalition").budget.transform("max")]
    else:
        df = df[df.budget == budget]
        if df.empty:
            raise ValueError(f"no cells at budget={budget}")

    out = pd.DataFrame(
        {
            "method": [
                name.format(model=model, coalition=c, M=int(m))
                for c, m in zip(df.coalition, df.budget)
            ],
            "dataset": df.dataset.to_numpy(),
            "fold": df.split.to_numpy(),
            "metric_error": to_metric_error(df.score, df.task),
            "metric": np.where(df.task.to_numpy() == "binary", "roc_auc", "log_loss"),
            "problem_type": df.task.to_numpy(),
        }
    )
    return out.drop_duplicates(subset=["method", "dataset", "fold"])


def effect_sizes(
    scores: pd.DataFrame, *, baseline: str = "base", budget: int | None = None
) -> pd.DataFrame:
    """Per-method mean gain over ``baseline``, in the native metric.

    This is the column that goes NEXT TO Elo. Reported per task type, never
    pooled: binary is ROC-AUC, multiclass is -log-loss, and averaging them is
    the mistake that started this whole line of work.
    """
    base = (
        scores[scores.coalition == baseline]
        .groupby(["dataset", "split"])
        .score.mean()
        .rename("base")
        .reset_index()
    )
    df = scores.copy()
    if budget is None:
        df = df[df.budget == df.groupby("coalition").budget.transform("max")]
    else:
        df = df[df.budget == budget]
    d = df.merge(base, on=["dataset", "split"])
    d["gain"] = d.score - d.base
    g = (
        d.groupby(["coalition", "task"])
        .agg(M=("budget", "max"), gain=("gain", "mean"), n_ds=("dataset", "nunique"))
        .reset_index()
    )
    per_ds = (
        d.groupby(["coalition", "task", "dataset"])
        .gain.mean()
        .groupby(level=[0, 1])
        .apply(lambda s: (s > 0).sum())
        .rename("helps")
    )
    return g.merge(per_ds.reset_index(), on=["coalition", "task"])


# --------------------------------------------------------------------- CLI --
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download the pool (LOGIN NODE ONLY)")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--suite", default=FROZEN_SUITE)
    f.add_argument("--force", action="store_true")
    f.add_argument("--user-agent", default=USER_AGENT)

    pr = sub.add_parser("probe", help="diagnose a failing fetch")
    pr.add_argument("--suite", default=FROZEN_SUITE)
    pr.add_argument("--user-agent", default=USER_AGENT)

    u = sub.add_parser("urls", help="write a URL list to fetch elsewhere")
    u.add_argument("--out", type=Path, default=Path("tabarena_urls.txt"))
    u.add_argument("--suite", default=FROZEN_SUITE)
    u.add_argument(
        "--script",
        type=Path,
        default=None,
        help="also write a curl script producing the expected layout",
    )

    i = sub.add_parser("inspect", help="summarise a downloaded pool")
    i.add_argument("--pool", type=Path, required=True)

    c = sub.add_parser("checksums", help="copy the manifest out of a tabarena clone")
    c.add_argument("--tabarena", type=Path, required=True)

    a = p.parse_args(argv)
    if a.cmd == "fetch":
        fetch_pool(a.out, a.suite, force=a.force, user_agent=a.user_agent)
    elif a.cmd == "probe":
        return probe(a.suite, user_agent=a.user_agent)
    elif a.cmd == "urls":
        write_urls(a.out, a.suite, a.script)
    elif a.cmd == "inspect":
        pool = load_pool(
            a.pool if a.pool.name.startswith("tabarena-") else a.pool / FROZEN_SUITE
        )
        print("\nmethods:")
        for m, n in pool.method.value_counts().items():
            print(f"  {m:46s} {n:5d} rows")
        print(
            f"\nunits per dataset: "
            f"{pool.groupby('dataset').fold.nunique().value_counts().to_dict()}"
        )
        print(f"metrics: {pool.metric.value_counts().to_dict()}")
    elif a.cmd == "checksums":
        src = (
            a.tabarena
            / "packages/tabarena/src/tabarena/models/_artifacts"
            / "results_checksums.json"
        )
        CHECKSUMS.parent.mkdir(parents=True, exist_ok=True)
        CHECKSUMS.write_text(src.read_text())
        print(
            f"copied {src} -> {CHECKSUMS} "
            f"({len(json.loads(CHECKSUMS.read_text()))} artifacts)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
