"""Full-campaign analysis. Supersedes analyze.py / analyze2.py (pilot-only).

Why those two must not be reused at full scale:
  * analyze2.py hard-codes the pilot's Shapley values, so it would report pilot
    numbers as if they were the campaign's;
  * analyze.py pools ROC-AUC with negative log-loss into one mean, which is
    meaningless across task types and inverted the shipped-default result in
    the pilot;
  * neither knows about `split_severity`.

Two stages, because extraction is the expensive half and the tables are the
half you iterate on:

    python -m experiments.analyze_full extract --cache $CACHE_DIR --out results/
    python -m experiments.analyze_full report  --out results/

`extract` streams the cache cell by cell, appending to results/scores.csv, and
skips cells already present -- so it resumes after an interrupt.
`report` is pure pandas over that CSV and takes seconds.

Every table is reported SEPARATELY for binary (ROC-AUC) and multiclass
(negative log-loss). They are never pooled.

Severity filter (`--severity`):
    ok        only splits matching TabArena's published train/test sizes
    usable    ok + resized (default) -- excludes leakage only
    all       everything, including overlap
Use `ok` for any claim against the published leaderboard; `usable` for trends.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")

BUDGETS = [1, 2, 4, 8, 16, 32]
AXES = ["A1", "A2", "A3"]
ORDER = ["base", "A1", "A2", "A3", "A1A2", "A1A3", "A2A3", "A1A2A3", "shipped"]
SUBSET_REPS = 5          # random member subsets averaged when budget < M


# ---------------------------------------------------------------- extract ---
def softmax(x, t):
    z = (x / t)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def cell_score(p, y, n_classes):
    """TabArena metric convention. Higher is better in both cases."""
    if n_classes == 2:
        return roc_auc_score(y, p[:, 1])
    return -log_loss(y, p, labels=np.arange(n_classes))


def extract(cache: Path, out: Path, limit: int | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    scores_f = out / "scores.csv"

    done: set[tuple] = set()
    if scores_f.exists():
        prev = pd.read_csv(scores_f, usecols=["dataset", "coalition", "split"])
        done = set(map(tuple, prev.drop_duplicates().values))
        print(f"resuming: {len(done)} cells already extracted")

    metas = sorted(cache.rglob("meta.json"))
    print(f"cache: {len(metas)} cells at {cache}")
    if limit:
        metas = metas[:limit]

    rng = np.random.default_rng(0)
    rows, n_new, n_err = [], 0, 0

    for i, mf in enumerate(metas, 1):
        m = json.loads(mf.read_text())
        key = (m["dataset"], m["coalition"], m["split"])
        if key in done:
            continue
        d = mf.parent
        try:
            members = np.load(d / "members.npy").astype(np.float32)
            y = np.load(d / "y_test.npy", allow_pickle=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [err] {d}: {e}")
            n_err += 1
            continue

        classes = np.unique(y)
        y_idx = np.searchsorted(classes, y)
        C, M = len(classes), members.shape[0]
        tau = m.get("softmax_temperature", 0.9)

        # effective member count on the member error matrix
        err = (members.argmax(-1) != y_idx[None, :]).astype(np.float64)
        if M > 1 and err.std() > 0:
            R = np.nan_to_num(np.corrcoef(err), nan=0.0)
            np.fill_diagonal(R, 1.0)
            meff = float(M**2 / (R**2).sum())
        else:
            meff = 1.0

        # consensus set: fraction of test points where all members agree
        preds = members.argmax(-1)
        consensus = float((preds == preds[0]).all(0).mean()) if M > 1 else 1.0

        base_row = dict(
            dataset=m["dataset"], coalition=m["coalition"], split=m["split"],
            m_realised=m["m_realised"], requested=m["n_estimators_requested"],
            n_classes=m.get("n_classes", C), n_features=m["n_features"],
            n_train=m["n_train"], n_test=m["n_test"],
            task="binary" if C == 2 else "multiclass",
            severity=m.get("split_severity", "unknown"),
            verified=m.get("split_verified", None),
            fit_s=m["fit_seconds"], pred_s=m["predict_seconds"],
            sec_per_member_total=(m["fit_seconds"] + m["predict_seconds"]) / max(1, M),
            peak_mem=m.get("peak_mem_bytes"),
            meff=meff, meff_ratio=meff / max(1, M), consensus=consensus,
            tabicl_sha=m.get("tabicl_sha"), code_sha=m.get("code_sha"),
        )

        for B in BUDGETS:
            b = min(B, M)
            reps = 1 if b == M else SUBSET_REPS
            s = []
            for _ in range(reps):
                idx = rng.choice(M, size=b, replace=False) if b < M else np.arange(M)
                p = softmax(members[idx].mean(0), tau)
                p = p / p.sum(-1, keepdims=True)
                s.append(cell_score(p, y_idx, C))
            rows.append({**base_row, "budget": B, "eff_budget": b,
                         "score": float(np.mean(s))})

        n_new += 1
        if n_new % 200 == 0:
            pd.DataFrame(rows).to_csv(scores_f, mode="a", index=False,
                                      header=not scores_f.exists())
            rows = []
            print(f"  [{i}/{len(metas)}] {n_new} extracted", flush=True)

    if rows:
        pd.DataFrame(rows).to_csv(scores_f, mode="a", index=False,
                                  header=not scores_f.exists())
    print(f"\nextracted {n_new} new cells ({n_err} errors) -> {scores_f}")


# ----------------------------------------------------------------- report ---
def shapley(vals: dict) -> dict:
    """Exact Shapley over {A1,A2,A3} from the 8 measured coalition values."""
    n, S = len(AXES), {}
    for a in AXES:
        others = [x for x in AXES if x != a]
        tot = 0.0
        for k in range(len(others) + 1):
            for comb in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                wo = "".join(sorted(comb)) or "base"
                wi = "".join(sorted([*comb, a]))
                tot += w * (vals.get(wi, 0.0) - vals.get(wo, 0.0))
        S[a] = tot
    return S


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def report(out: Path, severity: str) -> None:
    df = pd.read_csv(out / "scores.csv").drop_duplicates(
        subset=["dataset", "coalition", "split", "budget"])

    keep = {"ok": ["ok"],
            "usable": ["ok", "resized", "unknown"],
            "all": None}[severity]
    if keep is not None:
        before = len(df)
        df = df[df.severity.isin(keep)]
        print(f"severity filter '{severity}': kept {len(df)}/{before} rows")

    hdr("T0  INVENTORY")
    inv = df[df.budget == 32]
    print(f"datasets   : {inv.dataset.nunique()}")
    print(f"coalitions : {inv.coalition.nunique()}")
    print(f"cells      : {len(inv)}")
    print(f"\nby task type:\n{inv.groupby('task').dataset.nunique().to_string()}")
    print(f"\nsplit severity:\n{inv.severity.value_counts().to_string()}")
    trunc = inv[inv.m_realised < inv.requested]
    print(f"\ncells where M_realised < requested: {len(trunc)}/{len(inv)}")
    if len(trunc):
        print(trunc.groupby("coalition").m_realised.agg(["min", "median", "max"])
              .round(1).to_string())

    # --- gain vs single unperturbed pass -------------------------------------
    base = (df[df.coalition == "base"].groupby(["dataset", "split"])["score"]
            .mean().rename("base_score").reset_index())
    d = df.merge(base, on=["dataset", "split"])
    d["gain"] = d.score - d.base_score

    hdr("T1  EFFECTIVE MEMBER COUNT  (the headline)")
    mm = (inv.groupby("coalition")
          .agg(M=("m_realised", "mean"), meff=("meff", "mean"),
               ratio=("meff_ratio", "mean"), consensus=("consensus", "mean")))
    print(mm.reindex([c for c in ORDER if c in mm.index]).round(3).to_string())
    print("\nratio = independent predictions purchased per nominal member.")

    from scipy.stats import wilcoxon
    for task, sub in d.groupby("task"):
        metric = "ROC-AUC" if task == "binary" else "-log-loss"
        hdr(f"T2  GAIN OVER SINGLE PASS - {task} ({metric}), "
            f"{sub.dataset.nunique()} datasets")
        t = sub.groupby(["coalition", "budget"])["gain"].mean().unstack()
        print(t.reindex([c for c in ORDER if c in t.index]).round(5).to_string())

        print(f"\n  paired Wilcoxon vs base @ M=32 (across dataset-splits):")
        for c in ORDER[1:]:
            s = sub[(sub.coalition == c) & (sub.budget == 32)]
            if len(s) < 8:
                continue
            try:
                _, p = wilcoxon(s.gain)
            except ValueError:
                p = 1.0
            per_ds = s.groupby("dataset")["gain"].mean()
            print(f"    {c:8s} mean {s.gain.mean():+.5f}  p={p:.4f}"
                  f"{'*' if p < .05 else ' '}  helps {(per_ds>0).sum()}/{len(per_ds)} ds")

        hdr(f"T3  SHAPLEY over {{A1,A2,A3}} @ M=32 - {task}")
        per_ds = {}
        for ds, s in sub[sub.budget == 32].groupby("dataset"):
            v = s.groupby("coalition")["gain"].mean().to_dict()
            v["base"] = 0.0
            per_ds[ds] = shapley(v)
        sh = pd.DataFrame(per_ds).T
        if len(sh):
            summary = pd.DataFrame({
                "mean_phi": sh.mean(), "std": sh.std(),
                "min": sh.min(), "max": sh.max(),
                "frac_positive": (sh > 0).mean()})
            print(summary.round(5).to_string())
            print(f"\n  (per-dataset spread vs mean tells you whether the pooled")
            print(f"   attribution is a population statement or a per-dataset one)")

        hdr(f"T4  COST AND EFFICIENCY-ADJUSTED ATTRIBUTION - {task}")
        cost = (inv[inv.task == task].groupby("coalition")
                .agg(fit=("fit_s", "mean"), pred=("pred_s", "mean"),
                     M=("m_realised", "mean"),
                     s_per_member=("sec_per_member_total", "mean")))
        cost["total"] = cost.fit + cost.pred
        if "base" in cost.index:
            cost["vs_single_pass"] = cost.total / cost.loc["base", "total"]
        print(cost.reindex([c for c in ORDER if c in cost.index]).round(3).to_string())
        if len(sh):
            eff = pd.DataFrame({"phi": sh.mean()})
            eff["s_per_member"] = cost["s_per_member"]
            eff["phi_per_gpu_sec"] = eff.phi / eff.s_per_member
            print("\n  phi-tilde (gain per GPU-second per member):")
            print(eff.round(5).to_string())

        hdr(f"T5  IS THE SHIPPED DEFAULT ON THE FRONTIER? - {task}")
        g8 = sub[sub.budget == 8].groupby("coalition")["gain"].mean()
        g32 = sub[sub.budget == 32].groupby("coalition")["gain"].mean()
        fr = pd.DataFrame({"gain@8": g8, "gain@32": g32})
        fr["cost@32"] = cost["total"]
        fr = fr.reindex([c for c in ORDER if c in fr.index])
        print(fr.round(5).to_string())
        if "shipped" in g32.index:
            best8 = g8.drop("base", errors="ignore").idxmax()
            print(f"\n  best @ M=8 : {best8}  ({g8[best8]:+.5f})")
            print(f"  shipped@32 : {g32['shipped']:+.5f}")
            if g32["shipped"] > 0:
                print(f"  ratio      : {g8[best8]/g32['shipped']:.2f}x the gain "
                      f"at 1/4 the members")
            elif g8[best8] > 0:
                print("  -> shipped is NET NEGATIVE at 4x the budget; any "
                      "positive coalition dominates it outright.")

        hdr(f"T6  RECOVERY DECOMPOSITION - {task}")
        m1 = sub[sub.budget == 1].groupby("coalition")["gain"].mean()
        rec = pd.DataFrame({"single_member": m1, "ensemble_M32": g32})
        rec["swing"] = rec.ensemble_M32 - rec.single_member
        rec["pct_repair"] = np.where(
            rec.swing.abs() < 1e-12, np.nan,
            (-rec.single_member / rec.swing * 100).clip(0, 100))
        print(rec.reindex([c for c in ORDER if c in rec.index]).round(4).to_string())
        print("\n  pct_repair = share of the swing that is undoing the damage")
        print("  perturbation did to the individual member.")

    hdr("T7  SATURATION - gain per GPU-second vs budget (shipped)")
    for task, sub in d[d.coalition == "shipped"].groupby("task"):
        c = inv[(inv.task == task) & (inv.coalition == "shipped")]
        fit, spm = c.fit_s.mean(), c.sec_per_member_total.mean()
        rows = []
        for B in BUDGETS:
            g = sub[sub.budget == B]["gain"].mean()
            sec = fit + spm * B
            rows.append(dict(budget=B, gain=g, gpu_sec=sec, gain_per_sec=g / sec))
        print(f"\n  {task}:")
        print(pd.DataFrame(rows).round(6).to_string(index=False))

    hdr("T8  PER-DATASET, shipped @ M=32")
    pds = (d[(d.coalition == "shipped") & (d.budget == 32)]
           .groupby(["task", "dataset"])
           .agg(base=("base_score", "mean"), ens=("score", "mean"),
                gain=("gain", "mean"), meff=("meff", "mean"),
                n_splits=("split", "nunique")).reset_index())
    print(pds.sort_values(["task", "gain"]).round(5).to_string(index=False))
    print(f"\n  helps on {(pds.gain>0).sum()}/{len(pds)} datasets")

    hdr("COST TOTAL")
    cells = inv.drop_duplicates(subset=["dataset", "coalition", "split"])
    print(f"campaign wall-clock (fit+predict): "
          f"{(cells.fit_s.sum()+cells.pred_s.sum())/3600:.2f} GPU-hours")

    d.to_csv(out / "gains.csv", index=False)
    print(f"\ntidy gains -> {out/'gains.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--cache", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--limit", type=int, default=None)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--severity", default="usable",
                   choices=["ok", "usable", "all"])
    a = ap.parse_args()

    if a.cmd == "extract":
        extract(a.cache, a.out, a.limit)
    else:
        report(a.out, a.severity)


if __name__ == "__main__":
    main()
