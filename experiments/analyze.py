"""Pilot analysis: the four questions from EXPERIMENTS.md §4."""
import json, itertools, warnings, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

warnings.filterwarnings("ignore")
ROOT = Path("cache")
RNG = np.random.default_rng(0)
BUDGETS = [1, 2, 4, 8, 16, 32]
AXES = ["A1", "A2", "A3"]


def softmax(x, t=0.9):
    z = x / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def score(p, y, n_classes):
    """TabArena convention: ROC-AUC binary, log-loss multiclass. Higher better."""
    if n_classes == 2:
        return roc_auc_score(y, p[:, 1])
    return -log_loss(y, p, labels=np.arange(n_classes))


def acc(p, y):
    return accuracy_score(y, p.argmax(-1))


rows, meff_rows = [], []
for meta_f in sorted(ROOT.rglob("meta.json")):
    m = json.loads(meta_f.read_text())
    cell = meta_f.parent
    members = np.load(cell / "members.npy").astype(np.float32)   # (M, n, C) logits
    y = np.load(cell / "y_test.npy", allow_pickle=True)
    classes = np.unique(y)
    y_idx = np.searchsorted(classes, y)
    C, M = len(classes), members.shape[0]

    # effective member count on the error matrix
    err = (members.argmax(-1) != y_idx[None, :]).astype(np.float64)
    if M > 1 and err.std() > 0:
        R = np.corrcoef(err)
        R = np.nan_to_num(R, nan=0.0)
        np.fill_diagonal(R, 1.0)
        meff = M**2 / (R**2).sum()
    else:
        meff = 1.0
    meff_rows.append(dict(dataset=m["dataset"], coalition=m["coalition"],
                          split=m["split"], M=M, meff=meff))

    for B in BUDGETS:
        b = min(B, M)
        reps = 1 if b == M else 5
        s, a = [], []
        for _ in range(reps):
            idx = RNG.choice(M, size=b, replace=False) if b < M else np.arange(M)
            p = softmax(members[idx].mean(0), m["softmax_temperature"])
            p = p / p.sum(-1, keepdims=True)
            s.append(score(p, y_idx, C)); a.append(acc(p, y_idx))
        rows.append(dict(dataset=m["dataset"], coalition=m["coalition"],
                         split=m["split"], budget=B, eff_budget=b, M=M,
                         score=np.mean(s), acc=np.mean(a),
                         n_classes=C, n_features=m["n_features"],
                         sec_per_member=m["seconds_per_member"],
                         fit_s=m["fit_seconds"], pred_s=m["predict_seconds"],
                         m_realised=m["m_realised"],
                         requested=m["n_estimators_requested"]))

df = pd.DataFrame(rows)
meff = pd.DataFrame(meff_rows)
df.to_csv("pilot_scores.csv", index=False)

print("=" * 78)
print("Q0  REALISED vs REQUESTED MEMBERS  (structural caps)")
print("=" * 78)
cap = (df[df.budget == 32].groupby(["dataset", "coalition"])
       .agg(n_classes=("n_classes", "first"), n_feat=("n_features", "first"),
            M=("m_realised", "first")).reset_index())
piv = cap.pivot(index="dataset", columns="coalition", values="M")
order = ["base", "A1", "A2", "A3", "A1A2", "A1A3", "A2A3", "A1A2A3", "shipped"]
print(piv[order].to_string())
print("\n(requested 32 everywhere; cells below 32 hit a structural cap)")

print()
print("=" * 78)
print("Q1  IS THE EFFECT ABOVE THE NOISE?")
print("=" * 78)
base = (df[df.coalition == "base"].groupby(["dataset", "split"])["score"]
        .mean().rename("base_score").reset_index())
d = df.merge(base, on=["dataset", "split"])
d["gain"] = d.score - d.base_score

noise = (d[d.coalition == "base"].groupby("dataset")["score"]
         .agg(["mean", "std", "count"]))
noise["sem"] = noise["std"] / np.sqrt(noise["count"])
noise["MDE_paired"] = 2.8 * noise["sem"]          # ~80% power, alpha .05, paired
print("Per-dataset split-to-split variability of the single-member baseline:")
print(noise.round(4).to_string())

g32 = d[(d.budget == 32) & (d.coalition == "A1A2A3")].groupby("dataset")["gain"]
cmp = pd.DataFrame({"gain_A1A2A3_M32": g32.mean(),
                    "MDE": noise["MDE_paired"]})
cmp["detectable"] = cmp.gain_A1A2A3_M32.abs() > cmp.MDE
print("\nFull-perturbation gain vs per-dataset MDE:")
print(cmp.round(4).to_string())
print(f"\ndetectable on {int(cmp.detectable.sum())}/{len(cmp)} datasets")

print()
print("=" * 78)
print("Q2  FLAT OR SHARP?  mean gain over single member (all datasets)")
print("=" * 78)
tab = (d.groupby(["coalition", "budget"])["gain"].mean().unstack() * 100)
print(tab.reindex(order).round(3).to_string())
print("\n(score points x100; ROC-AUC for binary, -log-loss for multiclass)")

# paired Wilcoxon across dataset-splits at M=8 and M=32
from scipy.stats import wilcoxon
print("\nPaired Wilcoxon vs base, across 72 dataset-splits:")
for B in (8, 32):
    print(f"  budget {B}:")
    for c in order[1:]:
        sub = d[(d.coalition == c) & (d.budget == B)]
        if len(sub) < 10:
            continue
        try:
            st, p = wilcoxon(sub.gain)
        except ValueError:
            st, p = np.nan, 1.0
        flag = "*" if p < 0.05 else " "
        print(f"    {c:8s} mean {sub.gain.mean()*100:+7.3f}  p={p:.4f} {flag}")

print()
print("=" * 78)
print("Q3  SHAPLEY ATTRIBUTION over {A1,A2,A3}  (budget 32)")
print("=" * 78)


def shapley(vals):
    S = {a: 0.0 for a in AXES}
    n = len(AXES)
    for a in AXES:
        others = [x for x in AXES if x != a]
        for k in range(len(others) + 1):
            for comb in itertools.combinations(others, k):
                w = (math.factorial(k) * math.factorial(n - k - 1)
                     / math.factorial(n))
                key_wo = "".join(sorted(comb)) or "base"
                key_w = "".join(sorted(list(comb) + [a]))
                S[a] += w * (vals.get(key_w, 0.0) - vals.get(key_wo, 0.0))
    return S


per_ds = {}
for ds, sub in d[d.budget == 32].groupby("dataset"):
    vals = sub.groupby("coalition")["gain"].mean().to_dict()
    vals["base"] = 0.0
    per_ds[ds] = shapley(vals)
sh = pd.DataFrame(per_ds).T * 100
sh["sum"] = sh.sum(1)
sh["v(A1A2A3)"] = (d[(d.budget == 32) & (d.coalition == "A1A2A3")]
                   .groupby("dataset")["gain"].mean() * 100)
print(sh.round(3).to_string())
print("\nmean phi across datasets:")
print((sh[AXES].mean()).round(4).to_string())

print()
print("=" * 78)
print("Q4  COST PER AXIS  ->  phi-tilde = phi / GPU-seconds-per-member")
print("=" * 78)
cost = (df[df.budget == 32].groupby("coalition")
        .agg(sec_per_member=("sec_per_member", "mean"),
             fit_s=("fit_s", "mean"), pred_s=("pred_s", "mean")))
print(cost.round(4).to_string())

single = {a: cost.loc[a, "sec_per_member"] for a in AXES}
phi = sh[AXES].mean()
eff = pd.DataFrame({"phi_x100": phi,
                    "sec_per_member": pd.Series(single),
                    })
eff["phi_per_gpu_sec"] = eff.phi_x100 / eff.sec_per_member
print("\nefficiency-adjusted attribution:")
print(eff.round(4).to_string())

print()
print("=" * 78)
print("Q5  EFFECTIVE MEMBER COUNT")
print("=" * 78)
mm = meff.groupby("coalition").agg(M=("M", "mean"), meff=("meff", "mean"))
mm["ratio"] = mm.meff / mm.M
print(mm.reindex(order).round(3).to_string())

print()
print("=" * 78)
print("Q6  SATURATION  (gain vs budget, shipped coalition)")
print("=" * 78)
sat = (d[d.coalition == "shipped"].groupby(["dataset", "budget"])["gain"]
       .mean().unstack() * 100)
print(sat.round(3).to_string())

print()
print("=== campaign estimate ===")
tot_pred = df[df.budget == 32].groupby(["dataset", "coalition", "split"])["pred_s"].first().sum()
tot_fit = df[df.budget == 32].groupby(["dataset", "coalition", "split"])["fit_s"].first().sum()
print(f"pilot wall-clock (fit+predict, summed): {(tot_fit+tot_pred)/3600:.2f} GPU-hours")
print(f"  fit    {tot_fit/3600:.2f} h")
print(f"  predict{tot_pred/3600:.2f} h")
