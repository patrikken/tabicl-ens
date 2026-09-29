"""Follow-ups: true cost, sign-consistency, and the recovery decomposition."""
import json
from pathlib import Path
import numpy as np, pandas as pd

df = pd.read_csv("pilot_scores.csv")
order = ["base","A1","A2","A3","A1A2","A1A3","A2A3","A1A2A3","shipped"]

base = (df[df.coalition=="base"].groupby(["dataset","split"])["score"]
        .mean().rename("base_score").reset_index())
d = df.merge(base, on=["dataset","split"]); d["gain"] = d.score - d.base_score

print("="*78); print("C1  TRUE COST PER MEMBER  (fit + predict, not predict alone)")
print("="*78)
cell = (df[df.budget==32]
        .groupby(["dataset","coalition","split"])
        .agg(fit=("fit_s","first"), pred=("pred_s","first"),
             M=("m_realised","first")).reset_index())
cell["total"] = cell.fit + cell.pred
cost = cell.groupby("coalition").agg(fit=("fit","mean"), pred=("pred","mean"),
                                     total=("total","mean"), M=("M","mean"))
cost["sec_per_member_total"] = cost.total / cost.M
cost["vs_base_cost"] = cost.total / cost.loc["base","total"]
print(cost.reindex(order).round(3).to_string())
print("\nA3 builds 5 preprocessing pipelines at fit time; dividing only predict")
print("time by member count understates it ~6x.")

phi = {"A1":0.0017,"A2":0.0282,"A3":0.1428}   # from analyze.py Q3
eff = pd.DataFrame({"phi_x100": pd.Series(phi)})
eff["sec_per_member_total"] = cost["sec_per_member_total"]
eff["phi_per_gpu_sec"] = eff.phi_x100 / eff.sec_per_member_total
print("\nefficiency-adjusted attribution (corrected):")
print(eff.round(4).to_string())

print(); print("="*78)
print("C2  SIGN CONSISTENCY  (datasets where the coalition beats 1 plain pass)")
print("="*78)
for B in (8,32):
    sub = d[d.budget==B]
    t = sub.groupby(["coalition","dataset"])["gain"].mean().unstack()
    res = pd.DataFrame({"helps": (t>0).sum(1), "hurts": (t<=0).sum(1),
                        "mean_x100": t.mean(1)*100})
    print(f"\nbudget {B}:"); print(res.reindex(order[1:]).round(3).to_string())

print(); print("="*78)
print("C3  RECOVERY DECOMPOSITION")
print("="*78)
m1 = d[d.budget==1].groupby("coalition")["gain"].mean()*100
m32 = d[d.budget==32].groupby("coalition")["gain"].mean()*100
rec = pd.DataFrame({"single_member_vs_base": m1, "ensemble_vs_base": m32})
rec["swing_from_own_member"] = rec.ensemble_vs_base - rec.single_member_vs_base
rec["pct_of_swing_that_is_repair"] = (-rec.single_member_vs_base /
                                      rec.swing_from_own_member * 100).clip(0,100)
print(rec.reindex(order[1:]).round(3).to_string())
print("\nPerturbing degrades the individual member; most of the apparent")
print("'ensemble gain' is climbing back out of that hole.")

print(); print("="*78)
print("C4  WHAT DOES THE SHIPPED DEFAULT BUY?")
print("="*78)
sec_per = cost.loc["shipped","total"] / cost.loc["shipped","M"]
rows=[]
for B in [1,2,4,8,16,32]:
    g = d[(d.coalition=="shipped") & (d.budget==B)]["gain"].mean()*100
    rows.append(dict(budget=B, gain_x100=g, gpu_sec=cost.loc["shipped","fit"]+sec_per*B))
s = pd.DataFrame(rows)
s["gain_per_gpu_sec"] = s.gain_x100/s.gpu_sec
s["vs_M32_compute"] = s.gpu_sec/s.gpu_sec.iloc[-1]
print(s.round(4).to_string(index=False))

best = d[d.budget==8].groupby("coalition")["gain"].mean().idxmax()
bg = d[d.budget==8].groupby("coalition")["gain"].mean().max()*100
sg = d[(d.coalition=="shipped")&(d.budget==32)]["gain"].mean()*100
print(f"\nbest coalition at budget 8 : {best}  (+{bg:.3f})")
print(f"shipped at budget 32       : +{sg:.3f}")
print(f"-> {bg/sg:.1f}x the gain at 1/4 the members")

print(); print("="*78)
print("C5  ABSOLUTE SCALE")
print("="*78)
bin_ds = d[d.n_classes==2]
b32 = bin_ds[(bin_ds.coalition=="shipped")&(bin_ds.budget==32)]["gain"].mean()
b1  = bin_ds[(bin_ds.coalition=="base")&(bin_ds.budget==1)]["score"].mean()
print(f"binary datasets: single pass ROC-AUC = {b1:.4f}")
print(f"shipped 32-member ensemble adds       = {b32:+.5f} ROC-AUC")
print(f"cost                                  = {cost.loc['shipped','vs_base_cost']:.1f}x a single pass")
