import sys, pandas as pd, numpy as np
sys.path.insert(0,".")
import arena as A, elo as E
ok=True
def chk(c,m):
    global ok; print(("  PASS  " if c else "  FAIL  ")+m); ok=ok and c

print("=== checksum manifest ===")
k=A.artifact_keys()
chk(len(k)==42, f"frozen suite has {len(k)} artifacts (expect 42)")
meth=sorted({x.split('/')[3] for x in k})
chk(len(meth)==22, f"{len(meth)} methods: {', '.join(meth[:6])}...")
print("  sample URL:", A.BASE_URL+sorted(k)[0])

print("\n=== TabFM shipped results ===")
tf=A.load_tabfm_shipped(__import__('pathlib').Path("tabfm_repo"))
chk(len(tf)==1188 and set(tf.method)=={"TabFM","TabFM-Ensemble"}, f"{len(tf)} rows, methods {sorted(set(tf.method))}")
u=tf.groupby('dataset').fold.nunique().value_counts().to_dict()
chk(u=={9:26,30:12}, f"fold counts per dataset: {u}  (26x9 + 12x30 = 594)")

print("\n=== our scores -> leaderboard schema ===")
S=pd.read_csv("scores.csv")
ours=A.our_results(S, model="TabICLv2", coalitions=["base","shipped","A1A3","A1A2A3"])
print(ours.head(3).to_string(index=False))
chk(set(ours.columns)==set(A.NEEDED), "schema matches")
chk(ours.metric_error.min()>=0 and ours.metric_error.max()<2, f"metric_error in [{ours.metric_error.min():.4f}, {ours.metric_error.max():.4f}]")
b=ours[ours.problem_type=="binary"]
chk((b.metric=="roc_auc").all(), "binary -> roc_auc")
# spot check the conversion against the raw score
row=S[(S.coalition=="base")&(S.budget==1)].iloc[0]
m=ours[(ours.dataset==row.dataset)&(ours.fold==row.split)&(ours.method.str.contains("base"))]
exp = 1-row.score if row.task=="binary" else -row.score
chk(abs(float(m.metric_error.iloc[0])-exp)<1e-12, f"spot check {row.dataset} f{row.split}: {float(m.metric_error.iloc[0]):.6f} == {exp:.6f}")

print("\n=== units align with TabArena's ===")
ta=set(zip(tf.dataset,tf.fold)); ou=set(zip(ours.dataset,ours.fold))
chk(ou<=ta, f"our {len(ou)} units are a subset of TabArena's {len(ta)}")
chk(len(ou)==594, f"we cover all {len(ou)} classification units")

print("\n=== end-to-end Elo: our coalitions + TabFM's two ===")
pool=pd.concat([ours,tf],ignore_index=True)
pool=E.complete_subgrid(pool)
r=E.elo_ratings(pool, anchor="TabFM", bootstrap_rounds=50, prefer_bencheval=False)
print(r.ratings.round(1).to_string())
chk(r.ratings.loc["TabFM-Ensemble","elo"]>r.ratings.loc["TabFM","elo"],
    "TabFM-Ensemble > TabFM, reproducing their own ordering")
d=E.crosscheck_against_bencheval(pool, anchor="TabFM", bootstrap_rounds=10)
chk(d is not None and d["diff"].abs().max()<1e-6, f"cross-check vs bencheval: max diff {d['diff'].abs().max():.2e} Elo")

print("\n=== effect sizes, the column that goes next to Elo ===")
es=A.effect_sizes(S)
print(es[es.coalition.isin(["shipped","A1A3","A1A2A3"])].round(5).to_string(index=False))
chk(len(es)>0 and "gain" in es.columns, "effect sizes computed per task type")

print("\n"+("ALL PASS" if ok else "FAILURES")); sys.exit(0 if ok else 1)
