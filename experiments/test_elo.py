import sys, numpy as np, pandas as pd
sys.path.insert(0,".")
import elo as E
ok=True
def chk(c,m):
    global ok; print(("  PASS  " if c else "  FAIL  ")+m); ok=ok and c

print("=== metric_error conversion ===")
s=np.array([0.85,0.92,-0.31,-1.20]); t=np.array(["binary","binary","multiclass","multiclass"])
me=E.to_metric_error(s,t)
chk(np.allclose(me,[0.15,0.08,0.31,1.20]), f"binary->1-AUC, multiclass->log_loss: {me.round(3)}")
try: E.to_metric_error(np.array([1.5]),np.array(["binary"])); chk(False,"negative error rejected")
except ValueError: chk(True,"negative metric_error rejected")

def synth(n_ds=12, seed=0, folds=None, strengths=(0.0,0.3,0.6,1.0)):
    rng=np.random.default_rng(seed); rows=[]
    for d in range(n_ds):
        nf = folds[d] if folds else (9 if d%3 else 30)
        for f in range(nf):
            base=rng.normal(0,1.0)
            for i,st in enumerate(strengths):
                rows.append(dict(method=f"M{i}",dataset=f"ds{d}",fold=f,
                                 metric_error=float(np.exp(base - st + rng.normal(0,.5)))))
    return pd.DataFrame(rows)

df=synth()
print("\n=== cross-check vs the real bencheval ===")
d=E.crosscheck_against_bencheval(df, anchor="M0", bootstrap_rounds=20)
chk(d is not None, "bencheval available")
if d is not None:
    print(d.round(9).to_string())
    chk(d["diff"].abs().max()<1e-6, f"max |ours - bencheval| = {d['diff'].abs().max():.3e} Elo")

r=E.elo_ratings(df, anchor="M0", prefer_bencheval=False, bootstrap_rounds=50).ratings
print("\n=== ordering & anchoring ===")
print(r.round(2).to_string())
chk(abs(r.loc["M0","elo"]-1000)<1e-9, "anchor pinned at 1000")
chk(list(r.index)==["M3","M2","M1","M0"], "stronger methods rank higher")
chk((r.elo_lo<r.elo).all() and (r.elo<r.elo_hi).all(), "point estimate inside its CI")

print("\n=== ties get equal Elo ===")
tie=pd.concat([df.assign(method="A"),df.assign(method="B")])
rt=E.elo_ratings(tie,anchor="A",prefer_bencheval=False,bootstrap_rounds=10).ratings
chk(abs(rt.elo.iloc[0]-rt.elo.iloc[1])<1e-6, f"identical methods -> identical Elo (gap {abs(rt.elo.iloc[0]-rt.elo.iloc[1]):.2e})")

print("\n=== every dataset carries weight 1, regardless of fold count ===")
#  one dataset with 300 folds must NOT outvote 11 datasets with 9
# REPLICATE ds0's existing folds 20x -- same information, 20x the units. Under
# equal-dataset weighting Elo must not move; under per-fold weighting it would.
a=synth(n_ds=12,seed=1,folds=[9]*12)
dup=[a]
d0=a[a.dataset=="ds0"]
for r in range(1,20):
    dup.append(d0.assign(fold=d0.fold+9*r))
b=pd.concat(dup,ignore_index=True)
ra=E.elo_ratings(a,anchor="M0",prefer_bencheval=False,bootstrap_rounds=10).ratings.elo
rb=E.elo_ratings(b,anchor="M0",prefer_bencheval=False,bootstrap_rounds=10).ratings.elo
shift=(rb-ra.reindex(rb.index)).abs().max()
nb=b.groupby(["dataset","fold"]).ngroups
chk(shift<1e-6, f"replicating one dataset's folds 20x ({nb} units) moves Elo by {shift:.2e}")
# the control: per-fold weighting WOULD be swayed. Verify by checking the raw
# win-total matrix column for ds0 keeps total weight 1.
_,W=E._win_totals(b,"method","dataset","fold","metric_error")
chk(abs(W.sum(axis=0)[0]-sum(range(len(a.method.unique()))))<1e-9,
    f"ds0 column still sums to the per-unit win total of a single unit ({W.sum(axis=0)[0]:.1f})")

print("\n=== win-rate link ===")
chk(abs(E.win_rate(1100,1000)-0.6401)<1e-3, f"+100 Elo = {E.win_rate(1100,1000):.4f} win rate")
chk(abs(E.win_rate(1000,1000)-0.5)<1e-12, "equal Elo = 50%")

print("\n=== ragged schedule is refused, not imputed ===")
bad=df[~((df.method=="M1")&(df.dataset=="ds0")&(df.fold==0))]
try: E.elo_ratings(bad,anchor="M0",prefer_bencheval=False); chk(False,"ragged refused")
except ValueError as e: chk("ragged schedule" in str(e),"ragged schedule raises with a diagnostic")
sub=E.complete_subgrid(bad)
chk(E.elo_ratings(sub,anchor="M0",prefer_bencheval=False,bootstrap_rounds=5) is not None,
    "complete_subgrid repairs it")

print("\n"+("ALL PASS" if ok else "FAILURES")); sys.exit(0 if ok else 1)
