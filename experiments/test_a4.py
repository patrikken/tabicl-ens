import numpy as np, sys
sys.path.insert(0,".")
from experiments.subsample import feature_subsets, expected_overlap, column_importance
from experiments.coalitions import (FEATURE_SUB_COALITIONS, A4_SWEEP, A4_VARIANTS,
                                    a4_applicable, is_feature_sub, is_context,
                                    destroys_information, all_coalitions)
ok=True
def chk(c,msg):
    global ok
    print(("  PASS  " if c else "  FAIL  ")+msg); ok = ok and c

print("== coalition wiring ==")
chk(set(A4_SWEEP+A4_VARIANTS)==set(FEATURE_SUB_COALITIONS), "sweep+variants == all A4")
chk(all(is_feature_sub(c) and not is_context(c) for c in FEATURE_SUB_COALITIONS), "A4 is feature_sub, not context")
chk(all(destroys_information(c) for c in list(FEATURE_SUB_COALITIONS)+["A5_f50"]), "A4/A5 destroy information")
chk(not any(destroys_information(c) for c in ["A1","A2","A3","A1A2A3","shipped","base"]), "A1-A3 preserve")
chk(len(all_coalitions())==19+8, f"all_coalitions has 27, got {len(all_coalitions())}")

print("\n== applicability gating ==")
chk(not a4_applicable("A4_g50", 6), "d=6 excluded (< 8)")
chk(a4_applicable("A4_g50", 8), "d=8 allowed for the sweep")
chk(not a4_applicable("A4lofo_M16", 20), "lofo M=16 excluded at d=20 (needs 32)")
chk(a4_applicable("A4lofo_M16", 32), "lofo M=16 allowed at d=32")
chk(a4_applicable("A4lofo_M8", 16), "lofo M=8 allowed at d=16")
chk(a4_applicable("A1", 3), "non-A4 coalitions unaffected")

print("\n== subset construction ==")
rng=np.random.default_rng(0)
for mode in ("random","roundrobin"):
    for d,M,g in [(20,16,.25),(20,16,.90),(50,16,.5),(8,16,.5),(1776,16,.25)]:
        S=feature_subsets(d,M,g,np.random.default_rng(0),mode)
        k=int(np.clip(round(g*d),2,d))
        sz={len(s) for s in S}
        uniq=all(len(np.unique(s))==len(s) for s in S)
        inb=all(s.min()>=0 and s.max()<d for s in S)
        chk(sz=={k} and uniq and inb, f"{mode:10s} d={d:4d} M={M} g={g}: all |S|={k}, unique, in range")

print("\n  -- round-robin really is more balanced --")
for d,M,g in [(20,16,.5),(50,16,.5),(100,16,.25)]:
    u={}
    for mode in ("random","roundrobin"):
        S=feature_subsets(d,M,g,np.random.default_rng(1),mode)
        usage=np.bincount(np.concatenate(S),minlength=d)
        u[mode]=(usage.std()/usage.mean(), (usage==0).sum())
    chk(u["roundrobin"][0] < u["random"][0],
        f"d={d} g={g}: usage CV rr {u['roundrobin'][0]:.3f} < random {u['random'][0]:.3f}"
        f" | unused rr={u['roundrobin'][1]} random={u['random'][1]}")

print("\n  -- lofo geometry: g = 1-1/M, overlap = (M-2)/(M-1) --")
for d,M in [(32,8),(64,16),(100,8),(1776,16)]:
    S=feature_subsets(d,M,0.0,np.random.default_rng(2),"lofo")
    g=np.mean([len(s) for s in S])/d
    ov=np.mean([len(np.intersect1d(S[i],S[i-1]))/len(S[i]) for i in range(1,M)])
    cov=np.bincount(np.concatenate(S),minlength=d)
    chk(abs(g-(1-1/M))<1e-9 and abs(ov-(M-2)/(M-1))<0.05 and (cov>0).all(),
        f"d={d:4d} M={M}: g={g:.5f} (want {1-1/M:.5f})  overlap={ov:.4f} (want {(M-2)/(M-1):.4f})  every column used")

print("\n  -- importance mode biases toward informative columns --")
rs=np.random.default_rng(3); n,d=800,20
Xn=rs.normal(size=(n,d)); yv=(Xn[:,0]+Xn[:,1]>0).astype(int)   # only cols 0,1 matter
import pandas as pd
imp=column_importance(pd.DataFrame(Xn,columns=[f"c{i}" for i in range(d)]),yv,seed=0)
chk(np.argsort(imp)[-2:].tolist() in ([0,1],[1,0]), f"ExtraTrees finds cols 0,1 (top2={np.argsort(imp)[-2:].tolist()})")
Si=feature_subsets(d,64,.25,np.random.default_rng(4),"importance",imp)
Sr=feature_subsets(d,64,.25,np.random.default_rng(4),"random")
fi=np.mean([int(0 in s or 1 in s) for s in Si]); fr=np.mean([int(0 in s or 1 in s) for s in Sr])
chk(fi>fr, f"informative cols appear in {fi:.0%} of importance members vs {fr:.0%} of random")
chk(all(len(s)==5 for s in Si), "importance mode preserves subset size")

print("\n  -- guards --")
try: feature_subsets(3,8,.5,rng,"random",min_features=4); chk(False,"too-few-columns raises")
except ValueError: chk(True,"too-few-columns raises")
try: feature_subsets(4,16,0.,rng,"lofo"); chk(False,"degenerate lofo raises")
except ValueError: chk(True,"degenerate lofo raises")
try: feature_subsets(20,8,.5,rng,"nope"); chk(False,"unknown mode raises")
except ValueError: chk(True,"unknown mode raises")
chk(expected_overlap(.25)==.25, "expected_overlap(g)=g")

print("\n"+("ALL PASS" if ok else "FAILURES ABOVE")); sys.exit(0 if ok else 1)
