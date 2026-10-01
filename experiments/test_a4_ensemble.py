import sys, types, numpy as np, pandas as pd
sys.path.insert(0,".")

# --- stub tabicl so the ensemble wiring can be exercised without a GPU ------
seen = []
class Stub:
    average_logits=True; softmax_temperature=0.9
    def __init__(self,**kw): self.kw=kw
    def fit(self,X,y):
        self.d=np.asarray(X).shape[1]
        self.cols=list(X.columns) if hasattr(X,"columns") else None
        self.classes_=np.unique(y); seen.append(self.cols); return self
    def predict_members(self,X):
        assert np.asarray(X).shape[1]==self.d, \
            f"test columns {np.asarray(X).shape[1]} != train columns {self.d}"
        assert (list(X.columns)==self.cols) if hasattr(X,"columns") else True, \
            "train/test column sets differ for the same member"
        rs=np.random.default_rng(abs(hash(tuple(self.cols or []))) % 2**31)
        return rs.normal(size=(1,np.asarray(X).shape[0],len(self.classes_)))
cap=types.ModuleType("experiments.capture"); cap.MemberCapturingTabICLClassifier=Stub
cap.verify_equivalence=lambda *a,**k: None
sys.modules["experiments.capture"]=cap

from experiments.subsample import FeatureSubsampleEnsemble

rs=np.random.default_rng(0); n,d,nt=400,20,60
Xtr=pd.DataFrame(rs.normal(size=(n,d)),columns=[f"c{i}" for i in range(d)])
ytr=rs.integers(0,3,n); Xte=pd.DataFrame(rs.normal(size=(nt,d)),columns=Xtr.columns)

ok=True
def chk(c,m):
    global ok; print(("  PASS  " if c else "  FAIL  ")+m); ok=ok and c

for name,kw in [("A4_g25",dict(frac=.25,mode="random")),
                ("A4_g90",dict(frac=.90,mode="random")),
                ("A4rr_g50",dict(frac=.50,mode="roundrobin")),
                ("A4imp_g50",dict(frac=.50,mode="importance")),
                ("A4lofo_M8",dict(mode="lofo",n_folds=8))]:
    seen.clear()
    fe=FeatureSubsampleEnsemble(n_estimators=16,seed=0,**kw)
    out=fe.fit_predict_members(Xtr,ytr,Xte)
    t=fe.timings
    M = kw.get("n_folds") or 16
    chk(out.shape==(M,nt,3), f"{name:10s} members shape {out.shape} == ({M},{nt},3)")
    chk(len(seen)==M and all(s is not None for s in seen), f"{name:10s} {M} separate fits, each on a named column subset")
    chk(t["mode"]==kw.get("mode"), f"{name:10s} mode recorded = {t['mode']}")
    chk(abs(t["frac_effective"]-(1-1/M if kw.get('mode')=='lofo' else kw.get('frac',0)))<0.03,
        f"{name:10s} frac_effective {t['frac_effective']:.4f}")
    chk(abs(t["pairwise_overlap_mean"]-t["pairwise_overlap_expected"])<0.12,
        f"{name:10s} overlap {t['pairwise_overlap_mean']:.3f} ~ expected {t['pairwise_overlap_expected']:.3f}")
    chk(t["n_features_total"]==d and t["subset_size_min"]>=2, f"{name:10s} coverage keys sane (unused={t['n_unused_columns']}, CV={t['usage_cv']:.3f})")
    chk(t["importance_used"]==(kw.get("mode")=="importance"), f"{name:10s} importance_used={t['importance_used']}")
    chk(np.isfinite(out).all(), f"{name:10s} no non-finite logits")

print("\n-- determinism: same seed -> identical subsets --")
a=FeatureSubsampleEnsemble(n_estimators=16,seed=0,frac=.5,mode="random").fit_predict_members(Xtr,ytr,Xte)
b=FeatureSubsampleEnsemble(n_estimators=16,seed=0,frac=.5,mode="random").fit_predict_members(Xtr,ytr,Xte)
c=FeatureSubsampleEnsemble(n_estimators=16,seed=1,frac=.5,mode="random").fit_predict_members(Xtr,ytr,Xte)
chk(np.array_equal(a,b),"seed 0 twice -> bitwise identical")
chk(not np.array_equal(a,c),"seed 1 -> different members")

print("\n"+("ALL PASS" if ok else "FAILURES")); sys.exit(0 if ok else 1)
