import sys, types, numpy as np, pandas as pd, json, tempfile
from pathlib import Path
sys.path.insert(0,".")
import sklearn.ensemble, scipy.stats          # before the torch stub
ok=True
def chk(c,m):
    global ok; print(("  PASS  " if c else "  FAIL  ")+m); ok=ok and c

# ---- stub the tabfm package, mimicking the real member/aggregate boundary ----
class _Gen:
    def __init__(s,E,norms): s.ensemble_configs_={n:[(np.arange(3),i,None,None)
        for i in range(E//len(norms))] for n in norms}
from sklearn.base import BaseEstimator
class StubClf(BaseEstimator):
    def __init__(s, model=None, n_estimators=32, norm_methods=None,
                 feat_shuffle_method="random", class_shift=True,
                 permute_categorical=False, max_num_features=500,
                 max_num_rows=None, softmax_temperature=0.9, average_logits=True,
                 enable_nnls=False, n_feature_crosses=0, n_svd_features=0,
                 random_state=42, **kw):
        if average_logits and enable_nnls:
            raise ValueError("average_logits and enable_nnls cannot both be True.")
        s.__dict__.update(locals()); del s.s
        s.norm_methods = norm_methods or ["none","power"]
        s.ensemble_weights_=None; s.active_calibration_method_=None
    def fit(s,X,y):
        s.classes_=np.unique(y); s.n_=len(s.classes_)
        s.ensemble_generator_=_Gen(s.n_estimators, s.norm_methods)
        s._X=np.asarray(X); s.__sklearn_is_fitted__=lambda: True
        if s.enable_nnls:
            w=np.random.default_rng(0).random(s.n_estimators); s.ensemble_weights_=w/w.sum()
        return s
    def _predict_proba_internal(s,X):
        rs=np.random.default_rng(abs(hash((s.feat_shuffle_method,s.class_shift,
            tuple(s.norm_methods),s.max_num_features,s.max_num_rows)))%2**31)
        return rs.normal(size=(s.n_estimators,np.asarray(X).shape[0],s.n_))
    @staticmethod
    def softmax(x,axis=-1,temperature=1.0):
        z=x/temperature; z=z-z.max(axis,keepdims=True); e=np.exp(z)
        return e/e.sum(axis,keepdims=True)
    def _process_logits(s,L):
        P=s.softmax(L,-1,s.softmax_temperature)
        if s.enable_nnls: p=np.tensordot(s.ensemble_weights_,P,axes=(0,0))
        elif s.average_logits: p=s.softmax(L.mean(0),-1,s.softmax_temperature)
        else: p=P.mean(0)
        return p
    def predict_proba(s,X): return s._process_logits(s._predict_proba_internal(X))
tf=types.ModuleType("tabfm"); tf.TabFMClassifier=StubClf
tf.tabfm_v1_0_0_pytorch=types.SimpleNamespace(load=lambda **k: "STUBMODEL")
sys.modules["tabfm"]=tf
t=types.ModuleType("torch"); t.bfloat16="bf16"; t.Tensor=type("T",(),{})
t.cuda=types.SimpleNamespace(is_available=lambda:False,get_device_name=lambda i:None,
                             max_memory_allocated=lambda:0)
sys.modules["torch"]=t

import experiments.tabfm_capture as TC

print("=== coalition table ===")
C=TC.TABFM_COALITIONS
chk(len(C)>=20, f"{len(C)} coalitions")
for n,want in (("A1","feature"),("A5_f50","context"),("A4_g50","feature")):
    chk(TC.axis_side(n)==want, f"{n:9s} axis_side={TC.axis_side(n)}")
chk(TC.is_expansion("A8both") and not TC.is_expansion("A4_g50"), "expansion flagged only for A8*")
chk(TC.destroys_information("A5_f50") and not TC.destroys_information("A1"), "A5 destroys, A1 preserves")
bad=[n for n,k in C.items() if k.get("average_logits") and k.get("enable_nnls")]
chk(not bad, f"no coalition violates the average_logits/enable_nnls exclusion ({bad})")

print("\n=== frac -> absolute caps ===")
r=TC.resolve_fracs(C["A4_g50"], n_rows=5000, n_features=40)
chk(r.get("max_num_features")==20 and "frac_features" not in r, f"A4_g50 on d=40 -> max_num_features={r.get('max_num_features')}")
r=TC.resolve_fracs(C["A5_f25"], n_rows=5000, n_features=40)
chk(r.get("max_num_rows")==1250, f"A5_f25 on n=5000 -> max_num_rows={r.get('max_num_rows')}")
r=TC.resolve_fracs(C["A4_g25"], n_rows=100, n_features=4)
chk(r["max_num_features"]==2, f"tiny d clamps to the floor ({r['max_num_features']})")

print("\n=== capture round-trips to predict_proba, bitwise ===")
Cap=TC.make_capturing_classifier()
rs=np.random.default_rng(0)
X=pd.DataFrame(rs.normal(size=(300,12))); y=rs.integers(0,3,300)
Xe=pd.DataFrame(rs.normal(size=(80,12)))
for name in ("shipped","A1","A3","A8both","plus_full","base"):
    kw=TC.resolve_fracs(C[name],300,12); ne=int(kw.pop("n_estimators",32))
    c=Cap(model="M",n_estimators=ne,**kw).fit(X,y)
    TC.verify_equivalence(c,Xe)                      # raises on mismatch
    M=c.predict_members(Xe)
    chk(M.shape==(ne,80,3), f"{name:10s} members {M.shape}, verify_equivalence OK")
chk(len(TC.member_metadata(c))>0, f"member_metadata returns {len(TC.member_metadata(c))} records")

print("\n=== run_cell writes the shared cache layout ===")
ds=types.ModuleType("experiments.datasets")
ds.load_splits=lambda n: iter([(X.iloc[:220],y[:220],X.iloc[220:],y[220:])])
ds.split_severity=lambda n,k:"ok"
ds.dataset_task=lambda n:"classification"
sys.modules["experiments.datasets"]=ds
import importlib, experiments.run_cell_tabfm as RC; importlib.reload(RC)
with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    for name in ("shipped","A5_f50","A8both","plus_full"):
        RC.run_cell("ds",name,32,root,checkpoint=None,device="cpu")
        m=json.loads((root/"ds"/name/"split0"/"meta.json").read_text())
        mem=np.load(root/"ds"/name/"split0"/"members.npy")
        want_info={"shipped":"preserving","A5_f50":"destroying","A8both":"expanding",
                   "plus_full":"expanding"}[name]
        chk(m["information"]==want_info and m["model"]=="TabFM" and m["space"]=="logits"
            and mem.shape[0]==32,
            f"{name:10s} M={mem.shape[0]} info={m['information']:11s} nnls={m['enable_nnls']}")
    RC.run_cell("ds","shipped",32,root,checkpoint=None,device="cpu")   # resume
    chk(True,"re-run skips the completed cell")
print("\n"+("ALL PASS" if ok else "FAILURES")); sys.exit(0 if ok else 1)
