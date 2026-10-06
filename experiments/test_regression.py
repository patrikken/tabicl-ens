"""Regression support, end to end, against stubbed upstream estimators.

Run from the repo root:  python experiments/test_regression.py

The stubs reproduce the STRUCTURE of the real upstream code that matters (read from
tabicl v2.2.0 regressor.py and tabfm classifier_and_regressor.py): per-view
predictions on a standardised target, an inverse transform per member, an
arithmetic-mean aggregator, and -- for TabFM -- an NNLS branch and a constructor
that REJECTS unknown kwargs, so a classification-only kwarg leaking into the
regression path fails loudly here instead of on the cluster.
What this cannot test is the real checkpoints; the first real cell asserts capture
== predict and says so in the log.
"""
import json, sys, tempfile, types
from collections import OrderedDict
from pathlib import Path

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import sklearn.ensemble, scipy.stats                       # before the torch stub
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

ok = True
def chk(c, m):
    global ok; print(("  PASS  " if c else "  FAIL  ") + m); ok = ok and bool(c)

# ------------------------------------------------------------------ stubs ----
class _TabICLBase(BaseEstimator):
    def __init__(self, **kw): self.kw = kw
class StubTabICLClassifier(_TabICLBase): pass

class _Gen:
    def __init__(s, k_per_norm, norms, d):
        s.norms, s.k, s.d = norms, k_per_norm, d
        s.feature_shuffles_ = OrderedDict((n, [np.arange(d)] * k_per_norm) for n in norms)
    def transform(s, X, mode="both"):
        T = X.shape[0]
        return OrderedDict((n, (np.zeros((s.k, T + 5, s.d)), np.zeros((s.k, 5)))) for n in s.norms)

class StubTabICLRegressor(_TabICLBase):
    """Mirrors TabICLRegressor.predict (v2.2.0) up to and including aggregation."""
    def __init__(self, n_estimators=8, norm_methods=None, feat_shuffle_method="latin",
                 checkpoint_version="tabicl-regressor-v2-20260212.ckpt", random_state=42,
                 device=None, kv_cache=False, batch_size=8, outlier_threshold=4.0):
        self.n_estimators, self.norm_methods = n_estimators, norm_methods
        self.feat_shuffle_method, self.checkpoint_version = feat_shuffle_method, checkpoint_version
        self.random_state, self.device, self.kv_cache = random_state, device, kv_cache
        self.batch_size, self.outlier_threshold = batch_size, outlier_threshold
    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float32)
        self.y_scaler_ = StandardScaler().fit(y.reshape(-1, 1))
        self.X_encoder_ = types.SimpleNamespace(transform=lambda X: np.asarray(X, dtype=float))
        norms = self.norm_methods or ["none", "power"]
        self.ensemble_generator_ = _Gen(max(1, self.n_estimators // len(norms)), norms,
                                        np.asarray(X).shape[1])
        self.model_kv_cache_ = None; self._seed = self.random_state
        return self
    def _batch_forward(self, Xs, ys, output_type="mean", alphas=None):
        rs = np.random.default_rng(abs(hash((Xs.shape, self.feat_shuffle_method,
                                             tuple(self.norm_methods or [])))) % 2**31)
        return rs.normal(size=(Xs.shape[0], Xs.shape[1] - 5)).astype(np.float32)
    def predict(self, X, output_type="mean", alphas=None):
        X = self.X_encoder_.transform(X)
        data = self.ensemble_generator_.transform(X, mode="both")
        res = [self._batch_forward(Xs, ys) for Xs, ys in data.values()]
        arr = np.concatenate(res, axis=0); E, T = arr.shape
        arr = self.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(E, T)
        return np.mean(arr, axis=0)

tabicl = types.ModuleType("tabicl")
tabicl.TabICLClassifier, tabicl.TabICLRegressor = StubTabICLClassifier, StubTabICLRegressor
sys.modules["tabicl"] = tabicl

torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(is_available=lambda: False, max_memory_allocated=lambda: 0)
torch.bfloat16 = "bf16"
sys.modules["torch"] = torch

# TabFM: signature copied from the real TabFMRegressor.__init__ -- NO class_shift,
# NO average_logits, NO calibration kwargs, and no **kw.
class StubTabFMRegressor(BaseEstimator):
    def __init__(self, model, n_estimators=32, norm_methods=None, feat_shuffle_method="random",
                 permute_categorical=False, outlier_threshold=4.0, max_num_features=500,
                 max_num_rows=None, use_amp=True, batch_size=1, random_state=42, verbose=False,
                 cat_encoder_mode="appearance", num_folds_for_cv=5, n_feature_crosses=0,
                 n_svd_features=0, total_svd_pool=None, enable_nnls=False, nnls_beta=0.75,
                 min_rows_for_single_val_split=2000, cache_context=False,
                 maybe_quantize_kv_cache=True, keep_cache_on_device=True):
        for k, v in locals().items():
            if k not in ("self", "__class__"): setattr(self, k, v)
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.y_scaler_ = StandardScaler().fit(y.reshape(-1, 1))
        norms = self.norm_methods or ["none", "power"]
        self.ensemble_generator_ = types.SimpleNamespace(
            ensemble_configs_={n: [(np.arange(3), i, None, None)
                                   for i in range(max(1, self.n_estimators // len(norms)))]
                               for n in norms})
        self.ensemble_weights_ = None
        if self.enable_nnls:
            w = np.random.default_rng(0).random(self.n_estimators); self.ensemble_weights_ = w / w.sum()
        self._E = self.n_estimators; return self
    def _inverse_transform_y(self, y):
        return self.y_scaler_.inverse_transform(np.asarray(y).reshape(-1, 1)).flatten()
    def _predict_internal(self, X):
        rs = np.random.default_rng(abs(hash((self.feat_shuffle_method, self.permute_categorical,
            self.max_num_features, self.max_num_rows, self.n_feature_crosses, self.n_svd_features))) % 2**31)
        return rs.normal(size=(self._E, np.asarray(X).shape[0]))
    def _combine_predictions(self, p):
        if self.enable_nnls:
            tp = np.zeros((p.shape[0], p.shape[1]))
            for i in range(p.shape[0]): tp[i] = self._inverse_transform_y(p[i])
            return np.dot(self.ensemble_weights_, tp)
        return self._inverse_transform_y(np.mean(p, axis=0))
    def predict(self, X): return self._combine_predictions(self._predict_internal(X))
tf = types.ModuleType("tabfm"); tf.TabFMRegressor = StubTabFMRegressor
tf.TabFMClassifier = type("C", (BaseEstimator,), {})
tf.tabfm_v1_0_0_pytorch = types.SimpleNamespace(load=lambda **k: ("STUBMODEL", k["model_type"]))
sys.modules["tabfm"] = tf

# ------------------------------------------------- 1. coalition tables --------
print("\n-- coalition tables --")
from experiments import coalitions as C
chk(C.native_coalitions("classification") ==
    ["base", "A1", "A2", "A3", "A1A2", "A1A3", "A2A3", "A1A2A3", "shipped"], "classification keeps all 9 native coalitions")
chk(C.native_coalitions("regression") == ["base", "A1", "A3", "A1A3", "shipped"], "regression drops A2 and every mix containing it")
chk("class_shuffle_method" not in C.tabicl_kwargs("shipped", "regression"), "regression kwargs carry no class_shuffle_method")
chk("class_shuffle_method" in C.tabicl_kwargs("shipped", "classification"), "classification kwargs unchanged")
chk(not C.coalition_applies("A2", "regression") and C.coalition_applies("A4_g50", "regression")
    and C.coalition_applies("A5_f50", "regression"), "A2 is classification-only; wrappers apply to both")
chk(not C.coalition_applies("A6_control", "classification"), "A6_control is a gate, not a campaign cell")
s = C.tabicl_coalition_set("all", "regression", 30, "small")
chk(all(C.coalition_applies(c, "regression") for c in s) and "A5lofo_M64" in s, f"regression/small set sane ({len(s)} coalitions, lofo present)")
chk("A5lofo_M8" not in C.tabicl_coalition_set("all", "regression", 30, "large"), "LOFO gated off the large bucket")
chk(not any(c.startswith("A4") for c in C.tabicl_coalition_set("all", "regression", 5, "small")), "A4 gated at d<8")
chk("A4lofo_M16" not in C.tabicl_coalition_set("all", "regression", 20, "small"), "A4 column-LOFO gated when M > d/2")

from experiments import tabfm_capture as T, tabfm_sets as S
bad = {"class_shift", "average_logits", "binary_calibration_method", "multiclass_calibration_method"}
chk(not any(bad & set(kw) for kw in T.TABFM_REG_COALITIONS.values()), "no classification-only kwarg in the TabFM regression table")
import inspect
allowed = set(inspect.signature(StubTabFMRegressor.__init__).parameters) | set(T.FRAC_KEYS)
chk(all(set(kw) <= allowed for kw in T.TABFM_REG_COALITIONS.values()), "every regression kwarg is a real TabFMRegressor parameter")
for task in ("classification", "regression"):
    for name in S.SETS:
        cs = S.coalition_set(name, 30, task)
        chk(all(c in T.tabfm_coalitions(task) for c in cs), f"set {name!r} resolves for {task} ({len(cs)} coalitions)")
chk("A2" not in S.coalition_set("axes", 30, "regression") and "plus_nocal" not in S.coalition_set("plus", 30, "regression"),
    "TabFM regression sets drop A2 and the calibration ablation")
chk(T.is_expansion("A8both", "regression") and T.destroys_information("A5_f50", "regression")
    and T.axis_side("A5_f50", "regression") == "context", "TabFM helpers are task-aware")

# ------------------------------------------------------- 2. strata ------------
print("\n-- strata / stratified draws for a continuous target --")
from experiments.context import strata, subsample_indices, lofo_partition
rs = np.random.default_rng(0); y = np.exp(rs.normal(size=1000))                  # heavy right tail
st = strata(y, "regression")
chk(2 <= len(np.unique(st)) <= 10 and np.bincount(st).min() > 50, f"{len(np.unique(st))} quantile bins, roughly equal")
chk(np.array_equal(strata(np.array([1, 2, 2]), "classification"), np.array([1, 2, 2])), "classification strata are the classes")
idx = subsample_indices(st, 0.25, np.random.default_rng(1), "random")
chk(abs(len(idx) - 250) <= 15 and y[idx].max() > np.quantile(y, 0.9), "a 25% draw still reaches the upper tail")
mem, pinned = lofo_partition(st, 8, np.random.default_rng(2))
chk(len(mem) == 8 and all(abs(len(m) - 875) < 40 for m in mem), "LOFO folds are balanced over target quantiles")

# ------------------------------------- 3. TabICL regression capture ----------
print("\n-- TabICLv2 regression capture --")
from experiments import capture as K
Xtr = pd.DataFrame(rs.normal(size=(300, 12)), columns=[f"c{i}" for i in range(12)])
ytr = 1000 + 50 * rs.normal(size=300); Xte = pd.DataFrame(rs.normal(size=(40, 12)), columns=Xtr.columns)
reg = K.MemberCapturingTabICLRegressor(n_estimators=8, norm_methods=["none", "power"]).fit(Xtr, ytr)
mem = reg.predict_members(Xte)
chk(mem.shape == (8, 40), f"members shape {mem.shape} == (8, 40)")
chk(abs(mem.mean() - 1000) < 100, "members are on the ORIGINAL target scale (~1000), not standardised")
K.verify_equivalence_regression(reg, Xte); chk(True, "captured-then-averaged == predict (bitwise)")
broken = K.MemberCapturingTabICLRegressor(n_estimators=8).fit(Xtr, ytr)
broken.predict = lambda X, **k: np.zeros(len(X)) + 1.0
try: K.verify_equivalence_regression(broken, Xte); chk(False, "verify must reject a diverging aggregator")
except AssertionError: chk(True, "verify rejects a diverging aggregator")
chk(K.default_checkpoint("regression").startswith("tabicl-regressor") and
    K.default_checkpoint("classification").startswith("tabicl-classifier"), "default checkpoints per task")

# ------------------------------------- 4. A4 / A5 wrappers for regression ----
print("\n-- A4 / A5 wrappers on a regression target --")
from experiments.subsample import FeatureSubsampleEnsemble
from experiments.context import ContextEnsemble
for name, kw in [("A4_g50", dict(frac=.5, mode="random")), ("A4imp", dict(frac=.5, mode="importance")),
                 ("A4lofo_M4", dict(mode="lofo", n_folds=4))]:
    fe = FeatureSubsampleEnsemble(n_estimators=8, seed=0, task="regression", **kw)
    out = fe.fit_predict_members(Xtr, ytr, Xte); M = kw.get("n_folds") or 8
    chk(out.shape == (M, 40) and np.isfinite(out).all(), f"{name:10s} members {out.shape}")
    chk(fe.average_logits is False and "regressor" in fe.checkpoint, f"{name:10s} regression metadata + regressor checkpoint")
    if kw.get("mode") == "importance": chk(fe.timings["importance_used"], "importance via ExtraTreesRegressor ran")
for name, kw in [("A5_f50", dict(frac=.5, mode="random")), ("A5bal", dict(frac=.5, mode="balanced")),
                 ("A5lofo_M4", dict(mode="lofo", n_folds=4))]:
    ce = ContextEnsemble(n_estimators=8, seed=0, task="regression", **kw)
    out = ce.fit_predict_members(Xtr, ytr, Xte); M = kw.get("n_folds") or 8
    chk(out.shape == (M, 40) and np.isfinite(out).all(), f"{name:10s} members {out.shape}")
cls = ContextEnsemble(n_estimators=4, seed=0, frac=.5)
chk(cls.task == "classification" and "classifier" in cls.checkpoint, "classification default untouched")

# ------------------------------------- 5. run_cell, regression, end to end ---
print("\n-- run_cell (TabICLv2) on a regression dataset --")
from experiments import run_cell as R
tmp = Path(tempfile.mkdtemp())
def fake_splits(name):
    for _ in range(2): yield Xtr, ytr, Xte, 1000 + 50 * rs.normal(size=40)
R.load_splits = fake_splits
R.dataset_task = lambda n: "regression" if n.startswith("reg") else "classification"
R.run_cell("reg_toy", "shipped", 8, tmp, verify=False)
m = json.loads((tmp / "reg_toy/shipped/split0/meta.json").read_text())
arr = np.load(tmp / "reg_toy/shipped/split0/members.npy")
chk(arr.dtype == np.float32 and arr.shape == (8, 40), f"regression members cached float32 {arr.shape}")
chk(m["task"] == "regression" and m["space"] == "predictions" and m["n_classes"] is None
    and m["softmax_temperature"] is None and m["y_train_std"] > 0, "meta: task/space/n_classes/temperature/y_std")
chk("regressor" in m["checkpoint"] and "class_shuffle_method" not in m["axis_kwargs"], "regressor checkpoint, no class kwarg")
R.run_cell("reg_toy", "A2", 8, tmp)
chk(not (tmp / "reg_toy/A2").exists(), "A2 on a regression dataset is a logged no-op, not a crash")
R.run_cell("reg_toy", "A4_g50", 8, tmp)
m4 = json.loads((tmp / "reg_toy/A4_g50/split1/meta.json").read_text()); a4 = np.load(tmp / "reg_toy/A4_g50/split1/members.npy")
chk(a4.shape[1] == 40 and a4.ndim == 2 and m4["information"] == "destroying", f"A4 wrapper on regression -> {a4.shape}, information=destroying")
R.run_cell("reg_toy", "shipped", 8, tmp)                                # idempotent
chk(True, "second call skips completed splits")

# ------------------------------------- 6. TabFM regression runner ------------
print("\n-- run_cell_tabfm on a regression dataset --")
from experiments import run_cell_tabfm as F
F.load_splits = fake_splits
F.dataset_task = R.dataset_task
tf_out = Path(tempfile.mkdtemp())
for coal in ("base", "shipped", "A8both", "plus_full", "A4_g50", "A5_f50", "A7"):
    F.run_cell("reg_toy", coal, 8, tf_out, "ckpt", seed=0, device="cpu")
    mm = json.loads((tf_out / f"reg_toy/{coal}/split0/meta.json").read_text())
    a = np.load(tf_out / f"reg_toy/{coal}/split0/members.npy")
    chk(a.dtype == np.float32 and a.ndim == 2, f"{coal:9s} members {a.shape} float32; information={mm['information']}")
mp = json.loads((tf_out / "reg_toy/plus_full/split0/meta.json").read_text())
chk(mp["enable_nnls"] and abs(sum(mp["nnls_weights"]) - 1) < 1e-9, "plus_full records NNLS weights (sum 1)")
chk(json.loads((tf_out / "reg_toy/base/split0/meta.json").read_text())["m_realised"] == 1, "base is a single member")
chk(mm["task"] == "regression" and mm["n_classes"] is None and mm["space"] == "predictions", "TabFM regression meta")
F.run_cell("reg_toy", "A2", 8, tf_out, "ckpt")
chk(not (tf_out / "reg_toy/A2").exists(), "A2 on TabFM regression is a no-op, not a crash")
Cap = T.make_capturing_regressor()
r = Cap(model="m", n_estimators=8).fit(Xtr, ytr)
T.verify_equivalence_regression(r, Xte); chk(True, "TabFM capture == predict (rtol 1e-6, no NNLS)")
r2 = Cap(model="m", n_estimators=8, enable_nnls=True).fit(Xtr, ytr)
T.verify_equivalence_regression(r2, Xte); chk(True, "TabFM capture == predict (NNLS weights applied)")

# ------------------------------------- 7. datasets + grid --------------------
print("\n-- manifest filtering, grid and skip-complete --")
import os
man = {}
def add(n, pt, nfeat, nsp, bucket): man[n] = dict(problem_type=pt, n_features=nfeat, n_splits=nsp, bucket=bucket, splits=[dict(k=i, severity="ok") for i in range(nsp)])
add("cls_a", "binary", 20, 3, "small"); add("cls_b", "multiclass", 6, 3, "small"); add("cls_big", "binary", 40, 3, "large")
add("reg_a", "regression", 12, 3, "small"); add("reg_b", "regression", 5, 3, "medium")
td = Path(tempfile.mkdtemp()); (td / "manifest.json").write_text(json.dumps(man))
os.environ["TABARENA_DIR"] = str(td)
import importlib, experiments.datasets as D; importlib.reload(D)
import experiments.grid as G; importlib.reload(G)
chk(D.tabarena_datasets() == ["cls_a", "cls_b", "cls_big"], "default tabarena_datasets() stays classification-only (old scripts unaffected)")
chk(D.tabarena_datasets("regression") == ["reg_a", "reg_b"] and len(D.tabarena_datasets(None)) == 5, "regression / all filters")
chk(D.dataset_task("reg_a") == "regression" and D.dataset_task("cls_a") == "classification", "dataset_task from manifest")
for model in ("tabicl", "tabfm"):
    for task in ("classification", "regression"):
        ps = G.pairs(model, task, "all")
        chk(len({p[0] for p in ps}) == len(D.tabarena_datasets(task)) and len(ps) > 0, f"{model}/{task}: {len(ps)} pairs over {len({p[0] for p in ps})} datasets")
        chk(all((p[1] in C.all_coalitions() if model == "tabicl" else p[1] in T.tabfm_coalitions(task)) for p in ps), f"{model}/{task}: every coalition exists for the task")
chk(not any(c.startswith("A4") for d, c, *_ in G.pairs("tabicl", "classification", "all") if d == "cls_b"), "A4 skipped on the 6-column dataset")
chk(not any(c in C.LOFO_COALITIONS for d, c, *_ in G.pairs("tabicl", "classification", "large") if d == "cls_big"), "LOFO not scheduled on the large bucket")
cache = Path(tempfile.mkdtemp())
for k in range(3): (cache / "cls_a/base" / f"split{k}").mkdir(parents=True); (cache / "cls_a/base" / f"split{k}" / "meta.json").write_text("{}")
(cache / "cls_a/A1/split0").mkdir(parents=True); (cache / "cls_a/A1/split0/meta.json").write_text("{}")
full = {(d, c) for d, c, *_ in G.pairs("tabicl", "classification", "small")}
skip = {(d, c) for d, c, *_ in G.pairs("tabicl", "classification", "small", skip_complete=True, cache=cache)}
chk(("cls_a", "base") in full and ("cls_a", "base") not in skip, "a fully cached pair is skipped")
chk(("cls_a", "A1") in skip, "a half-done pair is kept (the worker resumes at the first missing split)")
(cache / "tabfm/reg_a/shipped").mkdir(parents=True)
for k in range(3): (cache / "tabfm/reg_a/shipped" / f"split{k}").mkdir(); (cache / "tabfm/reg_a/shipped" / f"split{k}" / "meta.json").write_text("{}")
sk = {(d, c) for d, c, *_ in G.pairs("tabfm", "regression", "small", skip_complete=True, cache=cache)}
chk(("reg_a", "shipped") not in sk and ("reg_a", "A1") in sk, "TabFM caches are read from <cache>/tabfm")

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
