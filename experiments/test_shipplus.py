"""'Shipped + one axis' ablations: views, A7 relabelling, wrapper wiring, coalition tables."""
import sys, types, numpy as np, pandas as pd
sys.path.insert(0, ".")

made = []                         # kwargs of every member estimator built
fits = []                         # (column order, labels) of every fit
class Stub:
    average_logits = True; softmax_temperature = 0.9
    def __init__(self, **kw): self.kw = kw; made.append(kw)
    def fit(self, X, y):
        self.X = X; self.classes_ = np.unique(y)
        self.majority = np.bincount(np.searchsorted(self.classes_, y)).argmax()   # in the FITTED label space
        fits.append((list(X.columns) if hasattr(X, "columns") else None, np.asarray(y).copy()))
        return self
    def predict_members(self, X):
        out = np.zeros((1, len(X), len(self.classes_)))
        out[..., self.majority] = 1.0          # always predicts the majority class it saw
        return out
ti = types.ModuleType("tabicl"); ti.TabICLClassifier = Stub; ti.TabICLRegressor = Stub
sys.modules["tabicl"] = ti
import experiments.capture as cap
cap.MemberCapturingTabICLClassifier = Stub; cap.MemberCapturingTabICLRegressor = Stub

from experiments import coalitions as C, tabfm_capture as T, tabfm_sets as S, views as V
from experiments.context import ContextEnsemble
from experiments.subsample import FeatureSubsampleEnsemble

ok = True
def chk(c, m):
    global ok; print(("  PASS  " if c else "  FAIL  ") + m); ok = ok and c

# ---- views ----------------------------------------------------------------
v0, v1 = V.shipped_view("classification", 0), V.shipped_view("classification", 1)
chk(v0["norm_methods"] == ["none"] and v1["norm_methods"] == ["power"], "norm alternates none/power across members")
chk(v0["feat_shuffle_method"] == "none" and v0["class_shuffle_method"] == "none",
    "feature/class order are NOT estimator kwargs (a one-member estimator ignores them)")
chk("class_shuffle_method" not in V.shipped_view("regression", 0), "regression view has no class axis (kwarg would raise upstream)")

# ---- A7 relabelling ---------------------------------------------------------
rng = np.random.default_rng(0)
Xtr = pd.DataFrame({"num": rng.normal(size=60),
                    "cat": pd.Categorical(rng.choice(list("abcde"), 60)),
                    "s": rng.choice(["x", "y", "z"], 60).astype(object)})
Xte = pd.DataFrame({"num": rng.normal(size=10), "cat": pd.Categorical(list("abcdeabcde")),
                    "s": ["x", "y", "z", "q", "x", "y", "z", "x", "y", "z"]})
Xtr0, Xte0 = Xtr.copy(), Xte.copy()
a_tr, a_te, cols = V.relabel_categories(Xtr, Xte, np.random.default_rng(1))
chk(sorted(cols) == ["cat", "s"], "categorical columns detected, numeric left alone")
chk(Xtr.equals(Xtr0) and Xte.equals(Xte0), "inputs not mutated")
chk(np.array_equal(a_tr["num"], Xtr["num"]), "numeric column untouched")
# bijection + train/test consistency
pair = set(zip(Xtr["cat"].astype(str), a_tr["cat"]))
chk(len({p[0] for p in pair}) == len({p[1] for p in pair}) == len(pair), "bijection on seen categories")
m = dict(zip(Xtr["cat"].astype(str), a_tr["cat"]))
chk(all(m[o] == n for o, n in zip(Xte["cat"].astype(str), a_te["cat"])), "same mapping applied to test")
chk(a_te["s"].iloc[3] != a_te["s"].iloc[3], "category unseen in training becomes missing (NaN)")
b_tr, _, _ = V.relabel_categories(Xtr, Xte, np.random.default_rng(2))
chk(not a_tr["cat"].equals(b_tr["cat"]), "different member seed -> different relabelling")
chk(V.categorical_columns(Xtr[["num"]]) == [] and V.categorical_columns(np.zeros((3, 3))) == [], "no categoricals -> empty")

# ---- ContextEnsemble wiring ------------------------------------------------
y = rng.integers(0, 3, 60)
made.clear()
ce = ContextEnsemble(n_estimators=4, seed=0, **C.CONTEXT_COALITIONS["S_ctl"])
out = ce.fit_predict_members(Xtr, y, Xte)
chk(out.shape == (4, 10, 3), "S_ctl: members shape")
chk([k["norm_methods"][0] for k in made] == ["none", "power", "none", "power"], "S_ctl: shipped-style views reach the estimator")
cols_seen = [tuple(f[0]) for f in fits]
chk(len(set(cols_seen)) > 1, f"S_ctl: members see different column orders ({len(set(cols_seen))} distinct of 4)")
chk(len({f[1].tobytes() for f in fits}) > 1, "S_ctl: members see different class codings")
chk(all(sorted(c) == sorted(Xtr.columns) for c in cols_seen), "S_ctl: column order changes, column set does not")
# majority-class stub: every member must still vote for the ORIGINAL majority class after restore
ymaj = np.where(rng.random(60) < 0.7, 2, rng.integers(0, 2, 60))
fits.clear(); made.clear()
out = ContextEnsemble(n_estimators=6, seed=3, **C.CONTEXT_COALITIONS["S_ctl"]).fit_predict_members(Xtr, ymaj, Xte)
chk(out.shape == (6, 10, 3) and (out.argmax(-1) == 2).all(), "S_ctl: logits restored to the ORIGINAL class order")
chk(len({f[1].tobytes() for f in fits}) > 1 and not all(np.array_equal(f[1], ymaj) for f in fits), "S_ctl: fitted labels really were permuted")
fits.clear(); made.clear()
FeatureSubsampleEnsemble(n_estimators=6, seed=3, **C.FEATURE_SUB_COALITIONS["S_A4_g90"]).fit_predict_members(
    Xn0 := pd.DataFrame(rng.normal(size=(80, 20)), columns=[f"c{i}" for i in range(20)]), ymaj_n := np.where(rng.random(80) < .7, 1, 0), Xn0.iloc[:10])
chk(len({tuple(f[0]) for f in fits}) > 1 and all(len(f[0]) == 18 for f in fits), "S_A4: subset size 18 of 20, orders differ")
fits.clear(); made.clear()
ContextEnsemble(n_estimators=4, seed=0, **C.CONTEXT_COALITIONS["S_ctl"]).fit_predict_members(Xtr, y, Xte)
chk(ce.timings["context_size_mean"] == 60, "S_ctl: full context (frac=1.0 keeps every row)")
made.clear(); fits.clear()
ContextEnsemble(n_estimators=3, seed=0, frac=.5, mode="random").fit_predict_members(Xtr, y, Xte)
chk(all(k["feat_shuffle_method"] == "none" and k["norm_methods"] == ["none"] for k in made), "plain A5 unchanged: every axis off")
chk(len({tuple(f[0]) for f in fits}) == 1, "plain A5 unchanged: column order untouched")
made.clear()
ce = ContextEnsemble(n_estimators=3, seed=0, **C.CONTEXT_COALITIONS["A7"]); ce.fit_predict_members(Xtr, y, Xte)
chk(ce.timings["relabel"] and ce.timings["n_categorical"] == 2 and ce.timings["view"] == "none", "A7: relabel recorded, no shipped view")

mvr = V.MemberView(5, None, np.random.default_rng(0))
chk(mvr.class_perm is None and np.array_equal(mvr.restore(np.ones((1, 2, 3))), np.ones((1, 2, 3))), "regression view: columns only")

# ---- FeatureSubsampleEnsemble wiring ------------------------------------------
Xn = pd.DataFrame(rng.normal(size=(80, 20)), columns=[f"c{i}" for i in range(20)]); yn = rng.integers(0, 2, 80)
made.clear()
fe = FeatureSubsampleEnsemble(n_estimators=4, seed=0, **C.FEATURE_SUB_COALITIONS["S_A4_g50"])
fe.fit_predict_members(Xn, yn, Xn.iloc[:10])
chk([k["norm_methods"][0] for k in made] == ["none", "power", "none", "power"], "S_A4: shipped-style views reach the estimator")
chk(fe.timings["subset_size_mean"] == 10, "S_A4_g50: half the columns")

# ---- coalition tables -----------------------------------------------------------
chk(set(C.SHIPPED_PLUS) == {"S_ctl", "S_A7", "A7", "S_A5_f50", "S_A5_f75", "S_A5_f90", "S_A4_g50", "S_A4_g75", "S_A4_g90"}, "TabICL S_* names")
chk(not C.destroys_information("S_ctl") and not C.destroys_information("S_A7") and not C.destroys_information("A7"), "S_ctl / A7 preserve information")
chk(C.destroys_information("S_A5_f75") and C.destroys_information("S_A4_g90") and C.destroys_information("A5lofo_M8"), "S_A4/S_A5/LOFO destroy information")
chk(not C.destroys_information("shipped") and not C.destroys_information("A1"), "native axes preserve")
chk(C.needs_categoricals("S_A7") and C.needs_categoricals("A7") and not C.needs_categoricals("S_ctl"), "needs_categoricals")
chk("S_A4_g50" not in C.tabicl_coalition_set("shipplus", "classification", 5, "small"), "S_A4 gated on >= 8 columns")
chk("S_ctl" in C.tabicl_coalition_set("shipplus", "regression", 5, "small"), "S_* apply to regression")
chk(set(C.SHIPPED_PLUS) <= set(C.TABICL_SETS["all"]), "S_* in the 'all' set")
chk(C.coalition_applies("S_A5_f50", "regression"), "coalition_applies for S_*")

chk(set(S.SETS["shipplus"]) <= set(T.TABFM_COALITIONS), "TabFM classification S_* defined")
chk(set(S.SETS["shipplus"]) <= set(T.TABFM_REG_COALITIONS), "TabFM regression S_* defined")
for c, kw in T.TABFM_COALITIONS.items():
    if c.startswith("S_"):
        base = T.TABFM_COALITIONS["shipped"]
        extra = {k: v for k, v in kw.items() if base.get(k) != v}
        chk(len(extra) in (1, 2) and all(k in base or k in extra for k in base), f"TabFM {c}: shipped + {sorted(extra)}")
chk(T.axis_side("S_A5_f50") == "context" and T.destroys_information("S_A4_g75") and T.is_expansion("S_A8both"), "TabFM axis helpers on S_*")
chk("S_A4_g50" not in S.coalition_set("shipplus", 5, "classification"), "TabFM S_A4 gated on >= 8 columns")
chk(set(S.SETS["shipplus"]) <= set(S.SETS["all"]), "TabFM S_* in 'all'")

print("\n" + ("ALL PASS" if ok else "FAILURES")); sys.exit(0 if ok else 1)
