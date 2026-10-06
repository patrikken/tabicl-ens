"""score_cells on synthetic cached cells (no torch, no tabarena)."""
import json, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from experiments import score_cells as sc
from experiments.elo import to_metric_error

n_ok = 0
def check(c, msg):
    global n_ok
    assert c, msg
    n_ok += 1

def write(root, ds, coal, split, mem, y, **meta):
    d = root / ds / coal / f"split{split}"
    d.mkdir(parents=True)
    np.save(d / "members.npy", mem); np.save(d / "y_test.npy", y)
    (d / "meta.json").write_text(json.dumps(dict(dataset=ds, coalition=coal, split=split,
        n_estimators_requested=mem.shape[0], n_train=50, n_test=len(y), n_features=3, **meta)))

rng = np.random.default_rng(0)
root = Path(tempfile.mkdtemp())
# regression: 'base' noisy members, 'A1' members closer to y
y = rng.normal(size=200)
for coal, noise in (("base", 1.0), ("A1", 0.3)):
    mem = y[None] + rng.normal(scale=noise, size=(32, 200))
    write(root, "reg", coal, 0, mem.astype(np.float32), y, task="regression", n_classes=None, y_train_std=1.0)
# classification: logits aligned with labels, huge TabFM-scale logits
yc = rng.integers(0, 3, 300)
lg = np.eye(3)[yc][None] * 900.0 + rng.normal(size=(8, 300, 3))
for coal in ("base", "A1"):
    write(root, "clf", coal, 0, lg.astype(np.float32), yc, task="multiclass", n_classes=3,
          softmax_temperature=0.9)

for model in ("tabicl", "tabfm"):
    rows = [sc.score_cell(m.parent, model) for m in sorted(root.rglob("meta.json"))]
    df = sc.add_gain(pd.DataFrame([r[0] for r in rows]))
    sw = sc.add_gain(pd.DataFrame([x for r in rows for x in r[1]]), extra=("M",))
    r = df[(df.dataset == "reg")].set_index("coalition")
    check((df[df.dataset == "reg"].task == "regression").all(), "task detected")
    check(r.loc["A1", "score"] > r.loc["base", "score"], "closer members -> higher (-RMSE)")
    check(np.isclose(r.loc["A1", "rel_gain"], r.loc["A1", "gain"] / abs(r.loc["base", "score"])), "rel_gain")
    check(r.loc["base", "gain"] == 0, "base gain 0")
    # averaging 32 members beats a single member (noise averages out)
    check(r.loc["base", "score"] > r.loc["base", "score_single"], "ensembling helps")
    # M sweep monotone-ish and ends at full-M score
    s = sw[(sw.dataset == "reg") & (sw.coalition == "base")].sort_values("M")
    check(list(s.M) == [1, 2, 4, 8, 16, 32], "budgets")
    check(np.isclose(s.score.iloc[-1], r.loc["base", "score"]), "M=32 equals full")
    check(s.score.iloc[-1] > s.score.iloc[0], "sweep improves")
    # sweep gain is against base at the SAME M
    a1 = sw[(sw.dataset == "reg") & (sw.coalition == "A1") & (sw.M == 4)].iloc[0]
    b4 = sw[(sw.dataset == "reg") & (sw.coalition == "base") & (sw.M == 4)].iloc[0]
    check(np.isclose(a1.gain, a1.score - b4.score), "gain at same M")
    # classification with ~1e3 logits: TabFM never clipped
    c = df[df.dataset == "clf"].iloc[0]
    check(c.task == "multiclass" and np.isfinite(c.score), "clf scored")
    check(c.score > -1e-3, "separable logits -> ~0 log loss")
    # error conversion
    e = to_metric_error(df.score.values, df.task.values)
    check((e >= 0).all(), "metric_error nonneg")
    check(np.isclose(e[df.task == "regression"], -df.score[df.task == "regression"]).all(), "rmse = -score")

# clipping contract
m = np.full((2, 4, 3), 1e3, dtype=np.float32)
check(sc._repair(m, 80.0)[0].max() == 80.0, "tabicl clips")
check(sc._repair(m, None)[0].max() == 1e3, "tabfm does not")
m[0, 0, 0] = np.inf
r, h = sc._repair(m, None)
check(np.isfinite(r).all() and h["n_posinf"] == 1, "inf repaired + counted")
try:
    to_metric_error(np.array([0.1]), np.array(["ranking"])); check(False, "unknown task")
except ValueError:
    check(True, "unknown task rejected")
print(f"{n_ok} checks passed")

# ---- ablation references ------------------------------------------------------
from experiments.score_cells import reference_of, add_ablation_gain
check(reference_of("S_A4_g50", "tabicl") == "S_ctl" and reference_of("S_A4_g50", "tabfm") == "shipped", "ref by model")
check(reference_of("S_ctl", "tabicl") == "shipped" and reference_of("A1", "tabicl") is None, "S_ctl vs shipped; native has none")
d = pd.DataFrame([dict(dataset="d", split=0, coalition=c, score=s, task="binary")
                  for c, s in [("base", .5), ("shipped", .6), ("S_ctl", .62), ("S_A4_g50", .65), ("A7", .55)]])
g = add_ablation_gain(d, "tabicl").set_index("coalition")
check(np.isclose(g.loc["S_A4_g50", "gain_ref"], .03) and np.isclose(g.loc["S_ctl", "gain_ref"], .02), "gain_ref values")
check(np.isnan(g.loc["A7", "gain_ref"]) and np.isnan(g.loc["base", "gain_ref"]), "no ref -> NaN")
g2 = add_ablation_gain(d, "tabfm").set_index("coalition")
check(np.isclose(g2.loc["S_A4_g50", "gain_ref"], .05), "tabfm S_* vs shipped")
print(f"{n_ok} checks passed (with ablations)")
