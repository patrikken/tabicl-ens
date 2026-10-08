"""Tables and figures from ``score_cells`` output (TabICLv2 and TabFM).

    python -m experiments.report RESULTS_DIR OUT_DIR
        RESULTS_DIR/{tabicl,tabfm}/{scores,sweep}.csv   (score_cells output)

Conventions
-----------
* Effect size: AUC gain (binary), log-likelihood gain (multiclass), relative RMSE
  reduction (regression). Dataset-level means, 95% bootstrap CI over datasets.
* Elo: LOCAL Elo (TabArena's Bradley-Terry protocol on a pool of our own
  methods), anchored at 1000 on the pool's reference method, 95% CI from a
  bootstrap over datasets. Each panel gets its own pool restricted to the
  (dataset, split) units every method in it ran. Local Elo exaggerates the
  magnitude of differences between near-copies; read it as a RANKING with a CI,
  and read the effect size for magnitude.
* TabFM+ cells (``plus_*``) are excluded unless ``recipe_ok`` says the recipe
  was replayed (see ``score_cells.full_prediction``).
* No titles and no text inside axes: identity is carried by legends, the rest
  by captions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elo as E                                                    # noqa: E402

TASKS = ("binary", "multiclass", "regression")
NATIVE = ["A1", "A2", "A3", "A1A2", "A1A3", "A2A3", "A1A2A3", "shipped"]
MODELS = ("tabicl", "tabfm")
MODEL_NAME = {"tabicl": "TabICLv2", "tabfm": "TabFM"}
EFFECT = {"binary": "ROC-AUC", "multiclass": "log-likelihood", "regression": "relative RMSE reduction"}
BOOT = 2000


# ------------------------------------------------------------------ loading --
def load(results: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    out = {}
    for m in MODELS:
        f = results / m / "scores.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        w = pd.read_csv(results / m / "sweep.csv")
        if "recipe_ok" not in s:                    # older score_cells: plus_* were not replayed
            s["recipe_ok"] = ~s.coalition.str.startswith("plus_")
        bad = set(s.loc[~s.recipe_ok.astype(bool), "coalition"])
        s = s[s.recipe_ok.astype(bool)]
        w = w[~w.coalition.isin(bad)]
        s["ef"] = np.where(s.task == "regression", s.rel_gain, s.gain)
        ref = s.score - s.gain_ref
        s["ef_ref"] = np.where(s.task == "regression", s.gain_ref / ref.abs(), s.gain_ref)
        out[m] = (s.reset_index(drop=True), w.reset_index(drop=True))
    return out


# --------------------------------------------------------------- statistics --
def boot_ci(x: np.ndarray, rng: np.random.Generator, n: int = BOOT) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan,) * 3
    idx = rng.integers(0, len(x), (n, len(x)))
    bm = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def effect_table(scores: pd.DataFrame, col: str = "ef", seed: int = 0) -> pd.DataFrame:
    """Dataset-level mean effect with CI, per task/coalition."""
    rng = np.random.default_rng(seed)
    ds = scores.groupby(["task", "coalition", "dataset"])[col].mean().reset_index()
    rows = []
    for (t, c), g in ds.groupby(["task", "coalition"]):
        mu, lo, hi = boot_ci(g[col].to_numpy(), rng)
        rows.append(dict(task=t, coalition=c, n_datasets=len(g), effect=mu, lo=lo, hi=hi,
                         frac_improved=float((g[col] > 0).mean()),
                         M=float(scores[(scores.task == t) & (scores.coalition == c)].M.median())))
    return pd.DataFrame(rows)


def _units(df):  # noqa: ANN001
    return df[["dataset", "split"]].drop_duplicates()


def elo_pool(rows: pd.DataFrame, method_col: str = "coalition") -> pd.DataFrame:
    p = pd.DataFrame({"method": rows[method_col].to_numpy(), "dataset": rows.dataset.to_numpy(),
                      "fold": rows.split.to_numpy(),
                      "metric_error": E.to_metric_error(rows.score.to_numpy(), rows.task.to_numpy()),
                      "metric": "m", "problem_type": rows.task.to_numpy()})
    return p


def cover_filter(rows: pd.DataFrame, method_col: str, min_cover: float = 0.9) -> pd.DataFrame:
    """Keep methods present on >= ``min_cover`` of the units the best-covered method
    has. Without this a method that exists on only some datasets (A2 at M=4 needs
    >= 4 classes) would force the common subgrid down to those datasets alone; the
    90% default tolerates a dataset or two missing from one coalition."""
    u = rows.groupby(method_col).apply(lambda g: len(_units(g)), include_groups=False)
    keep = u[u >= min_cover * u.max()].index
    return rows[rows[method_col].isin(keep)]


def elo_panel(rows: pd.DataFrame, anchor: str, method_col: str = "coalition",
              boot: int = 100, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Elo relative to ``anchor`` (=0) with bootstrap CI. Returns (table, info)."""
    rows = rows.drop_duplicates(subset=[method_col, "dataset", "split"])
    pool = E.complete_subgrid(elo_pool(rows, method_col), verbose=False)
    if anchor not in set(pool.method) or pool.method.nunique() < 2:
        return pd.DataFrame(columns=["elo", "lo", "hi"]), dict(n_datasets=0, n_units=0)
    r = E.elo_ratings(pool, anchor=anchor, bootstrap_rounds=boot, seed=seed).ratings
    r = r.reset_index() if "method" not in r.columns else r
    r = r.set_index("method")
    t = pd.DataFrame({"elo": r["elo"] - E.INIT_RATING, "lo": r["elo_lo"] - E.INIT_RATING,
                      "hi": r["elo_hi"] - E.INIT_RATING})
    return t, dict(n_datasets=int(pool.dataset.nunique()), n_units=int(len(pool[["dataset", "fold"]].drop_duplicates())))


# ------------------------------------------------------------------- tables --
def build_tables(data: dict, boot: int) -> dict[str, pd.DataFrame]:
    out = {}
    nat, wrap4, wrap5, ship = [], [], [], []
    sweep_rows = []
    for m, (s, w) in data.items():
        eff = effect_table(s)
        eff["model"] = m
        out.setdefault("effect_all", []).append(eff)
        eff_ref = effect_table(s[s.ef_ref.notna()], col="ef_ref")
        eff_ref["model"] = m
        out.setdefault("effect_vs_ref", []).append(eff_ref)
        for t in TASKS:
            st = s[s.task == t]
            if st.empty:
                continue
            # native coalitions, base as anchor
            cs = ["base"] + [c for c in NATIVE if c in set(st.coalition)]
            tb, info = elo_panel(st[st.coalition.isin(cs)], "base", boot=boot)
            tb["model"], tb["task"], tb["panel"] = m, t, "native"
            tb["n_datasets"] = info["n_datasets"]
            nat.append(tb.reset_index())
            for fam, lst in (("A4", [c for c in st.coalition.unique() if c.startswith("A4_g")]),
                             ("A5", [c for c in st.coalition.unique() if c.startswith("A5_f")])):
                if not lst:
                    continue
                tb, info = elo_panel(st[st.coalition.isin(["base", "shipped"] + lst)], "base", boot=boot)
                tb["model"], tb["task"], tb["panel"], tb["n_datasets"] = m, t, fam, info["n_datasets"]
                (wrap4 if fam == "A4" else wrap5).append(tb.reset_index())
            # shipped + one axis, against its reference. One pool PER AXIS FAMILY: A4 needs
            # >= 8 columns and A7 needs categorical columns, so a single pool would be
            # restricted to the datasets where every axis exists.
            sc_ = [c for c in st.coalition.unique() if c.startswith("S_") and c != "S_ctl"]
            if sc_:
                ref = "S_ctl" if m == "tabicl" else "shipped"
                refrows = st[st.coalition == ref]
                for fam, pre in (("A4", "S_A4"), ("A5", "S_A5"), ("A7", "S_A7"), ("A8", "S_A8")):
                    names = [c for c in sc_ if c.startswith(pre)]
                    if not names:
                        continue
                    tb, info = elo_panel(pd.concat([st[st.coalition.isin(names)], refrows]), ref, boot=boot)
                    tb = tb.drop(index=ref, errors="ignore")
                    tb["model"], tb["task"], tb["panel"], tb["n_datasets"] = m, t, fam, info["n_datasets"]
                    ship.append(tb.reset_index())
                if m == "tabicl":
                    # native shipped runs 32 members, the S_* family 16: compare at 16
                    wsh = w[(w.task == t) & (w.coalition == "shipped") & (w.M <= 16)]
                    wsh = wsh.sort_values("M").groupby(["dataset", "split"], as_index=False).last()
                    tb, info = elo_panel(pd.concat([wsh, refrows[wsh.columns.intersection(refrows.columns)]]), ref, boot=boot)
                    tb = tb.drop(index=ref, errors="ignore")
                    tb["model"], tb["task"], tb["panel"], tb["n_datasets"] = m, t, "control", info["n_datasets"]
                    ship.append(tb.reset_index())
            # member-budget sweep
            wt = w[(w.task == t) & w.coalition.isin(["base"] + NATIVE)].copy()
            wt["method"] = np.where(wt.coalition == "base", "base", wt.coalition + "@" + wt.M.astype(str))
            wt = cover_filter(wt, "method")
            tb, info = elo_panel(wt, "base", method_col="method", boot=max(50, boot // 2))
            if len(tb):
                tb = tb.reset_index()
                tb["coalition"] = tb["method"].str.split("@").str[0]
                tb["M"] = tb["method"].str.split("@").str[1].fillna("1").astype(int)
                tb["model"], tb["task"], tb["n_datasets"] = m, t, info["n_datasets"]
                sweep_rows.append(tb)
    res = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    for name, lst in (("elo_native", nat), ("elo_A4", wrap4), ("elo_A5", wrap5),
                      ("elo_shipplus", ship), ("elo_vs_M", sweep_rows)):
        res[name] = pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()
    return res


# ------------------------------------------------------------------ figures --
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#dcdcd8"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False, "pdf.fonttype": 42})
    return plt


COL = {"tabicl": "#2a78d6", "tabfm": "#eb6834", "A4": "#e5a52c", "A5": "#1baf7a"}
INK2 = "#52514e"


def _despine(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", visible=False)


def _save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")


def fig_elo_native(T, out, plt):
    """Elo gain over a single pass: native coalitions, TabICLv2 vs TabFM."""
    from matplotlib.patches import Patch
    d = T["elo_native"]
    tasks = [t for t in TASKS if (d.task == t).any()]
    fig, axs = plt.subplots(1, len(tasks), figsize=(3.5 * len(tasks), 2.9), squeeze=False)
    for ax, t in zip(axs[0], tasks):
        cs = [c for c in NATIVE if ((d.task == t) & (d.method == c)).any()]
        w = 0.38
        for j, m in enumerate(MODELS):
            sub = d[(d.task == t) & (d.model == m)].set_index("method")
            y = np.array([sub.elo.get(c, np.nan) for c in cs])
            lo = np.array([sub.lo.get(c, np.nan) for c in cs])
            hi = np.array([sub.hi.get(c, np.nan) for c in cs])
            x = np.arange(len(cs)) + (j - 0.5) * w
            ax.bar(x, y, width=w * 0.94, color=COL[m], zorder=2)
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=INK2, lw=0.9, capsize=0, zorder=3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([c.replace("A1A2A3", "A1+A2+A3").replace("A1A2", "A1+A2").replace("A1A3", "A1+A3")
                            .replace("A2A3", "A2+A3") for c in cs], rotation=40, ha="right")
        ax.set_ylabel("Elo gain over single pass")
        _despine(ax)
    axs[0][0].legend(handles=[Patch(color=COL[m], label=MODEL_NAME[m]) for m in MODELS],
                     loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=7.5, borderaxespad=0)
    fig.tight_layout(w_pad=1.4)
    _save(fig, out, "fig_elo_native")
    plt.close(fig)


def fig_effect_native(effects, out, plt):
    """Effect size (dataset means with CI) for native coalitions."""
    from matplotlib.lines import Line2D
    d = effects[effects.coalition.isin(NATIVE)]
    tasks = [t for t in TASKS if (d.task == t).any()]
    fig, axs = plt.subplots(1, len(tasks), figsize=(3.5 * len(tasks), 3.0), squeeze=False)
    for ax, t in zip(axs[0], tasks):
        cs = [c for c in NATIVE if ((d.task == t) & (d.coalition == c)).any()]
        ys = np.arange(len(cs))[::-1]
        for j, m in enumerate(MODELS):
            sub = d[(d.task == t) & (d.model == m)].set_index("coalition")
            for y, c in zip(ys, cs):
                if c not in sub.index:
                    continue
                r = sub.loc[c]
                yy = y + (0.14 if j == 0 else -0.14)
                ax.plot([r.lo, r.hi], [yy, yy], color=COL[m], lw=1.6, solid_capstyle="butt", zorder=3)
                ax.scatter([r.effect], [yy], s=22, color=COL[m], edgecolor="white", lw=0.7, zorder=4)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([c.replace("A1A2A3", "A1+A2+A3").replace("A1A2", "A1+A2").replace("A1A3", "A1+A3")
                            .replace("A2A3", "A2+A3") for c in cs])
        ax.set_xlabel(f"{EFFECT[t]} gain over single pass" if t != "regression" else "relative RMSE reduction over single pass")
        _despine(ax)
        ax.grid(axis="y", visible=False)
    axs[0][0].legend(handles=[Line2D([], [], marker="o", color=COL[m], lw=1.6, ms=4.5, label=MODEL_NAME[m])
                              for m in MODELS], loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
                     fontsize=7.5, borderaxespad=0)
    fig.tight_layout(w_pad=1.4)
    _save(fig, out, "fig_effect_native")
    plt.close(fig)



def fig_wrappers(T, out, plt):
    """A4 (column) and A5 (row) withholding: Elo gain over a single pass, by fraction kept."""
    from matplotlib.patches import Patch
    tasks = [t for t in TASKS if (T["elo_A5"].task == t).any()]
    fig, axs = plt.subplots(2, len(tasks), figsize=(3.5 * len(tasks), 4.6), squeeze=False)
    for i, m in enumerate(MODELS):
        for k, t in enumerate(tasks):
            ax = axs[i][k]
            fr = [25, 50, 75, 90]
            for j, fam in enumerate(("A4", "A5")):
                d = T[f"elo_{fam}"]
                d = d[(d.model == m) & (d.task == t)].set_index("method")
                pre = "A4_g" if fam == "A4" else "A5_f"
                y = np.array([d.elo.get(f"{pre}{f}", np.nan) for f in fr])
                lo = np.array([d.lo.get(f"{pre}{f}", np.nan) for f in fr])
                hi = np.array([d.hi.get(f"{pre}{f}", np.nan) for f in fr])
                x = np.arange(len(fr)) + (j - 0.5) * 0.38
                ax.bar(x, y, width=0.36, color=COL[fam], zorder=2)
                ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=INK2, lw=0.9, capsize=0, zorder=3)
                sh = d.elo.get("shipped", np.nan)
                if j == 0 and np.isfinite(sh):
                    ax.axhline(sh, color="#0b0b0b", lw=1.0, ls=(0, (4, 2)), zorder=1)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_ylim(bottom=max(ax.get_ylim()[0], ELO_FLOOR - 40))
            ax.set_xticks(range(len(fr)))
            ax.set_xticklabels([f"{f}%" for f in fr])
            ax.set_xlabel("fraction of the table each member keeps")
            ax.set_ylabel(f"{MODEL_NAME[m]}: Elo gain over single pass")
            _despine(ax)
    from matplotlib.lines import Line2D
    axs[0][0].legend(handles=[Patch(color=COL["A4"], label="A4  columns withheld"),
                              Patch(color=COL["A5"], label="A5  rows withheld"),
                              Line2D([], [], color="#0b0b0b", lw=1.0, ls=(0, (4, 2)), label="shipped (reference)")],
                     loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=7, borderaxespad=0,
                     columnspacing=1.0, handlelength=1.4)
    fig.tight_layout(w_pad=1.4, h_pad=1.2)
    _save(fig, out, "fig_wrappers")
    plt.close(fig)


FAM = [("A4", "A4  columns", COL["A4"]), ("A5", "A5  rows", COL["A5"]),
       ("A7", "A7  category relabelling", "#8e6bbf"), ("A8", "A8  feature expansion", "#d6457a"),
       ("control", "shipped at M=16 (vs control)", "#0b0b0b")]
ELO_FLOOR = -800        # bars below this are cut: those methods lose every unit (Elo is unbounded)


def fig_shipplus(T, out, plt):
    """Marginal Elo of one extra axis ON TOP of the shipped recipe."""
    from matplotlib.patches import Patch
    d = T["elo_shipplus"]
    tasks = [t for t in TASKS if (d.task == t).any()]
    fig, axs = plt.subplots(2, len(tasks), figsize=(3.7 * len(tasks), 4.8), squeeze=False)
    for i, m in enumerate(MODELS):
        for k, t in enumerate(tasks):
            ax = axs[i][k]
            sub = d[(d.model == m) & (d.task == t)]
            sub = pd.concat([sub[sub.panel == f[0]].sort_values("method") for f in FAM])
            x = np.arange(len(sub))
            col = {f[0]: f[2] for f in FAM}
            ax.bar(x, sub.elo.clip(lower=ELO_FLOOR), color=[col[p] for p in sub.panel], width=0.72, zorder=2)
            y, lo, hi = sub.elo.to_numpy(), sub.lo.to_numpy(), sub.hi.to_numpy()
            ax.errorbar(x, y.clip(min=ELO_FLOOR), yerr=[np.maximum(0, y.clip(min=ELO_FLOOR) - lo.clip(min=ELO_FLOOR)),
                                                          np.maximum(0, hi.clip(min=ELO_FLOOR) - y.clip(min=ELO_FLOOR))],
                        fmt="none", ecolor=INK2, lw=0.9, capsize=0, zorder=3)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_ylim(bottom=max(ELO_FLOOR - 40, min(ax.get_ylim()[0], ELO_FLOOR - 40)))
            ax.set_xticks(x)
            ax.set_xticklabels([c.replace("S_", "") for c in sub.method], rotation=55, ha="right", fontsize=7)
            ax.set_ylabel(f"{MODEL_NAME[m]}: Elo vs {'matched control' if m == 'tabicl' else 'shipped'}")
            _despine(ax)
    axs[0][0].legend(handles=[Patch(color=f[2], label=f[1]) for f in FAM if f[0] != "control" or (d.panel == "control").any()],
                     loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=7, borderaxespad=0,
                     columnspacing=1.0, handlelength=1.2)
    fig.tight_layout(w_pad=1.4, h_pad=1.2)
    _save(fig, out, "fig_shipplus")
    plt.close(fig)


def fig_elo_vs_M(T, out, plt):
    d = T["elo_vs_M"]
    tasks = [t for t in TASKS if (d.task == t).any()]
    fig, axs = plt.subplots(2, len(tasks), figsize=(3.5 * len(tasks), 4.6), squeeze=False)
    sty = {"A1": ("#9bbbe0", "s"), "A2": ("#e5a52c", "^"), "A3": ("#1baf7a", "D"),
           "A1A3": ("#8e6bbf", "P"), "A1A2A3": ("#4f86c6", "o"), "shipped": ("#0b0b0b", "*")}
    for i, m in enumerate(MODELS):
        for k, t in enumerate(tasks):
            ax = axs[i][k]
            for c, (col, mk) in sty.items():
                s = d[(d.model == m) & (d.task == t) & (d.coalition == c)].sort_values("M")
                if s.empty:
                    continue
                ax.plot(s.M, s.elo, color=col, marker=mk, ms=5.5 if mk != "*" else 8, lw=1.3,
                        mec="white", mew=0.6, label=c.replace("A1A2A3", "A1+A2+A3"), zorder=3)
                ax.fill_between(s.M, s.lo, s.hi, color=col, alpha=0.12, lw=0, zorder=1)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xscale("log", base=2)
            ms = [1, 2, 4, 8, 16, 32]
            ax.set_xticks(ms)
            ax.set_xticklabels(ms)
            ax.minorticks_off()
            ax.set_xlabel("members M (forward passes)")
            ax.set_ylabel(f"{MODEL_NAME[m]}: Elo gain over single pass")
            _despine(ax)
    h, l = axs[0][0].get_legend_handles_labels()
    for ax in axs.ravel():
        hh, ll = ax.get_legend_handles_labels()
        if len(ll) > len(l):
            h, l = hh, ll
    axs[0][0].legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=5, fontsize=7,
                     borderaxespad=0, columnspacing=1.0, handlelength=1.4)
    fig.tight_layout(w_pad=1.4, h_pad=1.2)
    _save(fig, out, "fig_elo_vs_M")
    plt.close(fig)


def fig_gap(T, out, plt):
    """Elo gain against effect size, one point per coalition: the gap between the
    two ways of measuring the same ensemble."""
    from matplotlib.lines import Line2D
    eff = T["effect_all"].set_index(["model", "task", "coalition"])
    el = T["elo_native"].rename(columns={"method": "coalition"}).set_index(["model", "task", "coalition"])
    wrapper = pd.concat([T["elo_A4"], T["elo_A5"]]).rename(columns={"method": "coalition"})
    wrapper = wrapper[wrapper.coalition.str.startswith(("A4_", "A5_"))].set_index(["model", "task", "coalition"])
    allel = pd.concat([el[["elo"]], wrapper[["elo"]]])
    allel = allel[~allel.index.duplicated()]
    tasks = [t for t in TASKS if (T["elo_native"].task == t).any()]
    fig, axs = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.1), squeeze=False)
    mk = lambda c: "o" if c in NATIVE else ("s" if c.startswith("A4") else "^")   # noqa: E731
    for ax, t in zip(axs[0], tasks):
        for (m, tt, c), r in allel.iterrows():
            if tt != t or (m, tt, c) not in eff.index or c == "base":
                continue
            ax.scatter(eff.loc[(m, tt, c), "effect"], r.elo, s=26, marker=mk(c), color=COL[m],
                       edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xscale("symlog", linthresh=max(1e-3, 0.002))
        ax.set_xlabel(f"{EFFECT[t]} gain over single pass" if t != "regression" else "relative RMSE reduction over single pass")
        ax.set_ylabel("Elo gain over single pass")
        _despine(ax)
        ax.grid(axis="x", visible=True)
    axs[0][0].legend(handles=[Line2D([], [], marker="o", ls="", color=COL[m], label=MODEL_NAME[m]) for m in MODELS]
                     + [Line2D([], [], marker=k, ls="", color="#52514e", label=l)
                        for k, l in (("o", "native"), ("s", "A4"), ("^", "A5"))],
                     loc="lower left", bbox_to_anchor=(0, 1.01), ncol=5, fontsize=7, borderaxespad=0,
                     columnspacing=0.9, handlelength=1.0)
    fig.tight_layout(w_pad=1.4)
    _save(fig, out, "fig_gap")
    plt.close(fig)


# ----------------------------------------------------------------- markdown --
def _cell(mu, lo, hi, d=0):
    return "n/a" if not np.isfinite(mu) else f"{mu:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def write_markdown(T, out: Path):
    """Compact tables: Elo gain and effect size, with 95% CIs."""
    L = ["# Results tables", "",
         "Elo is local (pool of our own methods), anchored at the single pass = 0; CIs are 95% bootstrap over",
         "datasets. Effect size: AUC gain / log-likelihood gain / relative RMSE reduction.", ""]
    eff = T["effect_all"].set_index(["model", "task", "coalition"])
    for t in TASKS:
        d = T["elo_native"][T["elo_native"].task == t]
        if d.empty:
            continue
        L += [f"## Native coalitions - {t}", "",
              "| coalition | TabICLv2 Elo | TabICLv2 effect | TabFM Elo | TabFM effect |", "|---|---|---|---|---|"]
        for c in [x for x in NATIVE if (d.method == x).any()]:
            cells = []
            for m in MODELS:
                r = d[(d.model == m) & (d.method == c)]
                e = eff.loc[(m, t, c)] if (m, t, c) in eff.index else None
                cells += [_cell(*r[["elo", "lo", "hi"]].iloc[0]) if len(r) else "n/a",
                          _cell(e.effect, e.lo, e.hi, 4) if e is not None else "n/a"]
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")
    for name, title in (("elo_A4", "A4 column withholding (Elo vs single pass)"),
                        ("elo_A5", "A5 row withholding (Elo vs single pass)")):
        d = T[name]
        if d.empty:
            continue
        L += [f"## {title}", "", "| coalition | task | TabICLv2 | TabFM |", "|---|---|---|---|"]
        for t in TASKS:
            for c in sorted(x for x in d[d.task == t].method.unique() if x.startswith(("A4_", "A5_"))):
                cells = []
                for m in MODELS:
                    r = d[(d.model == m) & (d.task == t) & (d.method == c)]
                    cells.append(_cell(*r[["elo", "lo", "hi"]].iloc[0]) if len(r) else "n/a")
                L.append(f"| {c} | {t} | " + " | ".join(cells) + " |")
        L.append("")
    d = T["elo_shipplus"]
    if not d.empty:
        L += ["## Shipped + one axis (Elo vs reference)", "",
              "TabICLv2 reference = S_ctl (matched control); TabFM reference = shipped. Pools are per axis family.",
              "", "| coalition | task | TabICLv2 | TabFM |", "|---|---|---|---|"]
        for t in TASKS:
            for c in sorted(d[d.task == t].method.unique()):
                cells = []
                for m in MODELS:
                    r = d[(d.model == m) & (d.task == t) & (d.method == c)]
                    cells.append(_cell(*r[["elo", "lo", "hi"]].iloc[0]) + f" (n={int(r.n_datasets.iloc[0])})" if len(r) else "n/a")
                L.append(f"| {c} | {t} | " + " | ".join(cells) + " |")
        L.append("")
    d = T["elo_vs_M"]
    if not d.empty:
        L += ["## Elo gain vs member budget (native coalitions)", "",
              "| model | task | coalition | " + " | ".join(f"M={m}" for m in (1, 2, 4, 8, 16, 32)) + " |",
              "|---|---|---|" + "---|" * 6]
        for m in MODELS:
            for t in TASKS:
                for c in ("A1", "A2", "A3", "A1A3", "A1A2A3", "shipped"):
                    r = d[(d.model == m) & (d.task == t) & (d.coalition == c)].set_index("M")
                    if r.empty:
                        continue
                    L.append(f"| {MODEL_NAME[m]} | {t} | {c} | " + " | ".join(
                        f"{r.elo[k]:.0f}" if k in r.index else "" for k in (1, 2, 4, 8, 16, 32)) + " |")
        L.append("")
    (out / "RESULTS_TABLES.md").write_text("\n".join(L))


# --------------------------------------------------------------------- main --
def _md(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--boot", type=int, default=100, help="Elo bootstrap rounds per panel")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path(a.results))
    if not data:
        raise SystemExit(f"no results under {a.results}")
    T = build_tables(data, a.boot)
    for k, v in T.items():
        v.to_csv(out / f"table_{k}.csv", index=False)
    plt = _style()
    fig_elo_native(T, out, plt)
    fig_effect_native(T["effect_all"], out, plt)
    if len(T["elo_A5"]):
        fig_wrappers(T, out, plt)
    if len(T["elo_shipplus"]):
        fig_shipplus(T, out, plt)
    if len(T["elo_vs_M"]):
        fig_elo_vs_M(T, out, plt)
    fig_gap(T, out, plt)
    write_markdown(T, out)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
