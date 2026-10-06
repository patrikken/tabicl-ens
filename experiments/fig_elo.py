"""Elo paired with effect size: what a leaderboard place is actually worth.

Builds the pool (ours + whatever published artifacts are present), rates every
method the way TabArena does, and puts the raw gain beside the rating. The point
of the figure is the gap between the two columns.

    python -m experiments.fig_elo --scores results/scores.csv --model TabICLv2 \\
        --pool $PROJECT_ROOT/arena_pool --tabfm-repo tabfm --out .

Elo is pool-dependent, so every number is captioned with the pool it came from.
Ratings are computed per task type here -- binary units only, then multiclass
units only -- because the effect size on the other axis is per task type and
ROC-AUC and log-loss must never be pooled. TabArena's published Elo mixes both
(battles are within-unit, so that is legitimate); the headline ladder uses that
convention and is labelled accordingly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from experiments import arena as A                   # noqa: E402
from experiments import elo as E                     # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#dcdcd8"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "text.color": INK, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "legend.frameon": False, "pdf.fonttype": 42})


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def build_pool(scores: pd.DataFrame, model: str, pool_dir: Path | None,
               tabfm_repo: Path | None, coalitions: list[str] | None,
               budget: int | None) -> pd.DataFrame:
    parts = [A.our_results(scores, model=model, coalitions=coalitions, budget=budget)]
    if pool_dir is not None and pool_dir.exists():
        root = pool_dir if pool_dir.name.startswith("tabarena-") else pool_dir / A.FROZEN_SUITE
        if root.exists():
            parts.append(A.load_pool(root))
        else:
            print(f"[pool] {root} absent -- run `arena.py fetch` on a login node")
    if tabfm_repo is not None and (tabfm_repo / "results").exists():
        parts.append(A.load_tabfm_shipped(tabfm_repo))
    pool = pd.concat(parts, ignore_index=True)
    return pool.drop_duplicates(subset=["method", "dataset", "fold"])


def rate(pool: pd.DataFrame, problem_types: tuple[str, ...], anchor: str | None,
         bootstrap: int = 200) -> pd.DataFrame:
    sub = pool[pool.problem_type.isin(problem_types)]
    sub = E.complete_subgrid(sub, verbose=False)
    if anchor is None or anchor not in set(sub.method):
        # anchor on the single unperturbed pass when RF is not in the pool, so
        # the zero point is still something the paper can name.
        cand = [m for m in sub.method.unique() if "base" in m]
        anchor = cand[0] if cand else sorted(sub.method.unique())[0]
    r = E.elo_ratings(sub, anchor=anchor, bootstrap_rounds=bootstrap)
    out = r.ratings.copy()
    out.attrs["anchor"] = anchor
    out.attrs["n_datasets"] = sub.dataset.nunique()
    out.attrs["n_units"] = int(sub.groupby(["dataset", "fold"]).ngroups)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--model", default="TabICLv2")
    p.add_argument("--pool", type=Path, default=None)
    p.add_argument("--tabfm-repo", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--anchor", default=None)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--coalitions", nargs="*", default=None)
    a = p.parse_args(argv)

    scores = pd.read_csv(a.scores)
    pool = build_pool(scores, a.model, a.pool, a.tabfm_repo, a.coalitions, a.budget)
    eff = A.effect_sizes(scores, budget=a.budget)
    print(f"[pool] {pool.method.nunique()} methods, "
          f"{pool.groupby(['dataset','fold']).ngroups} units")

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 4.8))
    for col, (task, pts, metric) in enumerate([
            ("binary", ("binary",), "ROC-AUC"),
            ("multiclass", ("multiclass",), "$-$log-loss")]):
        r = rate(pool, pts, a.anchor, a.bootstrap)
        e = eff[eff.task == task].set_index("coalition")
        ours = {m: c for m in r.index for c in e.index
                if m.startswith(a.model) and f" {c} (" in m}
        x, y, lab = [], [], []
        for m, c in ours.items():
            x.append(r.loc[m, "elo"]); y.append(e.loc[c, "gain"]); lab.append(c)
        if len(x) < 3:
            print(f"[{task}] only {len(x)} of our coalitions in the pool; skipping")
            continue
        x, y, lab = np.asarray(x), np.asarray(y), np.asarray(lab)
        k = np.polyfit(x, y, 1)
        per100 = k[0] * 100

        # Top: every coalition. Bottom: only those that do not LOSE to a single
        # pass -- the region any real wrapper lives in, where the whole spread
        # is a rounding error and the Elo ladder still looks decisive.
        for row in (0, 1):
            ax = axes[row, col]
            sel = np.ones(len(x), bool) if row == 0 else (y > -1e-4)
            if sel.sum() < 2:
                ax.set_visible(False)
                continue
            # The zoom gets its OWN fit: the global slope is set by the
            # catastrophic low-f coalitions and overstates the exchange rate in
            # the region any shippable wrapper occupies.
            kk = k if row == 0 else np.polyfit(x[sel], y[sel], 1)
            xs = np.linspace(x[sel].min(), x[sel].max(), 50)
            ax.plot(xs, np.polyval(kk, xs), color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
            ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
            ax.scatter(x[sel], y[sel], s=30, color=BLUE, zorder=3,
                       edgecolor="white", linewidth=0.6)
            if row == 1:
                # stagger labels alternately above/below to stop them colliding
                order = np.argsort(x[sel])
                for j, i in enumerate(order):
                    dy = (7, -11, 15)[j % 3]
                    ax.annotate(lab[sel][i], (x[sel][i], y[sel][i]),
                                textcoords="offset points", xytext=(0, dy),
                                ha="center", fontsize=6.0, color=INK2)
                ax.set_xlabel(f"Elo  (anchor: {r.attrs['anchor'].split('(')[0].strip()})")
                ax.set_title("coalitions that beat a single pass", loc="right",
                             pad=4, fontsize=7, color=MUTED)
                ax.annotate(f"+100 Elo $=$ {kk[0]*100:+.5f} here",
                            xy=(0.04, 0.95), xycoords="axes fraction", va="top",
                            fontsize=7.5, color=ORANGE)
                local = kk[0] * 100
            else:
                ax.set_title(f"{task} ({metric})", loc="left", pad=5)
                ax.annotate(f"+100 Elo $=$ {per100:+.5f}", xy=(0.04, 0.95),
                            xycoords="axes fraction", va="top", fontsize=8,
                            color=ORANGE, fontweight="bold")
            if col == 0:
                ax.set_ylabel("gain over a single\nunperturbed pass" if row == 0
                              else "gain (zoom)")
            ax.margins(0.18); despine(ax)

        # the headline sentence, as numbers: what the shipped wrapper buys
        base = [m for m in ours if " base (" in m]
        ship = [m for m in ours if " shipped (" in m]
        d_elo = d_eff = float("nan")
        if base and ship:
            d_elo = r.loc[ship[0], "elo"] - r.loc[base[0], "elo"]
            d_eff = e.loc["shipped", "gain"] - e.loc["base", "gain"]
        rows.append(dict(task=task, elo_per_unit=k[0], unit_per_100_elo=per100,
                         unit_per_100_elo_usable=locals().get("local", float("nan")),
                         shipped_minus_base_elo=d_elo,
                         shipped_minus_base_effect=d_eff,
                         n_methods=len(r), n_datasets=r.attrs["n_datasets"],
                         n_units=r.attrs["n_units"], anchor=r.attrs["anchor"]))
        r.assign(task=task).to_csv(a.out / f"elo_{task}.csv")

    fig.tight_layout(h_pad=1.1, w_pad=1.8)
    fig.savefig(a.out / "fig_elo_vs_effect.pdf")
    print("  fig_elo_vs_effect.pdf")

    conv = pd.DataFrame(rows)
    conv.to_csv(a.out / "elo_conversion.csv", index=False)
    print("\n=== Elo <-> effect size conversion ===")
    print(conv.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
