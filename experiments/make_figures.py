"""Paper figures from the full-campaign scores.csv.

Vector PDF, serif to match the paper's Times body text.

Palette: validated categorical slots 1 (blue) and 2 (orange) --
worst-pair CVD dE 24.7, normal-vision 33.6, both clear of the floors.
Two hues only; identity is carried by direct labels and panel titles, never
by colour alone. No dual-axis panels: measures with different units get their
own panel.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
GRID = "#dcdcd8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

ORDER = ["base", "A1", "A2", "A3", "A1A2", "A1A3", "A2A3", "A1A2A3", "shipped"]
BUDGETS = [1, 2, 4, 8, 16, 32]
AXES = ["A1", "A2", "A3"]


def tidy():
    df = pd.read_csv("scores.csv").drop_duplicates(
        subset=["dataset", "coalition", "split", "budget"])
    base = (df[df.coalition == "base"].groupby(["dataset", "split"])["score"]
            .mean().rename("b").reset_index())
    d = df.merge(base, on=["dataset", "split"])
    d["gain"] = d.score - d.b
    m1 = (d[d.budget == 1].groupby(["dataset", "coalition", "split"])["score"]
          .mean().rename("m1").reset_index())
    d = d.merge(m1, on=["dataset", "coalition", "split"], how="left")
    d["matched"] = d.score - d.m1
    return df, d


def despine(ax, left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)


# --------------------------------------------------------------- figure 1 ---
def fig_meff(df):
    inv = df[df.budget == 32]
    m = (inv.groupby("coalition").agg(M=("m_realised", "mean"),
                                      meff=("meff", "mean")))
    m = m.reindex([c for c in ORDER if c != "base"])
    m["ratio"] = m.meff / m.M

    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    y = np.arange(len(m))[::-1]
    # A2/A3/A2A3 are structurally capped far below M=32 (n_classes, 5 norm
    # methods), so their ratio is not comparable to the M=32 coalitions.
    # Lighter fill keeps them visible without reading as a ranking.
    LIGHT = "#a8c8ee"
    cols = [ORANGE if i == "shipped" else (LIGHT if r.M < 20 else BLUE)
            for i, r in m.iterrows()]
    ax.barh(y, m.ratio * 100, height=0.62, color=cols, zorder=3)
    for yi, (name, r) in zip(y, m.iterrows()):
        ax.text(r.ratio * 100 + 1.2, yi,
                f"{r.meff:.2f} of {r.M:.0f}", va="center", ha="left",
                fontsize=7.5, color=INK2)
    ax.set_yticks(y, m.index)
    ax.set_xlim(0, 62)
    ax.set_xlabel("effective members as % of nominal members "
                  "($M_{\\mathrm{eff}}/M$)")
    ax.text(0.995, -0.30, "pale bars: structurally capped below $M=32$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=MUTED)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    despine(ax, left=False)
    ax.tick_params(axis="y", length=0)
    fig.savefig("fig_meff.pdf")
    plt.close(fig)
    print("  fig_meff.pdf")


# --------------------------------------------------------------- figure 2 ---
def fig_budget(df, d):
    inv = df[df.budget == 32]
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.7), sharex=True)
    for col, (task, metric) in enumerate([("binary", "ROC-AUC"),
                                          ("multiclass", "$-$log-loss")]):
        sub = d[(d.task == task) & (d.coalition == "shipped")]
        cc = inv[(inv.task == task) & (inv.coalition == "shipped")]
        fit, spm = cc.fit_s.mean(), cc.sec_per_member_total.mean()
        gains = np.array([sub[sub.budget == B].gain.mean() for B in BUDGETS])
        secs = np.array([fit + spm * B for B in BUDGETS])
        per = gains / secs

        a0, a1 = axes[0, col], axes[1, col]
        a0.axhline(0, color=MUTED, lw=0.6, zorder=1)
        a0.plot(BUDGETS, gains, "-o", color=BLUE, lw=1.6, ms=4, zorder=3)
        a0.set_xscale("log", base=2)
        a0.set_title(f"{task} ({metric})", loc="left", pad=5)
        if col == 0:
            a0.set_ylabel("gain over one\nunperturbed pass")
        despine(a0)

        k = int(np.argmax(per))
        a1.plot(BUDGETS, per, "-o", color=BLUE, lw=1.6, ms=4, zorder=3)
        a1.plot([BUDGETS[k]], [per[k]], "o", color=ORANGE, ms=7, zorder=4)
        a1.annotate("peak $M=" + str(BUDGETS[k]) + "$", (BUDGETS[k], per[k]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7.5, color=ORANGE)
        a1.axvline(32, color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=2)
        a1.annotate("shipped", (32, per.max()), textcoords="offset points",
                    xytext=(-5, -2), fontsize=7, color=INK2,
                    ha="right", va="top")
        a1.set_xscale("log", base=2)
        a1.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        a1.set_xlabel("ensemble members $M$")
        if col == 0:
            a1.set_ylabel("gain per GPU-second")
        despine(a1)

    fig.tight_layout(h_pad=0.9, w_pad=1.6)
    fig.savefig("fig_budget.pdf")
    plt.close(fig)
    print("  fig_budget.pdf")


# --------------------------------------------------------------- figure 3 ---
def fig_frontier(df, d):
    inv = df[df.budget == 32]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6))
    for ax, (task, metric) in zip(axes, [("binary", "ROC-AUC"),
                                         ("multiclass", "$-$log-loss")]):
        sub = d[(d.task == task) & (d.budget == 32)]
        cost = (inv[inv.task == task].groupby("coalition")
                .apply(lambda g: g.fit_s.mean() + g.pred_s.mean()))
        gain = sub.groupby("coalition")["gain"].mean()
        pts = pd.DataFrame({"cost": cost, "gain": gain}).dropna()
        pts = pts.reindex([c for c in ORDER if c in pts.index])

        # hand-placed to avoid collisions; verified by rendering
        OFF = {
            "binary": {"A1": (5, -10), "A1A3": (5, -10), "A2": (-4, 6),
                       "A1A2A3": (5, 3), "shipped": (-6, 5), "base": (5, -10)},
            "multiclass": {"A1A3": (-2, 8), "A1A2A3": (-6, -12),
                           "A1": (-16, -4), "shipped": (5, -3),
                           "A2": (-4, 6), "A3": (-4, 6), "A1A2": (5, -9),
                           "A2A3": (-6, 7), "base": (5, -10)},
        }[task]
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        for name, r in pts.iterrows():
            is_ship = name == "shipped"
            ax.scatter(r.cost, r.gain, s=42 if is_ship else 26,
                       color=ORANGE if is_ship else BLUE,
                       zorder=4 if is_ship else 3,
                       edgecolor="white", linewidth=0.7)
            ax.annotate(name, (r.cost, r.gain), textcoords="offset points",
                        xytext=OFF.get(name, (5, 3)), fontsize=7,
                        color=ORANGE if is_ship else INK2,
                        fontweight="bold" if is_ship else "normal")
        ax.set_xlabel("GPU-seconds per dataset")
        ax.set_title(f"{task} ({metric})", loc="left", pad=5)
        ax.set_xlim(left=0)
        ax.margins(x=0.22, y=0.22)
        despine(ax)
    axes[0].set_ylabel("gain over one\nunperturbed pass")
    fig.tight_layout(w_pad=1.8)
    fig.savefig("fig_frontier.pdf")
    plt.close(fig)
    print("  fig_frontier.pdf")


# --------------------------------------------------------------- figure 4 ---
def shapley(vals):
    import itertools, math
    n, S = len(AXES), {}
    for a in AXES:
        others = [x for x in AXES if x != a]
        tot = 0.0
        for k in range(len(others) + 1):
            for comb in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                tot += w * (vals.get("".join(sorted([*comb, a])), 0.0)
                            - vals.get("".join(sorted(comb)) or "base", 0.0))
        S[a] = tot
    return S


def fig_phi(df, d):
    inv = df[df.budget == 32]
    task = "multiclass"
    sub = d[(d.task == task) & (d.budget == 32)]
    per = {}
    for ds, s in sub.groupby("dataset"):
        v = s.groupby("coalition")["gain"].mean().to_dict()
        v["base"] = 0.0
        per[ds] = shapley(v)
    sh = pd.DataFrame(per).T
    phi = sh[AXES].mean()
    spm = (inv[inv.task == task].groupby("coalition")["sec_per_member_total"]
           .mean())
    phit = phi / spm[AXES]

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3))
    x = np.arange(len(AXES))
    for ax, vals, lab, ttl in [
            (axes[0], phi, "$\\phi_a$", "Raw contribution"),
            (axes[1], phit, "$\\tilde\\phi_a = \\phi_a\\,/\\,$GPU-s per member",
             "Cost-adjusted")]:
        top = vals.idxmax()
        ax.bar(x, vals.values, width=0.55,
               color=[ORANGE if a == top else BLUE for a in AXES], zorder=3)
        for xi, a in zip(x, AXES):
            ax.text(xi, vals[a], f"{vals[a]:.4f}".lstrip("0"),
                    ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x, AXES)
        ax.set_title(ttl, loc="left", pad=5)
        ax.set_xlabel(lab)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.margins(y=0.22)
        despine(ax, left=False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticklabels([])
    fig.tight_layout(w_pad=2.0)
    fig.savefig("fig_phi.pdf")
    plt.close(fig)
    print("  fig_phi.pdf")
    print(f"     phi:  {dict(phi.round(5))}")
    print(f"     phit: {dict(phit.round(5))}")




# --------------------------------------------------------------- figure 5 ---
# A5: the diversity / member-quality trade-off.
# absolute = member_quality + matched_gain, exactly, so the panel is an additive
# decomposition rather than three unrelated curves. All three are score deltas,
# so they share one axis (no dual-axis).
# Palette slots 1/2/3; #1baf7a WARNs on contrast, so every series is
# direct-labelled as well as legended -- the relief the validator requires.
AQUA = "#1baf7a"
A5F = ["A5_f25", "A5_f50", "A5_f75", "A5_f90"]


def fig_a5(df, d):
    ds18 = sorted(df[df.coalition == "A5_f50"].dataset.unique())
    sub = d[d.dataset.isin(ds18)]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7))
    for ax, (task, metric) in zip(axes, [("binary", "ROC-AUC"),
                                         ("multiclass", "$-$log-loss")]):
        s = sub[(sub.task == task) & (sub.budget == 16)]
        fr, mq, mg, ab = [], [], [], []
        for c in A5F:
            x = s[s.coalition == c]
            fr.append(int(c.split("_f")[1]) / 100)
            mq.append((x.m1 - x.b).mean())
            mg.append(x.matched.mean())
            ab.append(x.gain.mean())
        ax.axhline(0, color=INK2, lw=0.7, zorder=2)
        ax.plot(fr, mg, "-o", color=ORANGE, lw=1.7, ms=4, zorder=4,
                label="diversity gained (matched)")
        ax.plot(fr, mq, "-o", color=AQUA, lw=1.7, ms=4, zorder=4,
                label="member quality lost")
        ax.plot(fr, ab, "-o", color=BLUE, lw=2.0, ms=5, zorder=5,
                label="net vs full context")
        ax.annotate("diversity gained", (fr[0], mg[0]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color=ORANGE)
        ax.annotate("quality lost", (fr[0], mq[0]),
                    textcoords="offset points", xytext=(4, -10),
                    fontsize=7, color=AQUA)
        ax.annotate("net", (fr[-1], ab[-1]), textcoords="offset points",
                    xytext=(-2, -12), fontsize=7, color=BLUE, ha="right")
        ax.set_xlabel("context fraction $f$ held by each member")
        ax.set_xticks(fr, [f"{v:.2f}" for v in fr])
        ax.set_title(f"{task} ({metric})", loc="left", pad=5)
        ax.margins(y=0.16)
        despine(ax)
    axes[0].set_ylabel("score delta")
    axes[1].legend(loc="lower right", fontsize=7, handlelength=1.4)
    fig.tight_layout(w_pad=1.8)
    fig.savefig("fig_a5.pdf")
    plt.close(fig)
    print("  fig_a5.pdf")


def fig_a5_h1(df, d):
    """H1: context vs feature decorrelation, matched datasets and budget."""
    ds18 = sorted(df[df.coalition == "A5_f50"].dataset.unique())
    sub = d[d.dataset.isin(ds18) & (d.budget == 16)]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
    for ax, task in zip(axes, ["binary", "multiclass"]):
        s = sub[sub.task == task]
        rows = []
        for c in A5F + ["A5bal_f50", "A1", "A2", "A3", "A1A3", "A1A2A3", "shipped"]:
            x = s[s.coalition == c]
            if len(x):
                rows.append((c, x.matched.mean(),
                             c.startswith("A5")))
        rows.sort(key=lambda r: r[1])
        y = np.arange(len(rows))
        ax.barh(y, [r[1] for r in rows], height=0.6,
                color=[ORANGE if r[2] else BLUE for r in rows], zorder=3)
        ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
        ax.set_xlabel("diversity gain at matched context size")
        ax.set_title(task, loc="left", pad=5)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        despine(ax, left=False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=1.4)
    fig.savefig("fig_a5_h1.pdf")
    plt.close(fig)
    print("  fig_a5_h1.pdf")


if __name__ == "__main__":
    df, d = tidy()
    print("rendering:")
    fig_meff(df)
    fig_budget(df, d)
    fig_frontier(df, d)
    fig_phi(df, d)
    if any(c.startswith("A5") for c in df.coalition.unique()):
        fig_a5(df, d)
        fig_a5_h1(df, d)
