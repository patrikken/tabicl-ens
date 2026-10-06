"""TabFM coalition figures. No titles and no text annotations inside axes: identity goes in legends,
captions carry the rest."""
import sys, numpy as np, pandas as pd
sys.path[:0]=["/home/claude/arena/experiments"]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import elo as E

scores, sweep, out = sys.argv[1], sys.argv[2], sys.argv[3]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#dcdcd8"
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],
 "mathtext.fontset":"stix","font.size":8.5,"axes.labelsize":8.5,"xtick.labelsize":8,"ytick.labelsize":8,
 "axes.edgecolor":MUTED,"axes.linewidth":0.6,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.5,
 "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK2,"ytick.color":INK2,"figure.dpi":200,
 "savefig.bbox":"tight","savefig.pad_inches":0.02,"legend.frameon":False,"pdf.fonttype":42})
ORDER=["A1","A2","A3","A1A2","A1A3","A2A3","A1A2A3","shipped"]
LAB={"A1":"A1","A2":"A2","A3":"A3","A1A2":"A1+A2","A1A3":"A1+A3","A2A3":"A2+A3","A1A2A3":"A1+A2+A3","shipped":"shipped"}
GROUP={"A1":"one axis","A2":"one axis","A3":"one axis","A1A2":"mix of axes","A1A3":"mix of axes","A2A3":"mix of axes","A1A2A3":"mix of axes","shipped":"shipped"}
GREEN="#1baf7a"
TASK={"binary":"ROC-AUC","multiclass":"$-$log-loss","regression":"$-$RMSE (relative)"}
_ALLCOL={"binary":BLUE,"multiclass":ORANGE,"regression":GREEN}
PAL={"binary":{"one axis":"#9bbbe0","mix of axes":"#4f86c6","shipped":"#0b0b0b"},
     "multiclass":{"one axis":"#f4a98a","mix of axes":"#e0703f","shipped":"#0b0b0b"},
     "regression":{"one axis":"#8fd6b9","mix of axes":"#1baf7a","shipped":"#0b0b0b"}}
rng=np.random.default_rng(0)
def despine(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
d=pd.read_csv(scores); sw=pd.read_csv(sweep)
TCOL={t:c for t,c in _ALLCOL.items() if (d.task==t).any()}      # tasks present in the data
NT=len(TCOL); W=3.6*NT
if "rel_gain" not in d: d["rel_gain"]=np.nan
d["g"]=np.where(d.task=="regression",d.rel_gain,d.gain)          # regression: relative RMSE reduction
def order(task): return [c for c in ORDER if c in set(d[d.task==task].coalition)]
def per_ds(task): return d[d.task==task].groupby(["coalition","dataset"]).g.mean().unstack(0)[order(task)]

# ---------- Elo from the sweep pool: every (coalition, M) is a method; anchor = single pass
def sweep_elo(task, boot=100):
    z=sw[sw.task==task].copy()
    z["method"]=np.where(z.coalition=="base","base",z.coalition+" M="+z.M.astype(str))
    pool=pd.DataFrame({"method":z.method,"dataset":z.dataset,"fold":z.split,
        "metric_error":E.to_metric_error(z.score,z.task),"metric":"m","problem_type":task})
    pool=E.complete_subgrid(pool,verbose=False)
    r=E.elo_ratings(pool,anchor="base",bootstrap_rounds=boot).ratings
    r=r.reset_index() if "method" not in r.columns else r
    return r.set_index("method")[["elo","elo_lo","elo_hi"]]
ELO={t:sweep_elo(t) for t in TCOL}
pd.concat({t:ELO[t] for t in ELO}).to_csv(f"{out}/tabfm_elo_sweep.csv")

# ---------- Fig 1: gain, dataset dots + mean/CI
fig,axs=plt.subplots(1,NT,figsize=(W,3.1),sharey=False,squeeze=False); axs=axs[0]
for ax,task in zip(axs,TCOL):
    col=TCOL[task]; x=per_ds(task); n=len(x); O=list(x.columns); ys=np.arange(len(O))[::-1]
    for y,c in zip(ys,O):
        ax.scatter(x[c],y+rng.uniform(-.17,.17,n),s=9,color=col,alpha=.3,lw=0,zorder=2)
        bs=[x[c].iloc[rng.integers(0,n,n)].mean() for _ in range(2000)]; lo,hi=np.percentile(bs,[2.5,97.5])
        ax.plot([lo,hi],[y,y],color=INK,lw=1.4,zorder=3,solid_capstyle="butt")
        ax.scatter([x[c].mean()],[y],s=26,color=col,edgecolor="white",lw=.8,zorder=4)
    ax.axvline(0,color=INK2,lw=.8,zorder=1); ax.set_yticks(ys); ax.set_yticklabels([LAB[c] for c in O])
    ax.set_xlabel(f"{TASK[task]} gain over single pass"); despine(ax); ax.grid(axis="y",visible=False)
    XL={"binary":(-0.0035,0.0078),"multiclass":(-0.0105,0.0095)}
    if task in XL: ax.set_xlim(*XL[task])
    if task==list(TCOL)[0]:
        ax.legend(handles=[Line2D([],[],marker="o",ls="",color=col,alpha=.45,ms=3.5,label="dataset"),
            Line2D([0],[0],color=INK,lw=1.4,marker="o",mfc=col,mec="white",ms=5,label="mean, 95% CI")],loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,fontsize=7,borderaxespad=0)
fig.tight_layout(w_pad=1.2); fig.savefig(f"{out}/fig_tabfm_gain.pdf"); fig.savefig(f"{out}/fig_tabfm_gain.png"); plt.close(fig)

# ---------- Fig 2: Elo bars at M=32 (gain over single pass = Elo - 1000)
GC={"one axis":"#6c9bd1","mix of axes":"#2a5fa0","shipped":"#0b0b0b"}
fig,axs=plt.subplots(1,NT,figsize=(W,2.9),squeeze=False); axs=axs[0]
for ax,task in zip(axs,TCOL):
    O=order(task); r=ELO[task]; v=[r.loc[f"{c} M=32"] for c in O]
    el=np.array([x.elo for x in v])-1000; lo=np.array([x.elo_lo for x in v])-1000; hi=np.array([x.elo_hi for x in v])-1000
    cols=[PAL[task][GROUP[c]] for c in O]
    ax.bar(range(len(O)),el,color=cols,width=.72,zorder=2)
    ax.errorbar(range(len(O)),el,yerr=[el-lo,hi-el],fmt="none",ecolor=INK2,lw=.9,capsize=0,zorder=3)
    ax.axhline(0,color=INK2,lw=.8); ax.set_xticks(range(len(O))); ax.set_xticklabels([LAB[c] for c in O],rotation=40,ha="right")
    ax.set_ylabel("Elo gain over single pass"); despine(ax); ax.grid(axis="x",visible=False)
    pal=PAL[task]
    ax.legend(handles=[Patch(color=pal[k],label=k) for k in pal],loc="lower left",bbox_to_anchor=(0,1.01),ncol=3,fontsize=7,borderaxespad=0,columnspacing=1.2,handlelength=1.2)
fig.tight_layout(w_pad=1.5); fig.savefig(f"{out}/fig_tabfm_elo.pdf"); fig.savefig(f"{out}/fig_tabfm_elo.png"); plt.close(fig)

# ---------- Fig 3: Elo vs member budget M
MS=[1,2,4,8,16,32]
STY={"A1":("#9bbbe0","s"),"A2":("#e5a52c","^"),"A3":("#1baf7a","D"),"A1A2A3":("#4f86c6","o"),"shipped":("#0b0b0b","*")}
fig,axs=plt.subplots(1,NT,figsize=(W,3.0),sharex=True,squeeze=False); axs=axs[0]
for ax,task in zip(axs,TCOL):
    r=ELO[task]
    for c,(col,mk) in STY.items():
        y=[r.loc[f"{c} M={m}"].elo-1000 if f"{c} M={m}" in r.index else np.nan for m in MS]
        ax.plot(MS,y,color=col,marker=mk,ms=5.5 if mk!="*" else 8,lw=1.3,label=LAB[c],mec="white",mew=.6,zorder=3)
    ax.axhline(0,color=INK2,lw=.8); ax.set_xscale("log",base=2); ax.set_xticks(MS); ax.set_xticklabels(MS)
    ax.minorticks_off(); ax.set_xlabel("members M (forward passes)"); ax.set_ylabel("Elo gain over single pass"); despine(ax)
    if task==list(TCOL)[0]: ax.legend(loc="upper left",fontsize=7,ncol=1)
fig.tight_layout(w_pad=1.5); fig.savefig(f"{out}/fig_tabfm_elo_vs_M.pdf"); fig.savefig(f"{out}/fig_tabfm_elo_vs_M.png"); plt.close(fig)

# ---------- Fig 4: heatmap, no cell text
fig,axs=plt.subplots(1,NT,figsize=(3.7*NT,4.2),squeeze=False); axs=axs[0]
for ax,task in zip(axs,TCOL):
    x=per_ds(task); x=x.loc[x["shipped"].sort_values(ascending=False).index]; v=np.nanpercentile(np.abs(x.values),90)
    im=ax.imshow(x.values,aspect="auto",cmap="RdBu",norm=TwoSlopeNorm(0,-v,v))
    ax.set_xticks(range(len(x.columns))); ax.set_xticklabels([LAB[c] for c in x.columns],rotation=45,ha="right")
    ax.set_yticks(range(len(x))); ax.set_yticklabels([s[:24] for s in x.index],fontsize=7); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    cb=fig.colorbar(im,ax=ax,fraction=.05,pad=.02); cb.set_label(f"{TASK[task]} gain over single pass",fontsize=7.5); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
fig.tight_layout(w_pad=1.0); fig.savefig(f"{out}/fig_tabfm_heat.pdf"); fig.savefig(f"{out}/fig_tabfm_heat.png"); plt.close(fig)
print("ok")
