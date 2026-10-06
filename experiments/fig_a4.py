import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

BLUE,ORANGE,AQUA="#2a78d6","#eb6834","#1baf7a"
INK,INK2,MUTED,GRID="#0b0b0b","#52514e","#8a8a85","#dcdcd8"
plt.rcParams.update({"font.family":"serif",
 "font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix",
 "font.size":8.5,"axes.labelsize":8.5,"axes.titlesize":9,"xtick.labelsize":8,
 "ytick.labelsize":8,"legend.fontsize":7.5,"axes.edgecolor":MUTED,"axes.linewidth":0.6,
 "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.5,"xtick.color":INK2,
 "ytick.color":INK2,"axes.labelcolor":INK,"text.color":INK,"figure.dpi":200,
 "savefig.bbox":"tight","savefig.pad_inches":0.02,"legend.frameon":False,"pdf.fonttype":42})

A=pd.read_csv("a4_scores.csv"); S=pd.read_csv("scores.csv")
base=S[S.coalition=="base"].groupby(["dataset","split"]).agg(
    base=("score","mean"),task=("task","first")).reset_index()
def build(df):
    m1=df[df.budget==1].groupby(["dataset","coalition","split"]).score.mean().rename("m1").reset_index()
    x=df[df.budget==df.groupby("coalition").budget.transform("max")]
    x=x[["dataset","coalition","split","score"]].merge(base,on=["dataset","split"]).merge(
        m1,on=["dataset","coalition","split"])
    x["quality"]=x.m1-x.base; x["div"]=x.score-x.m1; x["net"]=x.score-x.base
    return x
GR={"A4_g25":.25,"A4_g50":.50,"A4_g75":.75,"A4_g90":.90}
FR={"A5_f25":.25,"A5_f50":.50,"A5_f75":.75,"A5_f90":.90}
a4=build(A); a4["f"]=a4.coalition.map(GR)
a5=build(S[S.coalition.isin(FR)]); a5["f"]=a5.coalition.map(FR)
a5=a5[a5.dataset.isin(set(a4.dataset))]

def despine(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)

fig,ax=plt.subplots(2,2,figsize=(5.8,4.4))
for col,(task,metric) in enumerate([("binary","ROC-AUC"),("multiclass","$-$log-loss")]):
    # --- top: recovery rate, one curve ---
    a=ax[0,col]
    a.axhline(1.0,color=INK2,lw=0.8,ls=(0,(4,3)),zorder=2)
    a.annotate("break-even",xy=(0.255,1.0),xytext=(0,4),textcoords="offset points",
               fontsize=7,color=INK2,va="bottom")
    pts={}
    for lbl,src,c,mk in (("A5: drop rows",a5,ORANGE,"o"),("A4: drop columns",a4,AQUA,"s")):
        g=src[src.task==task].groupby("f").apply(
            lambda d: -d["div"].mean()/d.quality.mean(), include_groups=False)
        pts[lbl]=g
        a.plot(g.index,g.values,"-"+mk,color=c,lw=1.5,ms=4.5,zorder=4,label=lbl,
               mec="white",mew=0.6)
    a.set_title(f"{task} ({metric})",loc="left",pad=5)
    a.set_ylabel("recovery rate\n(diversity $/$ quality lost)" if col==0 else "")
    a.set_xlabel("fraction of the data each member keeps")
    a.margins(0.12); despine(a)

    # --- bottom: magnitudes, log scale ---
    b=ax[1,col]
    for lbl,src,c,mk in (("A5: drop rows",a5,ORANGE,"o"),("A4: drop columns",a4,AQUA,"s")):
        g=src[src.task==task].groupby("f").quality.mean()
        b.plot(g.index,-g.values,"-"+mk,color=c,lw=1.5,ms=4.5,zorder=4,mec="white",mew=0.6)
    b.set_yscale("log")
    g4=a4[a4.task==task].groupby("f").quality.mean(); g5=a5[a5.task==task].groupby("f").quality.mean()
    r=(g4/g5).mean()
    b.annotate(f"A4 costs {r:.1f}$\\times$ more",xy=(0.5,0.86),xycoords="axes fraction",
               ha="center",fontsize=7.5,color=INK2)
    b.set_ylabel("member quality destroyed\n(log scale)" if col==0 else "")
    b.set_xlabel("fraction of the data each member keeps")
    b.margins(x=0.12); despine(b)
ax[0,0].legend(loc="upper left",handlelength=1.4)
fig.tight_layout(h_pad=1.2,w_pad=1.6)
fig.savefig("fig_a4_recovery.pdf")
print("  fig_a4_recovery.pdf")
