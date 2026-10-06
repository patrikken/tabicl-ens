import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.stats import pearsonr

BLUE,ORANGE="#2a78d6","#eb6834"; INK,INK2,MUTED,GRID="#0b0b0b","#52514e","#8a8a85","#dcdcd8"
plt.rcParams.update({"font.family":"serif",
 "font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix",
 "font.size":8.5,"axes.labelsize":8.5,"axes.titlesize":9,"xtick.labelsize":8,
 "ytick.labelsize":8,"legend.fontsize":7.5,"axes.edgecolor":MUTED,"axes.linewidth":0.6,
 "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.5,"xtick.color":INK2,
 "ytick.color":INK2,"axes.labelcolor":INK,"text.color":INK,"figure.dpi":200,
 "savefig.bbox":"tight","savefig.pad_inches":0.02,"legend.frameon":False,"pdf.fonttype":42})

L=pd.read_csv("lofo_scores.csv").drop_duplicates(subset=["dataset","coalition","split","budget"])
S=pd.read_csv("scores.csv").drop_duplicates(subset=["dataset","coalition","split","budget"])
base=(S[S.coalition=="base"].groupby(["dataset","split"])["score"].mean().rename("b").reset_index())
S=S.merge(base,on=["dataset","split"]); S["net"]=S.score-S.b
m1=(S[S.budget==1].groupby(["dataset","coalition","split"])["score"].mean().rename("m1").reset_index())
S=S.merge(m1,on=["dataset","coalition","split"]); S["quality"]=S.m1-S.b
L=L.merge(base,on=["dataset","split"]); L["net"]=L.score-L.b
o=L[L.budget==1][["dataset","coalition","split","score"]].rename(columns={"score":"m1"})
L=L.merge(o,on=["dataset","coalition","split"]); L["quality"]=L.m1-L.b

rows=[]
for task in ["binary","multiclass"]:
    for c,g in S[(S.task==task)&(S.budget==16)&(S.coalition!="base")].groupby("coalition"):
        rows.append(dict(task=task,coalition=c,meff=g.meff.mean(),quality=g.quality.mean(),
                         net=g.net.mean(),ctx=c.startswith("A5")))
    for c,g in L[L.task==task].groupby("coalition"):
        gg=g[g.budget==g.M]
        rows.append(dict(task=task,coalition=c,meff=gg.meff.mean(),quality=gg.quality.mean(),
                         net=gg.net.mean(),ctx=True))
r=pd.DataFrame(rows)

def despine(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)

fig,axes=plt.subplots(2,2,figsize=(5.8,4.6))
for col,(task,metric) in enumerate([("binary","ROC-AUC"),("multiclass","$-$log-loss")]):
    s=r[r.task==task]
    for row,(xv,xlab) in enumerate([("quality","member quality  (vs full-context pass)"),
                                    ("meff","member diversity  ($M_{\\mathrm{eff}}$)")]):
        ax=axes[row,col]
        ax.axhline(0,color=MUTED,lw=0.6,zorder=1)
        ax.scatter(s[~s.ctx][xv],s[~s.ctx].net,s=26,color=BLUE,zorder=3,
                   edgecolor="white",linewidth=0.6,label="feature-side")
        ax.scatter(s[s.ctx][xv],s[s.ctx].net,s=26,color=ORANGE,zorder=3,
                   edgecolor="white",linewidth=0.6,label="context-side")
        rp,_=pearsonr(s[xv],s.net)
        xs=np.linspace(s[xv].min(),s[xv].max(),50)
        k=np.polyfit(s[xv],s.net,1); ax.plot(xs,np.polyval(k,xs),color=INK2,lw=1.0,
                                             ls=(0,(4,3)),zorder=2)
        ax.set_title(f"{task} ({metric})" if row==0 else "",loc="left",pad=5)
        # park the correlation in the corner the fitted line leaves empty
        xy,va=((0.97,0.06),"bottom") if k[0]>0 else ((0.97,0.95),"top")
        ax.annotate(f"$r={rp:+.2f}$",xy=xy,xycoords="axes fraction",
                    ha="right",va=va,fontsize=8,
                    color=ORANGE if abs(rp)>0.7 else INK2,
                    fontweight="bold" if abs(rp)>0.7 else "normal")
        ax.set_xlabel(xlab)
        if col==0: ax.set_ylabel("net ensemble gain")
        ax.margins(0.14); despine(ax)
axes[0,0].legend(loc="upper left",handlelength=1.0)
fig.tight_layout(h_pad=1.0,w_pad=1.6)
fig.savefig("fig_quality_vs_diversity.pdf")
print("  fig_quality_vs_diversity.pdf")

# LOFO sweep figure: full range on top, the f->1 tail zoomed below, because
# the LOFO points all live in f in [0.875, 0.984] and vanish at full scale.
fig,axes2=plt.subplots(2,2,figsize=(5.8,4.0))
for col,(task,metric) in enumerate([("binary","ROC-AUC"),("multiclass","$-$log-loss")]):
    FR={"A5_f25":.25,"A5_f50":.50,"A5_f75":.75,"A5_f90":.90}
    ind=S[(S.budget==16)&(S.coalition.isin(FR))&(S.task==task)].copy()
    ind["f"]=ind.coalition.map(FR)
    gi=ind.groupby("f")["net"].mean()
    lf=L[(L.task==task)]; lf=lf[lf.budget==lf.M]
    gl=lf.groupby("frac")["net"].mean()
    allp=pd.concat([gi,gl]).sort_index(); pk=allp.idxmax()
    for row,lo in enumerate([0.0,0.85]):
        ax=axes2[row,col]
        ax.axhline(0,color=MUTED,lw=0.7,zorder=2)
        gi2,gl2=gi[gi.index>=lo],gl[gl.index>=lo]
        ax.plot(gi2.index,gi2.values,"-o",color=BLUE,lw=1.6,ms=4,zorder=3,
                label="independent draws")
        ax.plot(gl2.index,gl2.values,"-s",color=ORANGE,lw=1.6,ms=4,zorder=4,
                label="LOFO ($f=1-1/M$)")
        if pk>=lo:
            ax.plot([pk],[allp.max()],"o",ms=9,mfc="none",mec=INK,mew=1.2,zorder=5)
            ax.annotate(f"peak $f\\approx{pk:.3f}$",(pk,allp.max()),
                        textcoords="offset points",xytext=(0,9),ha="center",
                        fontsize=7.5,color=INK)
        if row==0: ax.set_title(f"{task} ({metric})",loc="left",pad=5)
        else:
            ax.set_xlabel("context fraction $f$ per member")
            ax.set_title("zoom: $f\\geq0.85$",loc="right",pad=4,
                         fontsize=7,color=MUTED)
        ax.margins(y=0.34); despine(ax)
axes2[0,0].set_ylabel("net gain vs\nfull-context single pass")
axes2[1,0].set_ylabel("net gain")
axes2[1,1].legend(loc="lower right",handlelength=1.4)
fig.tight_layout(h_pad=1.0,w_pad=1.6)
fig.savefig("fig_lofo_sweep.pdf")
print("  fig_lofo_sweep.pdf")
