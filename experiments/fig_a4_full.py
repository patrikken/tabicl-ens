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
R=pd.read_csv("recovery_points.csv")
VAR={"A4rr_g50":"round-robin","A4imp_g50":"importance","A4lofo_M8":"col-LOFO M8",
     "A4lofo_M16":"col-LOFO M16"}
def despine(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
fig,ax=plt.subplots(2,2,figsize=(5.8,4.6))
for col,(task,metric) in enumerate([("binary","ROC-AUC"),("multiclass","$-$log-loss")]):
    t=R[R.task==task]
    # --- top: recovery vs fraction, all 12 points ---
    a=ax[0,col]
    a.axhline(1.0,color=INK2,lw=0.8,ls=(0,(4,3)),zorder=2)
    sw5=t[(t.axis=="A5 rows")].sort_values("frac")
    sw4=t[(t.axis=="A4 cols")&(t.coalition.str.match(r"A4_g\d+"))].sort_values("frac")
    vr =t[(t.axis=="A4 cols")&(t.coalition.isin(VAR))]
    a.plot(sw5.frac,sw5.recovery,"-o",color=ORANGE,lw=1.5,ms=4.5,mec="white",mew=.6,
           zorder=4,label="A5: drop rows")
    a.plot(sw4.frac,sw4.recovery,"-s",color=AQUA,lw=1.5,ms=4.5,mec="white",mew=.6,
           zorder=4,label="A4: drop columns")
    a.plot(vr.frac,vr.recovery,"D",color=BLUE,ms=4.5,mec="white",mew=.6,zorder=5,
           label="A4: other selection rules")
    # round-robin and importance both sit at f~0.49, so stagger their labels
    OFF={"A4rr_g50":(-7,9,"right"),"A4imp_g50":(7,-10,"left"),
         "A4lofo_M8":(6,-1,"left"),"A4lofo_M16":(6,-1,"left")}
    for _,r in vr.iterrows():
        dx,dy,ha=OFF[r.coalition]
        a.annotate(VAR[r.coalition],(r.frac,r.recovery),textcoords="offset points",
                   xytext=(dx,dy),fontsize=6.3,color=BLUE,va="center",ha=ha)
    a.set_title(f"{task} ({metric})",loc="left",pad=5)
    if col==0: a.set_ylabel("recovery rate\n(diversity $/$ quality lost)")
    a.set_xlabel("fraction of the data each member keeps")
    a.set_xlim(0.15,1.12); a.margins(y=0.16); despine(a)

    # --- bottom: the fixed-k contrast ---
    b=ax[1,col]
    ORD=["A4rr_g50","A4_g50","A4imp_g50"]; LBL=["round-\nrobin","random","impor-\ntance"]
    s=t.set_index("coalition").loc[ORD]
    xp=np.arange(3)
    b.bar(xp-0.19,-s.quality.values/-s.quality.values[1],width=0.36,color=AQUA,zorder=3,
          label="quality destroyed")
    b.bar(xp+0.19,s.recovery.values/s.recovery.values[1],width=0.36,color=BLUE,zorder=3,
          label="recovery rate")
    b.axhline(1.0,color=INK2,lw=0.7,zorder=4)
    for i in range(3):
        b.text(xp[i]-0.19,-s.quality.values[i]/-s.quality.values[1]+.02,
               f"{-s.quality.values[i]/-s.quality.values[1]:.2f}",ha="center",va="bottom",
               fontsize=6.8,color=INK2)
        b.text(xp[i]+0.19,s.recovery.values[i]/s.recovery.values[1]+.02,
               f"{s.recovery.values[i]/s.recovery.values[1]:.2f}",ha="center",va="bottom",
               fontsize=6.8,color=INK2)
    b.set_xticks(xp,LBL); b.set_ylim(0,1.45)
    if col==0: b.set_ylabel("relative to random draws\n(same subset size $k$)")
    b.set_xlabel("column-selection rule at $g=0.50$")
    b.grid(axis="x",visible=False); b.set_axisbelow(True); despine(b)
ax[0,0].legend(loc="upper left",handlelength=1.3)
ax[1,1].legend(loc="upper right",handlelength=1.0,ncol=1)
fig.tight_layout(h_pad=1.3,w_pad=1.6)
fig.savefig("fig_a4_full.pdf"); print("  fig_a4_full.pdf")
