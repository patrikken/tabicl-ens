"""Score the TabFM cache: aggregate members with the recorded recipe, per cell.

    python -m experiments.score_tabfm cache/tabfm results/tabfm_scores.csv

No logit clipping: TabFM logits reach ~1e3, and the TabICLv2 pipeline's +-80 clip
saturates them into argmax ties (anneal accuracy 99.7% -> 19.7%). Only non-finite
values are repaired.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from experiments.analyze_full import softmax, cell_score
import numpy as _np
def sanitize(m):
    # TabFM logits reach ~1e3; the TabICLv2 +-80 clip saturates them into ties. Repair non-finite only.
    n_nan=int(_np.isnan(m).sum()); n_pos=int(_np.isposinf(m).sum()); n_neg=int(_np.isneginf(m).sum())
    if n_nan or n_pos or n_neg: m=_np.nan_to_num(m,nan=0.0,posinf=6e4,neginf=-6e4)
    return m, dict(n_nan=n_nan,n_posinf=n_pos,n_neginf=n_neg,frac_nonfinite=(n_nan+n_pos+n_neg)/max(1,m.size))

cache = Path(sys.argv[1]); out = Path(sys.argv[2])
rows = []
for mf in sorted(cache.rglob("meta.json")):
    m = json.loads(mf.read_text()); d = mf.parent
    mem = np.load(d/"members.npy").astype(np.float32)
    y = np.load(d/"y_test.npy", allow_pickle=True)
    mem, h = sanitize(mem)
    classes = np.unique(y); yi = np.searchsorted(classes, y); C = len(classes); M = mem.shape[0]
    tau = m.get("softmax_temperature", 0.9)
    base = dict(dataset=m["dataset"], coalition=m["coalition"], split=m["split"], M=M,
                m_req=m["n_estimators_requested"], truncated=m.get("truncated"),
                n_classes=C, task="binary" if C==2 else "multiclass",
                n_train=m["n_train"], n_test=m["n_test"], n_features=m["n_features"],
                severity=m.get("split_severity"), **h)
    err = (mem.argmax(-1) != yi[None]).astype(float)
    if M > 1 and err.std() > 0:
        R = np.nan_to_num(np.corrcoef(err), nan=0.0); np.fill_diagonal(R, 1.0)
        meff = float(M**2/(R**2).sum())
    else: meff = 1.0
    p = softmax(mem.mean(0), tau); p /= p.sum(-1, keepdims=True)
    # member-0 alone = a single pass of this coalition
    p1 = softmax(mem[0], tau); p1 /= p1.sum(-1, keepdims=True)
    rows.append(dict(**base, meff=meff, score=cell_score(p, yi, C), score_single=cell_score(p1, yi, C),
                     fit_s=m.get("fit_seconds"), pred_s=m.get("predict_seconds")))
df = pd.DataFrame(rows); df.to_csv(out, index=False)
print(len(df), "cells ->", out)
