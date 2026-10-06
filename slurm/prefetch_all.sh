#!/bin/bash
# LOGIN NODE, once, before submit_all.sh. Compute nodes have no internet.
#
#     cd ~/tabicl-ens && bash slurm/prefetch_all.sh
#
# Fetches everything the regression + TabFM campaign needs that the
# classification-only setup did not:
#   1. the TabICLv2 REGRESSOR checkpoint          (jingang/TabICL, HF cache)
#   2. the TabFM checkpoint, BOTH heads           (google/tabfm-1.0.0-pytorch,
#      classification/ and regression/ -- 6.5 GB each)
#   3. the regression datasets and their official splits, by re-running
#      prepare_tabarena with --include-regression (the classification entries
#      are regenerated identically; the OpenML cache makes it quick)
#
# NB the TabFM weights are tabfm-non-commercial-v1.0: research use only.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cc_activate
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE      # this script is the one place that must go online
export TABFM_CKPT="${TABFM_CKPT:-$PROJECT_ROOT/tabfm-ckpt}"

echo "--- 1/3 TabICLv2 regressor checkpoint ---"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ("tabicl-classifier-v2-20260212.ckpt", "tabicl-regressor-v2-20260212.ckpt"):
    print("cached:", hf_hub_download(repo_id="jingang/TabICL", filename=f))
PY

echo "--- 2/3 TabFM checkpoint (classification + regression) -> $TABFM_CKPT ---"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("google/tabfm-1.0.0-pytorch", local_dir="$TABFM_CKPT",
                  allow_patterns=["classification/**", "regression/**", "config.json"])
PY
ls "$TABFM_CKPT"

echo "--- 3/3 TabArena splits incl. regression -> $TABARENA_DIR ---"
python -m experiments.prepare_tabarena --out "$TABARENA_DIR" --include-regression

echo
python -m experiments.grid plan --cache "$CACHE_DIR" --skip-complete
echo
echo "Next:  bash slurm/submit_all.sh        (DRY=1 to preview)"
