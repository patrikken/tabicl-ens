#!/bin/bash
# Run ON A LOGIN NODE, once, before submitting anything.
#
#     cd ~/tabicl-ens && bash slurm/00_setup_env.sh
#
# Compute nodes have NO INTERNET. Everything that would download at runtime is
# fetched here: the TabICLv2 checkpoint, the OpenML tasks/datasets, every pip
# package. A job that tries to download on a compute node hangs until the wall
# clock kills it.
#
# NOTE: `tabarena` is deliberately NOT installed. It pulls a pre-release
# autogluon.tabular plus ray and does not install cleanly on the cluster. It is
# not needed: TabArena's splits are OpenML task splits (their wrapper is a pure
# passthrough to openml), so experiments/prepare_tabarena.py reproduces them
# from the vendored task table using `openml` alone, and asserts every split
# against TabArena's own recorded train/test sizes.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

mkdir -p "$PROJECT_ROOT" "$HF_HOME" "$OPENML_CACHE_DIR" "$CACHE_DIR" \
         "$TABARENA_DIR" "$PROJECT_ROOT/logs" "$CODE_ROOT/logs"

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 arrow/17.0.0

# venv on $SCRATCH: persists across jobs, unlike $SLURM_TMPDIR
[ -d "$VENV" ] || virtualenv --no-download "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# CC wheelhouse first, PyPI only for what it lacks. No autogluon, no ray.
pip install --no-index torch scikit-learn numpy scipy pandas tqdm psutil
pip install einops huggingface-hub openml pyyaml

# upstream tabicl, editable, from the nested clone
pip install -e "$TABICL_REPO"

echo
echo "--- prefetching TabICLv2 checkpoint ---"
python - <<'PY'
from huggingface_hub import hf_hub_download
print("cached:", hf_hub_download(repo_id="jingang/TabICL",
                                 filename="tabicl-classifier-v2-20260212.ckpt"))
PY

echo
echo "--- materialising official TabArena v0.1 splits (openml only) ---"
# Also warms the OpenML cache for every task, so compute nodes stay offline.
python -m experiments.prepare_tabarena --out "$TABARENA_DIR"

echo
echo "--- verifying imports resolve without cwd tricks ---"
( cd / && python -c "
import experiments.run_cell, tabicl
from experiments.datasets import tabarena_datasets
print('  imports OK;', len(tabarena_datasets()), 'datasets in manifest')
" )

cat <<EOF

Environment ready.
  CODE_ROOT    = $CODE_ROOT
  TABICL_REPO  = $TABICL_REPO
  PROJECT_ROOT = $PROJECT_ROOT
  TABARENA_DIR = $TABARENA_DIR

Next:
  sbatch slurm/02_a6_control.sh     # Phase 0 gates - must pass first
  bash   slurm/submit_full.sh       # full campaign (DRY=1 to preview)
EOF
