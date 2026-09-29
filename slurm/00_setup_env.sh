#!/bin/bash
# Run ON A LOGIN NODE. Once, before submitting anything.
#
# Compute Canada compute nodes have NO INTERNET. Anything that would download at
# runtime must be fetched here first:
#   - the TabICLv2 checkpoint from HuggingFace
#   - OpenML dataset files
#   - every pip package
# A job that tries to download on a compute node hangs until it hits the wall
# clock. This is the single most common way to lose a first day on CC.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SCRATCH/tfm-ens}"
VENV="$PROJECT_ROOT/venv"
export HF_HOME="$PROJECT_ROOT/hf"
export OPENML_CACHE_DIR="$PROJECT_ROOT/openml"

mkdir -p "$PROJECT_ROOT" "$HF_HOME" "$OPENML_CACHE_DIR" "$PROJECT_ROOT/cache" "$PROJECT_ROOT/logs"

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 arrow/17.0.0

# --- venv on $SCRATCH (persists; $SLURM_TMPDIR would vanish between jobs) ----
if [ ! -d "$VENV" ]; then
  virtualenv --no-download "$VENV"
fi
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# CC wheelhouse first (--no-index), PyPI only for what it lacks.
pip install --no-index torch scikit-learn numpy scipy pandas tqdm psutil
pip install einops huggingface-hub openml pyyaml

# tabicl itself, editable, from the clone.
pip install -e "${TABICL_REPO:-$PROJECT_ROOT/tabicl}"

# --- pre-fetch the checkpoint (login node has internet) ---------------------
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="jingang/TabICL",
                    filename="tabicl-classifier-v2-20260212.ckpt")
print("checkpoint cached at:", p)
print("HF_HOME =", os.environ.get("HF_HOME"))
PY

# --- pre-fetch every pilot dataset ------------------------------------------
python - <<'PY'
import openml
from experiments.datasets import PILOT
for name, (did, *_ ) in PILOT.items():
    ds = openml.datasets.get_dataset(did, download_data=True)
    ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    print(f"cached {name} (oml {did})")
PY

cat <<EOF

Environment ready.
  PROJECT_ROOT = $PROJECT_ROOT
  VENV         = $VENV
  HF_HOME      = $HF_HOME

Put these in your ~/.bashrc or pass them through sbatch --export:
  export PROJECT_ROOT=$PROJECT_ROOT
  export HF_HOME=$HF_HOME
  export OPENML_CACHE_DIR=$OPENML_CACHE_DIR
  export HF_HUB_OFFLINE=1     # fail fast instead of hanging on compute nodes
EOF
