#!/bin/bash
# Run ON A LOGIN NODE, once, before submitting anything.
#
#     cd ~/tabicl-ens && bash slurm/00_setup_env.sh
#
# Compute nodes have NO INTERNET. Everything that would download at runtime must
# be fetched here first: the TabICLv2 checkpoint, the OpenML files, every pip
# package. A job that tries to download on a compute node hangs until the wall
# clock kills it. This is the usual way to lose a first day on CC.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

mkdir -p "$PROJECT_ROOT" "$HF_HOME" "$OPENML_CACHE_DIR" "$CACHE_DIR" \
         "$PROJECT_ROOT/logs" "$CODE_ROOT/logs"

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 arrow/17.0.0

# venv on $SCRATCH: persists across jobs, unlike $SLURM_TMPDIR
[ -d "$VENV" ] || virtualenv --no-download "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# CC wheelhouse first, PyPI only for what it lacks
pip install --no-index torch scikit-learn numpy scipy pandas tqdm psutil
pip install einops huggingface-hub openml pyyaml

# upstream tabicl, editable, from the nested clone
pip install -e "$TABICL_REPO"

echo "--- prefetching checkpoint (login node has internet) ---"
python - <<'PY'
from huggingface_hub import hf_hub_download
print("cached:", hf_hub_download(repo_id="jingang/TabICL",
                                 filename="tabicl-classifier-v2-20260212.ckpt"))
PY

echo "--- prefetching pilot datasets ---"
python - <<'PY'
import openml
from experiments.datasets import PILOT
for name, (did, *_) in PILOT.items():
    ds = openml.datasets.get_dataset(did, download_data=True)
    ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    print(f"  cached {name} (oml {did})")
PY

echo "--- installing tabarena (LOGIN NODE ONLY, not needed on compute) ---"
# Pulls a pre-release autogluon.tabular + ray. Heavy, and only ever imported
# here: prepare_tabarena.py turns the official splits into a portable manifest
# that the compute-node worker reads with numpy alone.
pip install "tabarena==0.1.0" || {
  echo "WARN: tabarena install failed. Resolve before the full campaign;" >&2
  echo "      the pilot can still run with SPLIT_BACKEND=openml." >&2
}

echo "--- materialising official TabArena splits ---"
if python -c "import tabarena" 2>/dev/null; then
  python -m experiments.prepare_tabarena --out "$TABARENA_DIR"
else
  echo "SKIPPED: tabarena not importable."
fi

echo "--- verifying imports resolve without cwd tricks ---"
( cd / && python -c "import experiments.run_cell, tabicl; print('  imports OK')" )

cat <<EOF

Environment ready.
  CODE_ROOT    = $CODE_ROOT
  TABICL_REPO  = $TABICL_REPO
  PROJECT_ROOT = $PROJECT_ROOT
  VENV         = $VENV

Next:
  cd $CODE_ROOT
  sbatch slurm/02_a6_control.sh     # Phase 0 gates - must pass first
  sbatch slurm/01_pilot_array.sh    # the pilot

Submit from $CODE_ROOT so SLURM_SUBMIT_DIR resolves the checkout, or pass
  --export=ALL,CODE_ROOT=$CODE_ROOT
EOF
