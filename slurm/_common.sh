# Shared setup for every SLURM script. Source it; do not run it.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# Defines three roots that are NOT the same directory:
#
#   CODE_ROOT     the tabicl-ens checkout. Holds experiments/ and slurm/.
#                 This is what Python must import from.
#   TABICL_REPO   $CODE_ROOT/tabicl - the upstream clone. Only used for
#                 `pip install -e` and for recording its git SHA.
#   PROJECT_ROOT  scratch workspace: venv, cache, hf, openml. On $SCRATCH.
#
# Conflating CODE_ROOT with TABICL_REPO is what caused
# "No module named 'experiments'": the job cd'd into the nested clone, which
# has no experiments/ package.
#
# Nothing here depends on the current working directory. PYTHONPATH carries
# CODE_ROOT, so `python -m experiments.run_cell` resolves from anywhere.

# --- CODE_ROOT: explicit, else submit dir, else this script's parent --------
_resolve_code_root() {
  local c
  for c in "${CODE_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" \
           "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"; do
    if [ -n "$c" ] && [ -d "$c/experiments" ]; then
      printf '%s' "$c"
      return 0
    fi
  done
  return 1
}

if ! CODE_ROOT="$(_resolve_code_root)"; then
  cat >&2 <<'ERR'
FATAL: could not locate the tabicl-ens checkout (no experiments/ directory found).

Tried, in order: $CODE_ROOT, $SLURM_SUBMIT_DIR, the parent of slurm/.

Fix by either submitting from the checkout root:
    cd ~/tabicl-ens && sbatch slurm/01_pilot_array.sh
or passing it explicitly:
    sbatch --export=ALL,CODE_ROOT=$HOME/tabicl-ens slurm/01_pilot_array.sh
ERR
  exit 1
fi
export CODE_ROOT

export TABICL_REPO="${TABICL_REPO:-$CODE_ROOT/tabicl}"
export PROJECT_ROOT="${PROJECT_ROOT:-$SCRATCH/tabicl-ens}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf}"
export OPENML_CACHE_DIR="${OPENML_CACHE_DIR:-$PROJECT_ROOT/openml}"
export CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/cache}"
export VENV="${VENV:-$PROJECT_ROOT/venv}"
# Official TabArena splits, materialised on a login node by
# experiments/prepare_tabarena.py. The worker reads this instead of
# importing tabarena (which pulls a pre-release autogluon + ray).
export TABARENA_DIR="${TABARENA_DIR:-$PROJECT_ROOT/tabarena}"

# Import from the checkout without depending on cwd. This is the actual fix.
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- offline: compute nodes have no internet -------------------------------
# Fail immediately rather than hanging until the wall clock expires.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- determinism -----------------------------------------------------------
# Gate 0.2 asserts exactly-zero seed spread. That only means something if the
# numerics are pinned, and they must stay pinned identically for the campaign.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --- modules + venv --------------------------------------------------------
cc_activate() {
  module purge
  module load StdEnv/2023 python/3.11 cuda/12.2 arrow/17.0.0
  if [ ! -d "$VENV" ]; then
    echo "FATAL: no venv at $VENV. Run slurm/00_setup_env.sh on a login node first." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
}

cc_banner() {
  echo "=== $(date -Is) on $(hostname) ==="
  echo "  CODE_ROOT    = $CODE_ROOT"
  echo "  TABICL_REPO  = $TABICL_REPO"
  echo "  PROJECT_ROOT = $PROJECT_ROOT"
  echo "  PYTHONPATH   = $PYTHONPATH"
  command -v nvidia-smi >/dev/null && \
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  python -c "import experiments, tabicl, torch; \
print('  experiments  =', experiments.__file__); \
print('  tabicl       =', tabicl.__version__); \
print('  torch/cuda   =', torch.__version__, torch.cuda.is_available())"
}
