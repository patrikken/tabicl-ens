#!/bin/bash
#SBATCH --job-name=tfm-ens-analyze
#SBATCH --account=def-CHANGEME
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# CPU only - no GPU. Extraction streams the member cache and scores every
# (cell x budget); the report is pure pandas afterwards.
#
#     cd ~/tabicl-ens && sbatch slurm/04_analyze.sh
#
# Extraction is resumable: it skips cells already in results/scores.csv, so a
# timeout just needs resubmitting. Once scores.csv exists you can iterate on
# the tables locally with
#     python -m experiments.analyze_full report --out results/

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cc_activate

RESULTS="${RESULTS_DIR:-$PROJECT_ROOT/results}"
mkdir -p "$RESULTS"

echo "=== extract ==="
python -m experiments.analyze_full extract --cache "$CACHE_DIR" --out "$RESULTS"

echo
echo "=== report (severity=usable: ok + resized, excludes leakage) ==="
python -m experiments.analyze_full report --out "$RESULTS" --severity usable \
  | tee "$RESULTS/report_usable.txt"

echo
echo "=== report (severity=ok: leaderboard-comparable splits only) ==="
python -m experiments.analyze_full report --out "$RESULTS" --severity ok \
  | tee "$RESULTS/report_ok.txt"

cat <<EOF

Artifacts in $RESULTS:
  scores.csv         one row per (dataset, coalition, split, budget)
  gains.csv          the above, joined to the single-pass baseline
  report_usable.txt  trend tables
  report_ok.txt      leaderboard-comparable subset

Copy scores.csv + gains.csv back for plotting; they are small.
EOF
