#!/bin/bash
# One-off: move TabICLv2 cells written by the older scripts (03/05/06/07, flat at
# $CACHE_DIR/<dataset>/<coalition>/split*/) into $CACHE_DIR/tabiclv2/, where
# submit_all.sh and the grid now look. Without this, skip-complete cannot see the
# old cells and would recompute them.
#
#   bash slurm/migrate_cache.sh            # list what would move
#   APPLY=1 bash slurm/migrate_cache.sh    # move it
set -euo pipefail
source "$(dirname "$0")/_common.sh"
DEST="$CACHE_DIR/tabiclv2"
n=0
for d in "$CACHE_DIR"/*/; do
  name=$(basename "$d")
  case "$name" in tabiclv2|tabfm) continue;; esac
  # a dataset directory holds <coalition>/split<k>/meta.json
  compgen -G "$d*/split*/meta.json" > /dev/null || continue
  n=$((n+1))
  if [ "${APPLY:-0}" = 1 ]; then
    mkdir -p "$DEST"
    if [ -e "$DEST/$name" ]; then echo "  CONFLICT $name already in tabiclv2/ -- merge by hand"; continue; fi
    mv "$d" "$DEST/$name" && echo "  moved $name"
  else
    echo "  would move $name -> tabiclv2/"
  fi
done
echo "$n dataset dir(s) $([ "${APPLY:-0}" = 1 ] && echo processed || echo found; true)"
[ "${APPLY:-0}" = 1 ] || echo "(dry: re-run with APPLY=1)"
