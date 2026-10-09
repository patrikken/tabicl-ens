"""Move provisional cells out of the cache so skip-complete recomputes them.

Nothing is deleted: cells go to  $CACHE_DIR/_stale/<stamp>/<model>/<dataset>/...
(delete that directory yourself once the re-run has been scored).

  tabicl   every  <dataset>/S_*  directory           (written before the MemberView fix:
                                                      one-member estimators ignored the
                                                      feature/class shuffles)
  tabfm    every  plus_*  split directory of a CLASSIFICATION dataset that has no
           agg.npy                                    (scoring cannot replay calibration
                                                      from the cached logits alone)

Idempotent: re-run cells are fresh (and TabFM ones carry agg.npy), so a second call
finds nothing for TabFM; for TabICLv2 the marker file ``.quarantined_tabicl_S`` stops
a second call from moving the re-run cells.

    python -m experiments.quarantine CACHE_DIR [--dry] [--models tabicl tabfm] [--force]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

MARKER = ".quarantined_tabicl_S"


def _move(src: Path, root: Path, stale: Path, dry: bool) -> None:
    dst = stale / src.relative_to(root.parent)
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def tabicl_targets(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/S_*") if p.is_dir())


def tabfm_targets(root: Path, classification: set[str]) -> list[Path]:
    out = []
    for ds in classification:
        for split in (root / ds).glob("plus_*/split*"):
            if split.is_dir() and not (split / "agg.npy").exists():
                out.append(split)
    return sorted(out)


def run(cache: Path, models, dry: bool, force: bool = False,
        classification: set[str] | None = None) -> dict[str, int]:
    from experiments.grid import cache_root                                  # noqa: PLC0415
    stale = cache / "_stale" / time.strftime("%Y%m%d-%H%M%S")
    counts = {}
    if "tabicl" in models:
        root = cache_root("tabicl", cache)
        if (root / MARKER).exists() and not force:
            print(f"[tabicl] already quarantined once ({root / MARKER}); --force to repeat")
            counts["tabicl"] = 0
        else:
            t = tabicl_targets(root)
            for p in t:
                _move(p, root, stale, dry)
            if t and not dry:
                (root / MARKER).write_text(time.strftime("%F %T") + "\n")
            counts["tabicl"] = len(t)
            print(f"[tabicl] {len(t)} S_* coalition dirs "
                  f"({len({p.parent.name for p in t})} datasets)")
    if "tabfm" in models:
        root = cache_root("tabfm", cache)
        if classification is None:
            from experiments.datasets import tabarena_datasets               # noqa: PLC0415
            classification = set(tabarena_datasets("classification"))
        t = tabfm_targets(root, classification)
        for p in t:
            _move(p, root, stale, dry)
        counts["tabfm"] = len(t)
        print(f"[tabfm] {len(t)} classification plus_* splits without agg.npy "
              f"({len({p.parent.parent.name for p in t})} datasets)")
    print(("(dry) would move to " if dry else "moved to ") + str(stale)
          if any(counts.values()) else "nothing to move")
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", type=Path)
    ap.add_argument("--models", nargs="+", default=["tabicl", "tabfm"],
                    choices=["tabicl", "tabfm"])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    run(a.cache, a.models, a.dry, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
