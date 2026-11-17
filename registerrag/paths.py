"""Finding run folders wherever they live (baselines/<id>/results/, legacy runs/, or any directory given)."""
from __future__ import annotations

import glob
import os

from .registry import BASELINES_DIR, ROOT


def run_folders(arg=None, include_stability: bool = False) -> list:
    """Every folder that contains a results.jsonl, below `arg` (a results dir, a baseline folder, a single run folder,
    or the 'baselines' dir). With no argument: all baselines plus a legacy top-level runs/ directory if present.
    Stability repetitions (baselines/<id>/results/stability/...) are excluded unless asked for."""
    roots = [arg] if arg else [BASELINES_DIR] + [p for p in (os.path.join(ROOT, "runs"),) if os.path.isdir(p)]
    found = set()
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", "results.jsonl"), recursive=True):
            d = os.path.dirname(p)
            if not include_stability and f"{os.sep}stability{os.sep}" in d + os.sep:
                continue
            found.add(d)
    return sorted(found)
