"""
Deterministic stratified dev/test splits.

The dev split is the only data we inspect while designing changes; the test
split is touched only for final numbers.  Splits are written in the input's
raw format, so every loader reads them unchanged.
"""
from __future__ import annotations
import json
import logging
import os
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


def write_splits(data_path: str, key: str, dev_per_group: int, seed: int, out_dir: str) -> tuple[str, str]:
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(data):
        groups[str(item.get(key, ""))].append(i)

    rng = random.Random(seed)
    dev_idx: set[int] = set()
    for g in sorted(groups):
        idx = sorted(groups[g])
        rng.shuffle(idx)
        dev_idx.update(idx[:dev_per_group])

    dev = [x for i, x in enumerate(data) if i in dev_idx]
    test = [x for i, x in enumerate(data) if i not in dev_idx]
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(data_path))[0]
    paths = []
    for name, part in (("dev", dev), ("test", test)):
        p = os.path.join(out_dir, f"{stem}_{name}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=1)
        paths.append(p)
    counts = {g: sum(1 for i in groups[g] if i in dev_idx) for g in sorted(groups)}
    print(f"dev  {len(dev):4d} -> {paths[0]}   per {key}: {counts}")
    print(f"test {len(test):4d} -> {paths[1]}")
    with open(os.path.join(out_dir, f"{stem}_split.json"), "w", encoding="utf-8") as f:
        json.dump({"source": data_path, "key": key, "dev_per_group": dev_per_group,
                   "seed": seed, "dev": len(dev), "test": len(test)}, f, indent=1)
    return paths[0], paths[1]
