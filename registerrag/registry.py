"""Baseline registry. Every baseline lives in its own folder  baselines/<id>/  (id = b<N>_<name> or ref_<name>).

A folder with a baseline.py is an implemented baseline; baseline.py declares
    SYSTEM        the value of --system
    LABEL         label used in tables
    DEFAULT_UNIT  'sentence' | 'chunk'   (retrieval unit when --unit is not given)
    pipeline_class()                      returns the pipeline class (imported lazily, so heavy dependencies of one
                                          baseline never slow down or break another)
    variants(generator, coder=None, openai=False) -> [(label, extra_cli_args), ...]
A folder with only a README is listed as 'planned'. Legacy scripts in other folders are ignored.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_DIR = os.path.join(ROOT, "baselines")
_ID = re.compile(r"^(b\d+[a-z]?|ref)_[a-z0-9_]+$")
_CACHE: Optional[list] = None


@dataclass
class Baseline:
    folder: str
    system: Optional[str]
    label: str
    default_unit: str
    status: str                  # implemented | planned | broken
    module: object = None
    error: Optional[str] = None

    @property
    def path(self) -> str:
        return os.path.join(BASELINES_DIR, self.folder)

    @property
    def results_dir(self) -> str:
        return os.path.join(self.path, "results")

    def variants(self, generator, coder=None, openai=False):
        fn = getattr(self.module, "variants", None)
        return fn(generator, coder, openai) if fn else [(self.label, [])]

    def pipeline_class(self):
        return self.module.pipeline_class()


def _sort_key(folder: str):
    m = re.match(r"b(\d+)([a-z]?)_", folder)
    return (0, int(m.group(1)), m.group(2), folder) if m else (1, 0, "", folder)


def discover(refresh: bool = False) -> list:
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    out = []
    if os.path.isdir(BASELINES_DIR):
        for name in sorted(os.listdir(BASELINES_DIR), key=_sort_key):
            p = os.path.join(BASELINES_DIR, name)
            if not (os.path.isdir(p) and _ID.match(name)):
                continue
            if not os.path.exists(os.path.join(p, "baseline.py")):
                out.append(Baseline(name, None, name, "sentence", "planned"))
                continue
            try:
                mod = importlib.import_module(f"baselines.{name}.baseline")
                out.append(Baseline(name, mod.SYSTEM, mod.LABEL, getattr(mod, "DEFAULT_UNIT", "sentence"),
                                    "implemented", mod))
            except Exception as e:  # noqa: BLE001 - one broken baseline must not take the others down
                print(f"[registry] WARNING: baseline '{name}' failed to load: {e}", file=sys.stderr)
                out.append(Baseline(name, None, name, "sentence", "broken", error=str(e)))
    _CACHE = out
    return out


def implemented() -> list:
    return [b for b in discover() if b.status == "implemented"]


def systems() -> list:
    return [b.system for b in implemented()]


def get(system: str) -> Baseline:
    for b in implemented():
        if b.system == system:
            return b
    raise ValueError(f"unknown system '{system}'. Implemented: {systems()}")


def resolve(token: str) -> Baseline:
    """Accepts a --system value ('pyrag'), a folder id ('b8_pyrag') or a short id ('b8')."""
    for b in implemented():
        if token in (b.system, b.folder) or b.folder.startswith(token + "_"):
            return b
    raise ValueError(f"unknown baseline '{token}'. Implemented: {[b.folder for b in implemented()]}")


def pipeline_class(system: str):
    return get(system).pipeline_class()


def default_unit(system: str) -> str:
    return get(system).default_unit


def results_dir(system: str) -> str:
    return get(system).results_dir


def label(system: str) -> str:
    try:
        return get(system).label
    except ValueError:
        return system
