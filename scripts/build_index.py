"""Pre-build (and cache) the offline index of the baselines that have one: B4 HLG graph, B6 LaceMem tuples, B7 memory records.
Extraction is done with the generator LLM, cached on disk per (generator, prompt version, document) and RESUMABLE: finished
documents are skipped, so an interrupted build continues where it stopped. run_baselines.py calls this automatically.

  python scripts/build_index.py --system hlg --dataset 2wiki --data datasets/wiki2_data/2wiki_100.json --generator qwen2.5:3b
  python scripts/build_index.py --system evimem,lychee --dataset musique --data datasets/musique_data/musique_100.json
"""
import argparse, os, sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
from registerrag import registry
from registerrag.cli import build_run
from registerrag.config import RunConfig

ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True, help="comma separated --system names / folder ids")
ap.add_argument("--dataset", required=True, choices=["2wiki", "musique", "hotpot"])
ap.add_argument("--data", required=True)
ap.add_argument("--generator", default="qwen2.5:3b")
ap.add_argument("--limit", type=int)
ap.add_argument("--stub", action="store_true")
ap.add_argument("--method-arg", action="append", default=[], metavar="K=V")
a = ap.parse_args()

from registerrag.cli import parse_method_args
for token in [t for t in a.system.split(",") if t]:
    b = registry.resolve(token)
    cfg = RunConfig(system=b.system, dataset=a.dataset, data_path=a.data, generator=a.generator, limit=a.limit,
                    prepend_title=True, prompt_version="v2", method_args=parse_method_args(a.method_arg))
    _, pipe, _, store = build_run(cfg, a.stub, os.path.join(ROOT, ".cache"))
    if not hasattr(pipe, "build_index"):
        print(f"{b.folder}: no offline index, nothing to build"); continue
    if hasattr(pipe.generator, "ping"):
        print(f"{b.folder}: {pipe.generator.ping()}")
    print(f"{b.folder}: building index over {len(store.docs)} documents ...", flush=True)
    print(f"{b.folder}: done", pipe.build_index())
