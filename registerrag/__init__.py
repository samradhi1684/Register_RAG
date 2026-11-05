"""RegisterRAG shared package: evidence units, retrieval, baselines, metrics, run logging."""
import faulthandler
import os

# Native-crash diagnostics: if a compiled library segfaults (Windows exit code 0xC0000005 = 3221225477),
# print the Python stack instead of dying silently.
faulthandler.enable()

# Mitigation for intermittent OpenBLAS access violations seen on some Windows setups. Only numpy's BLAS is
# affected (our matrix products are tiny); torch/GPU work is untouched. Must be set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.8"
