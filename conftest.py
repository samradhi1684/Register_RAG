"""Test-session setup: single-threaded BLAS/OpenMP (inherited by the CLI subprocesses the tests start). This machine shows
intermittent native access violations in numpy/torch; single-threading makes them much rarer. No effect on results."""
import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
