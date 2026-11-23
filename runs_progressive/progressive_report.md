# Progressive results: score vs number of questions

One full run per configuration; the first N questions of that run form the N-question experiment (nested sets, so systems are compared on the same questions). Steps appear once all N questions are answered. Primary metric: **EM** for 2Wiki/MuSiQue, **token-F1** for DSIRE (free-text answers). Cells: `value [95% bootstrap CI]`.

*Caveat:* the corpus is pooled over all questions of the 600-question file, so retrieval difficulty is the same at every N (the curve shows sampling stability, not a growing corpus). The first 100 questions are the set used in the 100-question table, but its numbers are not identical to it (bigger pool, fresh run).

## 2wiki | qwen2.5:3b | primary metric: EM | 500 questions in the set

**EM (primary)**

| system | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| B1 vanilla RAG (chunk) | 0.350 [0.260, 0.450] | 0.320 [0.255, 0.390] | 0.320 [0.267, 0.373] | 0.315 [0.270, 0.357] | 0.322 [0.282, 0.366] |
| B2 vanilla RAG (sentence) | 0.360 [0.260, 0.450] | 0.345 [0.275, 0.420] | 0.343 [0.293, 0.393] | 0.335 [0.292, 0.380] | 0.328 [0.286, 0.368] |
| B3 GeoRAG | 0.250 [0.160, 0.340] | 0.235 [0.185, 0.290] | 0.237 [0.190, 0.283] | 0.233 [0.190, 0.278] | 0.242 [0.208, 0.280] |
| B3 GeoRAG [unit=sentence] | 0.280 [0.190, 0.370] | 0.235 [0.180, 0.290] | 0.237 [0.190, 0.287] | 0.223 [0.185, 0.265] | 0.224 [0.186, 0.260] |
| B5 SURE-RAG [gate=False] | 0.350 [0.260, 0.450] | 0.310 [0.255, 0.375] | 0.323 [0.273, 0.377] | 0.315 [0.270, 0.362] | 0.322 [0.282, 0.366] |
| B5 SURE-RAG [gate=False, premise=concat] | 0.350 [0.260, 0.450] | 0.310 [0.250, 0.380] | 0.323 [0.267, 0.377] | 0.315 [0.268, 0.362] | 0.322 [0.280, 0.364] |
| B6 EviMem | 0.330 [0.240, 0.420] | 0.325 [0.260, 0.395] | 0.323 [0.267, 0.373] | 0.310 [0.265, 0.355] | 0.304 [0.264, 0.346] |
| B6 EviMem [answer_context=raw] | 0.340 [0.250, 0.440] | 0.335 [0.270, 0.400] | 0.343 [0.290, 0.397] | 0.325 [0.280, 0.372] | 0.324 [0.284, 0.362] |
| B8 PyRAG | 0.370 [0.280, 0.470] | 0.365 [0.300, 0.430] | 0.363 [0.310, 0.417] | 0.335 [0.290, 0.380] | 0.346 [0.302, 0.392] |
| B8 PyRAG [plan_generator=qwen2.5-coder:3b] | 0.360 [0.270, 0.450] | 0.350 [0.285, 0.415] | 0.363 [0.310, 0.420] | 0.333 [0.287, 0.378] | 0.348 [0.306, 0.392] |
| B9 LiveLedger | 0.310 [0.220, 0.400] | 0.285 [0.220, 0.355] | - | - | - |
| ref: iterative refinement | 0.340 [0.260, 0.430] | 0.275 [0.215, 0.335] | 0.283 [0.233, 0.337] | 0.280 [0.235, 0.323] | 0.294 [0.256, 0.336] |
| ref: no retrieval (floor) | 0.170 [0.100, 0.250] | 0.160 [0.115, 0.210] | 0.160 [0.120, 0.203] | 0.150 [0.117, 0.185] | 0.150 [0.120, 0.180] |

**paired difference vs B1 chunk RAG in EM (* = interval excludes 0)**

| system | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| B1 vanilla RAG (chunk) | - | - | - | - | - |
| B2 vanilla RAG (sentence) | +0.010 [-0.080, +0.100] | +0.025 [-0.035, +0.085] | +0.023 [-0.027, +0.070] | +0.020 [-0.022, +0.062] | +0.006 [-0.026, +0.042] |
| B3 GeoRAG | -0.100 [-0.190, -0.010]* | -0.085 [-0.155, -0.010]* | -0.083 [-0.137, -0.030]* | -0.083 [-0.130, -0.035]* | -0.080 [-0.122, -0.038]* |
| B3 GeoRAG [unit=sentence] | -0.070 [-0.160, +0.020] | -0.085 [-0.155, -0.015]* | -0.083 [-0.137, -0.027]* | -0.092 [-0.142, -0.048]* | -0.098 [-0.138, -0.060]* |
| B5 SURE-RAG [gate=False] | +0.000 [-0.040, +0.040] | -0.010 [-0.040, +0.020] | +0.003 [-0.023, +0.027] | +0.000 [-0.020, +0.020] | +0.000 [-0.016, +0.016] |
| B5 SURE-RAG [gate=False, premise=concat] | +0.000 [-0.040, +0.040] | -0.010 [-0.040, +0.015] | +0.003 [-0.023, +0.030] | +0.000 [-0.022, +0.022] | +0.000 [-0.016, +0.018] |
| B6 EviMem | -0.020 [-0.110, +0.080] | +0.005 [-0.070, +0.080] | +0.003 [-0.057, +0.060] | -0.005 [-0.060, +0.048] | -0.018 [-0.062, +0.022] |
| B6 EviMem [answer_context=raw] | -0.010 [-0.110, +0.090] | +0.015 [-0.050, +0.085] | +0.023 [-0.033, +0.080] | +0.010 [-0.035, +0.055] | +0.002 [-0.042, +0.046] |
| B8 PyRAG | +0.020 [-0.080, +0.120] | +0.045 [-0.025, +0.115] | +0.043 [-0.013, +0.103] | +0.020 [-0.030, +0.070] | +0.024 [-0.024, +0.070] |
| B8 PyRAG [plan_generator=qwen2.5-coder:3b] | +0.010 [-0.090, +0.120] | +0.030 [-0.050, +0.110] | +0.043 [-0.013, +0.103] | +0.018 [-0.033, +0.070] | +0.026 [-0.016, +0.072] |
| B9 LiveLedger | -0.040 [-0.140, +0.060] | -0.035 [-0.105, +0.035] | - | - | - |
| ref: iterative refinement | -0.010 [-0.090, +0.080] | -0.045 [-0.110, +0.015] | -0.037 [-0.083, +0.010] | -0.035 [-0.078, +0.005] | -0.028 [-0.068, +0.010] |
| ref: no retrieval (floor) | -0.180 [-0.260, -0.100]* | -0.160 [-0.225, -0.095]* | -0.160 [-0.210, -0.110]* | -0.165 [-0.210, -0.122]* | -0.172 [-0.212, -0.132]* |

**F1**

| system | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| B1 vanilla RAG (chunk) | 0.416 | 0.401 | 0.396 | 0.390 | 0.392 |
| B2 vanilla RAG (sentence) | 0.438 | 0.427 | 0.412 | 0.398 | 0.386 |
| B3 GeoRAG | 0.299 | 0.298 | 0.294 | 0.287 | 0.289 |
| B3 GeoRAG [unit=sentence] | 0.316 | 0.282 | 0.282 | 0.267 | 0.262 |
| B5 SURE-RAG [gate=False] | 0.425 | 0.395 | 0.402 | 0.391 | 0.395 |
| B5 SURE-RAG [gate=False, premise=concat] | 0.425 | 0.395 | 0.402 | 0.391 | 0.395 |
| B6 EviMem | 0.414 | 0.418 | 0.411 | 0.392 | 0.379 |
| B6 EviMem [answer_context=raw] | 0.435 | 0.440 | 0.430 | 0.412 | 0.400 |
| B8 PyRAG | 0.434 | 0.447 | 0.449 | 0.420 | 0.421 |
| B8 PyRAG [plan_generator=qwen2.5-coder:3b] | 0.448 | 0.426 | 0.438 | 0.411 | 0.422 |
| B9 LiveLedger | 0.398 | 0.374 | - | - | - |
| ref: iterative refinement | 0.403 | 0.347 | 0.352 | 0.344 | 0.354 |
| ref: no retrieval (floor) | 0.180 | 0.165 | 0.163 | 0.155 | 0.154 |

**doc recall@10**

| system | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| B1 vanilla RAG (chunk) | 0.792 | 0.799 | 0.780 | 0.772 | 0.773 |
| B2 vanilla RAG (sentence) | 0.735 | 0.744 | 0.729 | 0.719 | 0.716 |
| B3 GeoRAG | 0.743 | 0.741 | 0.733 | 0.726 | 0.727 |
| B3 GeoRAG [unit=sentence] | 0.695 | 0.686 | 0.675 | 0.664 | 0.660 |
| B5 SURE-RAG [gate=False] | 0.792 | 0.799 | 0.780 | 0.772 | 0.773 |
| B5 SURE-RAG [gate=False, premise=concat] | 0.792 | 0.799 | 0.780 | 0.772 | 0.773 |
| B6 EviMem | 0.752 | 0.759 | 0.752 | 0.746 | 0.741 |
| B6 EviMem [answer_context=raw] | 0.752 | 0.759 | 0.752 | 0.746 | 0.741 |
| B8 PyRAG | 0.843 | 0.841 | 0.843 | 0.844 | 0.843 |
| B8 PyRAG [plan_generator=qwen2.5-coder:3b] | 0.843 | 0.835 | 0.843 | 0.838 | 0.839 |
| B9 LiveLedger | 0.818 | 0.795 | - | - | - |
| ref: iterative refinement | 0.725 | 0.729 | 0.710 | 0.704 | 0.705 |
| ref: no retrieval (floor) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Notes: B7b (LycheeMemory ACL) answers in sentences, so its strict EM/F1 are about 0 by construction; B5 SURE-RAG has its gate off (B1's answers). 'doc recall@10' is empty for systems without retrieval. See `baselines/FIDELITY_AUDIT.md`.
