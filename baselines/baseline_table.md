| dataset | generator | system | variant | n | primary metric | EM [95% CI] | F1 | doc R@10 | all gold in prompt | peak prompt tok | total tok/q | latency p50 / p95 (s) | calls/q (retr, rerank, comp, gen) | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2wiki | qwen2.5:3b | B1 vanilla RAG (chunk) | - | 100 | **EM 0.39** | 0.39 [0.30, 0.49] | 0.457 | 0.83 | 0.53 | 1443 | 8311 | 0.08 / 0.11 | 1.0, 1.0, 0.0, 1.0 | chunk_2wiki_1b07916eba |
| 2wiki | qwen2.5:3b | B2 vanilla RAG (sentence) | - | 100 | **EM 0.38** | 0.38 [0.29, 0.48] | 0.450 | 0.75 | 0.29 | 499 | 3003 | 0.05 / 0.08 | 1.0, 1.0, 0.0, 1.0 | sentence_2wiki_4018a2faa2 |
| 2wiki | qwen2.5:3b | B3 GeoRAG | - | 100 | **EM 0.28** | 0.28 [0.20, 0.37] | 0.342 | 0.78 | 0.05 | 1564 | 1774 | 2.20 / 3.07 | 9.3, 0.0, 1.0, 1.0 | georag_2wiki_babd0f90fc |
| 2wiki | qwen2.5:3b | B3 GeoRAG | unit=sentence | 100 | **EM 0.27** | 0.27 [0.19, 0.36] | 0.306 | 0.71 | 0.00 | 380 | 933 | 2.19 / 2.99 | 9.4, 0.0, 1.0, 1.0 | georag_2wiki_e803dc74cd |
| 2wiki | qwen2.5:3b | B4 HLG-RAG | - | 100 | **EM 0.32** | 0.32 [0.23, 0.41] | 0.396 | 0.76 | 0.21 | 350 | 3967 | 0.18 / 0.30 | 1.0, 1.0, 1.0, 1.0 | hlg_2wiki_d1c806ce0c |
| 2wiki | qwen2.5:3b | B4 HLG-RAG | retriever=topic | 100 | **EM 0.33** | 0.33 [0.24, 0.42] | 0.405 | 0.72 | 0.19 | 350 | 7501 | 0.58 / 0.96 | 1.0, 2.0, 1.0, 1.0 | hlg_2wiki_26f8488e96 |
| 2wiki | qwen2.5:3b | B5 SURE-RAG | gate=False | 100 | **EM 0.37** | 0.37 [0.28, 0.46] | 0.435 | 0.83 | 0.53 | 1443 | 9125 | 0.17 / 0.24 | 1.0, 1.0, 1.0, 1.0 | surerag_2wiki_e66456703e |
| 2wiki | qwen2.5:3b | B5 SURE-RAG | gate=False, premise=concat | 100 | **EM 0.37** | 0.37 [0.28, 0.46] | 0.440 | 0.83 | 0.53 | 1443 | 9044 | 0.14 / 0.21 | 1.0, 1.0, 1.0, 1.0 | surerag_2wiki_dcc730b302 |
| 2wiki | qwen2.5:3b | B6 EviMem | - | 100 | **EM 0.31** | 0.31 [0.22, 0.40] | 0.396 | 0.78 | 0.62 | 1140 | 2808 | 0.65 / 1.21 | 5.6, 0.0, 1.9, 1.0 | evimem_2wiki_25893c6ff9 |
| 2wiki | qwen2.5:3b | B6 EviMem | answer_context=raw | 100 | **EM 0.34** | 0.34 [0.25, 0.43] | 0.407 | 0.77 | 0.43 | 884 | 2571 | 0.66 / 0.97 | 5.5, 0.0, 1.9, 1.0 | evimem_2wiki_69539ceb9f |
| 2wiki | qwen2.5:3b | B7 LycheeMemory V2 | - | 100 | **EM 0.33** | 0.33 [0.24, 0.43] | 0.398 | 0.72 | 0.34 | 2376 | 1601 | 0.71 / 1.16 | 1.3, 0.0, 1.0, 1.0 | lychee_2wiki_d1958b5787 |
| 2wiki | qwen2.5:3b | B7 LycheeMemory V2 | plan=False | 100 | **EM 0.35** | 0.35 [0.26, 0.44] | 0.425 | 0.74 | 0.38 | 2376 | 1220 | 0.10 / 0.14 | 1.0, 0.0, 0.0, 1.0 | lychee_2wiki_9839fcba70 |
| 2wiki | qwen2.5:3b | B7b LycheeMemory (ACL 2026, released 7B checkpoint) | - | 100 | **EM 0.01** | 0.01 [0.00, 0.03] | 0.140 | 0.00 | 0.00 | 2372 | 1159 | 5.42 / 13.92 | 0.0, 0.0, 0.0, 1.0 | lycheeacl_2wiki_64ff627923 |
| 2wiki | qwen2.5:3b | B8 PyRAG | - | 100 | **EM 0.44** | 0.44 [0.35, 0.54] | 0.517 | 0.89 | 0.71 | 3938 | 30616 | 1.41 / 2.23 | 3.6, 3.6, 2.0, 5.3 | pyrag_2wiki_232015eda4 |
| 2wiki | qwen2.5:3b | B8 PyRAG | plan_generator=qwen2.5-coder:3b | 100 | **EM 0.44** | 0.44 [0.35, 0.54] | 0.512 | 0.88 | 0.73 | 4626 | 31870 | 1.34 / 2.20 | 3.7, 3.7, 2.0, 5.6 | pyrag_2wiki_22393a12a6 |
| 2wiki | qwen2.5:3b | B9 LiveLedger | - | 100 | **EM 0.29** | 0.29 [0.20, 0.38] | 0.353 | 0.81 | 0.45 | 2353 | 13579 | 1.09 / 2.63 | 3.0, 3.0, 1.0, 4.0 | liveledger_2wiki_77eabf942c |
| 2wiki | qwen2.5:3b | B9 LiveLedger | ledger=False | 100 | **EM 0.27** | 0.27 [0.18, 0.36] | 0.376 | 0.84 | 0.60 | 2496 | 7336 | 0.51 / 1.45 | 2.3, 2.3, 0.0, 3.3 | liveledger_2wiki_6f571f59c7 |
| 2wiki | qwen2.5:3b | ref: iterative refinement | - | 100 | **EM 0.33** | 0.33 [0.24, 0.42] | 0.387 | 0.73 | 0.29 | 670 | 5423 | 0.23 / 0.37 | 2.0, 2.0, 1.0, 1.0 | iterative_2wiki_28ce1761cc |
| 2wiki | qwen2.5:3b | ref: no retrieval (floor) | - | 100 | **EM 0.17** | 0.17 [0.10, 0.25] | 0.180 | 0.00 | 0.00 | 85 | 72 | 0.02 / 0.04 | 0.0, 0.0, 0.0, 1.0 | none_2wiki_15af7da1da |
| 2wiki | qwen2.5:7b | B1 vanilla RAG (chunk) | - | 100 | **EM 0.40** | 0.40 [0.31, 0.49] | 0.491 | 0.83 | 0.53 | 1443 | 8311 | 0.12 / 0.18 | 1.0, 1.0, 0.0, 1.0 | chunk_2wiki_4c7fc46094 |
| 2wiki | qwen2.5:7b | B2 vanilla RAG (sentence) | - | 100 | **EM 0.35** | 0.35 [0.26, 0.44] | 0.415 | 0.75 | 0.29 | 499 | 3003 | 0.07 / 0.11 | 1.0, 1.0, 0.0, 1.0 | sentence_2wiki_c7d6692fa9 |
| 2wiki | qwen2.5:7b | B3 GeoRAG | - | 100 | **EM 0.22** | 0.22 [0.14, 0.30] | 0.292 | 0.79 | 0.05 | 1562 | 1454 | 1.88 / 2.43 | 10.8, 0.0, 1.0, 1.0 | georag_2wiki_631bc2f2c2 |
| 2wiki | qwen2.5:7b | B3 GeoRAG | unit=sentence | 100 | **EM 0.26** | 0.26 [0.18, 0.34] | 0.305 | 0.70 | 0.00 | 380 | 607 | 1.76 / 2.38 | 10.8, 0.0, 1.0, 1.0 | georag_2wiki_42e26fca5c |
| 2wiki | qwen2.5:7b | B4 HLG-RAG | - | 100 | **EM 0.29** | 0.29 [0.21, 0.38] | 0.365 | 0.77 | 0.21 | 380 | 4390 | 0.25 / 0.37 | 1.0, 1.0, 1.0, 1.0 | hlg_2wiki_f5e6182bf3 |
| 2wiki | qwen2.5:7b | B4 HLG-RAG | retriever=topic | 100 | **EM 0.30** | 0.30 [0.22, 0.39] | 0.375 | 0.76 | 0.21 | 380 | 11490 | 0.58 / 0.84 | 1.0, 2.0, 1.0, 1.0 | hlg_2wiki_eacc09d82d |
| 2wiki | qwen2.5:7b | B5 SURE-RAG | gate=False | 100 | **EM 0.40** | 0.40 [0.31, 0.49] | 0.486 | 0.83 | 0.53 | 1443 | 9105 | 0.21 / 0.28 | 1.0, 1.0, 1.0, 1.0 | surerag_2wiki_230736ee76 |
| 2wiki | qwen2.5:7b | B5 SURE-RAG | gate=False, premise=concat | 100 | **EM 0.39** | 0.39 [0.30, 0.48] | 0.476 | 0.83 | 0.53 | 1443 | 9037 | 0.17 / 0.22 | 1.0, 1.0, 1.0, 1.0 | surerag_2wiki_032c8b862d |
| 2wiki | qwen2.5:7b | B6 EviMem | - | 100 | **EM 0.39** | 0.39 [0.30, 0.48] | 0.506 | 0.83 | 0.64 | 1331 | 2701 | 1.39 / 2.57 | 4.1, 0.0, 1.9, 1.0 | evimem_2wiki_28a6636dd7 |
| 2wiki | qwen2.5:7b | B6 EviMem | answer_context=raw | 100 | **EM 0.38** | 0.38 [0.29, 0.48] | 0.472 | 0.83 | 0.49 | 1157 | 2606 | 1.25 / 2.29 | 4.1, 0.0, 1.9, 1.0 | evimem_2wiki_8b983b09ad |
| 2wiki | qwen2.5:7b | B7 LycheeMemory V2 | - | 100 | **EM 0.46** | 0.46 [0.36, 0.56] | 0.525 | 0.70 | 0.39 | 1753 | 1418 | 0.68 / 1.05 | 1.0, 0.0, 1.0, 1.0 | lychee_2wiki_dbbabb2c36 |
| 2wiki | qwen2.5:7b | B7 LycheeMemory V2 | plan=False | 100 | **EM 0.43** | 0.43 [0.33, 0.52] | 0.498 | 0.72 | 0.40 | 1753 | 1128 | 0.14 / 0.19 | 1.0, 0.0, 0.0, 1.0 | lychee_2wiki_5d09302d17 |
| 2wiki | qwen2.5:7b | B8 PyRAG | - | 100 | **EM 0.35** | 0.35 [0.26, 0.45] | 0.437 | 0.86 | 0.65 | 3043 | 28571 | 1.92 / 3.77 | 3.3, 3.3, 2.0, 4.6 | pyrag_2wiki_dbc8c9e4f0 |
| 2wiki | qwen2.5:7b | B8 PyRAG | plan_generator=qwen2.5-coder:7b | 100 | **EM 0.45** | 0.45 [0.35, 0.55] | 0.528 | 0.89 | 0.65 | 3215 | 26842 | 1.87 / 3.69 | 3.2, 3.2, 2.0, 4.4 | pyrag_2wiki_4d09bfa47e |
| 2wiki | qwen2.5:7b | B9 LiveLedger | - | 100 | **EM 0.48** | 0.48 [0.39, 0.58] | 0.601 | 0.94 | 0.79 | 2386 | 15613 | 2.57 / 4.66 | 3.5, 3.5, 1.0, 4.5 | liveledger_2wiki_5cc493fd70 |
| 2wiki | qwen2.5:7b | B9 LiveLedger | ledger=False | 100 | **EM 0.41** | 0.41 [0.31, 0.51] | 0.509 | 0.93 | 0.72 | 2332 | 8548 | 0.81 / 1.61 | 2.7, 2.7, 0.0, 3.7 | liveledger_2wiki_1118318e67 |
| 2wiki | qwen2.5:7b | ref: iterative refinement | - | 100 | **EM 0.28** | 0.28 [0.20, 0.37] | 0.363 | 0.74 | 0.30 | 797 | 4946 | 0.29 / 0.52 | 1.9, 1.9, 1.0, 1.0 | iterative_2wiki_6687acd8d7 |
| 2wiki | qwen2.5:7b | ref: no retrieval (floor) | - | 100 | **EM 0.13** | 0.13 [0.07, 0.20] | 0.148 | 0.00 | 0.00 | 85 | 73 | 0.02 / 0.05 | 0.0, 0.0, 0.0, 1.0 | none_2wiki_92f402b721 |
| dsire | qwen2.5:3b | B1 vanilla RAG (chunk) | - | 100 | **F1 0.479** | 0.34 [0.25, 0.43] | 0.479 | 1.00 | 1.00 | 2355 | 20364 | 0.17 / 0.31 | 1.0, 1.0, 0.0, 1.0 | chunk_dsire_aad0e0eba8 |
| dsire | qwen2.5:3b | B2 vanilla RAG (sentence) | - | 100 | **F1 0.412** | 0.32 [0.23, 0.41] | 0.412 | 0.99 | 0.95 | 1191 | 5630 | 0.07 / 0.15 | 1.0, 1.0, 0.0, 1.0 | sentence_dsire_50c49ec1ff |
| dsire | qwen2.5:3b | B3 GeoRAG | - | 100 | **F1 0.432** | 0.31 [0.22, 0.40] | 0.432 | 1.00 | 0.90 | 2403 | 2809 | 2.11 / 2.67 | 8.5, 0.0, 1.0, 1.0 | georag_dsire_c177dc0cb0 |
| dsire | qwen2.5:3b | B3 GeoRAG | unit=sentence | 100 | **F1 0.312** | 0.26 [0.18, 0.35] | 0.312 | 1.00 | 0.95 | 1224 | 1393 | 2.51 / 3.03 | 8.4, 0.0, 1.0, 1.0 | georag_dsire_b1223d48b9 |
| dsire | qwen2.5:3b | B4 HLG-RAG | - | 100 | **F1 0.374** | 0.28 [0.19, 0.37] | 0.374 | 0.99 | 0.99 | 596 | 7120 | 0.25 / 0.59 | 1.0, 1.0, 1.0, 1.0 | hlg_dsire_79578ad41f |
| dsire | qwen2.5:3b | B4 HLG-RAG | retriever=topic | 100 | **F1 0.403** | 0.31 [0.22, 0.40] | 0.403 | 0.95 | 0.94 | 727 | 61333 | 0.83 / 1.67 | 1.0, 2.0, 1.0, 1.0 | hlg_dsire_fc4c3495f3 |
| dsire | qwen2.5:3b | B5 SURE-RAG | gate=False | 100 | **F1 0.445** | 0.33 [0.23, 0.42] | 0.445 | 1.00 | 1.00 | 2355 | 22431 | 0.29 / 0.41 | 1.0, 1.0, 1.0, 1.0 | surerag_dsire_1f12ee6832 |
| dsire | qwen2.5:3b | B5 SURE-RAG | gate=False, premise=concat | 100 | **F1 0.457** | 0.34 [0.25, 0.43] | 0.457 | 1.00 | 1.00 | 2355 | 22308 | 0.25 / 0.39 | 1.0, 1.0, 1.0, 1.0 | surerag_dsire_e01ea52e49 |
| dsire | qwen2.5:3b | B6 EviMem | - | 100 | **F1 0.523** | 0.35 [0.26, 0.45] | 0.523 | 0.97 | 0.99 | 3082 | 3995 | 0.83 / 1.49 | 3.5, 0.0, 1.3, 1.0 | evimem_dsire_e53e9f3402 |
| dsire | qwen2.5:3b | B6 EviMem | answer_context=raw | 100 | **F1 0.493** | 0.32 [0.23, 0.41] | 0.493 | 0.97 | 0.97 | 2419 | 3688 | 0.73 / 1.36 | 3.5, 0.0, 1.3, 1.0 | evimem_dsire_d01fb4d8f6 |
| dsire | qwen2.5:3b | B7 LycheeMemory V2 | - | 100 | **F1 0.369** | 0.29 [0.20, 0.38] | 0.369 | 0.81 | 0.83 | 4174 | 2504 | 0.90 / 1.31 | 1.0, 0.0, 1.0, 1.0 | lychee_dsire_99534ff0bd |
| dsire | qwen2.5:3b | B7 LycheeMemory V2 | plan=False | 100 | **F1 0.381** | 0.30 [0.21, 0.39] | 0.381 | 0.81 | 0.83 | 3984 | 2099 | 0.16 / 0.21 | 1.0, 0.0, 0.0, 1.0 | lychee_dsire_5246214ed7 |
| dsire | qwen2.5:3b | B7b LycheeMemory (ACL 2026, released 7B checkpoint) | - | 100 | **F1 0.263** | 0.01 [0.00, 0.03] | 0.263 | 0.00 | 0.00 | 1775 | 853 | 4.00 / 14.06 | 0.0, 0.0, 0.0, 1.0 | lycheeacl_dsire_dde27f58e2 |
| dsire | qwen2.5:3b | B8 PyRAG | - | 100 | **F1 0.356** | 0.22 [0.14, 0.31] | 0.356 | 0.99 | 0.95 | 8336 | 97495 | 2.30 / 3.99 | 4.6, 4.6, 2.1, 6.0 | pyrag_dsire_56bd29dbe1 |
| dsire | qwen2.5:3b | B8 PyRAG | plan_generator=qwen2.5-coder:3b | 100 | **F1 0.379** | 0.25 [0.17, 0.33] | 0.379 | 0.99 | 0.81 | 9790 | 92252 | 2.09 / 4.11 | 4.3, 4.3, 2.1, 5.6 | pyrag_dsire_a620e01570 |
| dsire | qwen2.5:3b | B9 LiveLedger | - | 100 | **F1 0.402** | 0.27 [0.19, 0.36] | 0.402 | 0.95 | 0.92 | 2617 | 23111 | 1.61 / 3.72 | 3.2, 3.2, 1.0, 4.2 | liveledger_dsire_d9a5875d60 |
| dsire | qwen2.5:3b | B9 LiveLedger | ledger=False | 100 | **F1 0.426** | 0.29 [0.20, 0.38] | 0.426 | 0.98 | 0.94 | 2050 | 12725 | 0.49 / 1.65 | 2.5, 2.5, 0.0, 3.5 | liveledger_dsire_ff24814766 |
| dsire | qwen2.5:3b | ref: iterative refinement | - | 100 | **F1 0.402** | 0.31 [0.22, 0.40] | 0.402 | 0.98 | 0.98 | 2077 | 9940 | 0.36 / 0.51 | 2.2, 2.2, 1.0, 1.0 | iterative_dsire_b3a696f703 |
| dsire | qwen2.5:3b | ref: no retrieval (floor) | - | 100 | **F1 0.085** | 0.08 [0.03, 0.13] | 0.085 | 0.00 | 0.00 | 120 | 84 | 0.02 / 0.03 | 0.0, 0.0, 0.0, 1.0 | none_dsire_6076bd17a5 |
| dsire | qwen2.5:7b | B1 vanilla RAG (chunk) | - | 100 | **F1 0.546** | 0.38 [0.28, 0.48] | 0.546 | 1.00 | 1.00 | 2355 | 20363 | 0.23 / 0.43 | 1.0, 1.0, 0.0, 1.0 | chunk_dsire_5ee536c841 |
| dsire | qwen2.5:7b | B2 vanilla RAG (sentence) | - | 100 | **F1 0.512** | 0.36 [0.27, 0.45] | 0.512 | 0.99 | 0.95 | 1191 | 5631 | 0.10 / 0.21 | 1.0, 1.0, 0.0, 1.0 | sentence_dsire_9b759c7d25 |
| dsire | qwen2.5:7b | B3 GeoRAG | - | 100 | **F1 0.467** | 0.33 [0.24, 0.42] | 0.467 | 1.00 | 0.90 | 2427 | 2408 | 1.52 / 2.23 | 10.2, 0.0, 1.0, 1.0 | georag_dsire_74abf3f27c |
| dsire | qwen2.5:7b | B3 GeoRAG | unit=sentence | 100 | **F1 0.407** | 0.30 [0.21, 0.39] | 0.407 | 1.00 | 0.94 | 1196 | 1003 | 1.86 / 2.43 | 10.1, 0.0, 1.0, 1.0 | georag_dsire_18a819ab2a |
| dsire | qwen2.5:7b | B4 HLG-RAG | - | 100 | **F1 0.520** | 0.36 [0.27, 0.46] | 0.520 | 1.00 | 0.96 | 500 | 7158 | 0.32 / 0.49 | 1.0, 1.0, 1.0, 1.0 | hlg_dsire_fd49517060 |
| dsire | qwen2.5:7b | B4 HLG-RAG | retriever=topic | 100 | **F1 0.515** | 0.36 [0.27, 0.46] | 0.515 | 1.00 | 0.96 | 500 | 21408 | 0.50 / 0.69 | 1.0, 2.0, 1.0, 1.0 | hlg_dsire_7172e04cd7 |
| dsire | qwen2.5:7b | B5 SURE-RAG | gate=False | 100 | **F1 0.547** | 0.38 [0.28, 0.48] | 0.547 | 1.00 | 1.00 | 2355 | 22416 | 0.39 / 0.65 | 1.0, 1.0, 1.0, 1.0 | surerag_dsire_04995d28fd |
| dsire | qwen2.5:7b | B5 SURE-RAG | gate=False, premise=concat | 100 | **F1 0.547** | 0.38 [0.28, 0.48] | 0.547 | 1.00 | 1.00 | 2355 | 22304 | 0.34 / 0.57 | 1.0, 1.0, 1.0, 1.0 | surerag_dsire_ce6494b73e |
| dsire | qwen2.5:7b | B6 EviMem | - | 100 | **F1 0.409** | 0.29 [0.20, 0.38] | 0.409 | 0.99 | 0.99 | 2711 | 3060 | 1.11 / 2.52 | 2.8, 0.0, 1.1, 1.0 | evimem_dsire_35b6498d6d |
| dsire | qwen2.5:7b | B6 EviMem | answer_context=raw | 100 | **F1 0.490** | 0.35 [0.26, 0.45] | 0.490 | 0.99 | 0.99 | 2686 | 3208 | 1.10 / 2.61 | 2.8, 0.0, 1.1, 1.0 | evimem_dsire_682b6854ce |
| dsire | qwen2.5:7b | B7 LycheeMemory V2 | - | 100 | **F1 0.447** | 0.34 [0.25, 0.43] | 0.447 | 0.89 | 0.97 | 3232 | 2363 | 0.98 / 1.86 | 1.0, 0.0, 1.0, 1.0 | lychee_dsire_3d7036ae87 |
| dsire | qwen2.5:7b | B7 LycheeMemory V2 | plan=False | 100 | **F1 0.461** | 0.36 [0.27, 0.45] | 0.461 | 0.89 | 0.98 | 3310 | 2064 | 0.26 / 0.36 | 1.0, 0.0, 0.0, 1.0 | lychee_dsire_eef3115764 |
| dsire | qwen2.5:7b | B8 PyRAG | - | 100 | **F1 0.470** | 0.31 [0.22, 0.41] | 0.470 | 1.00 | 0.99 | 8143 | 77481 | 3.23 / 6.29 | 3.7, 3.7, 2.0, 5.0 | pyrag_dsire_0406e9980e |
| dsire | qwen2.5:7b | B8 PyRAG | plan_generator=qwen2.5-coder:7b | 100 | **F1 0.509** | 0.35 [0.26, 0.45] | 0.509 | 1.00 | 0.98 | 8143 | 76390 | 3.17 / 5.89 | 3.7, 3.7, 2.0, 4.9 | pyrag_dsire_8fc7995da2 |
| dsire | qwen2.5:7b | B9 LiveLedger | - | 100 | **F1 0.471** | 0.33 [0.24, 0.42] | 0.471 | 1.00 | 0.97 | 2818 | 19147 | 2.05 / 6.55 | 2.9, 2.9, 1.0, 3.9 | liveledger_dsire_88c5a82e9e |
| dsire | qwen2.5:7b | B9 LiveLedger | ledger=False | 100 | **F1 0.444** | 0.32 [0.23, 0.41] | 0.444 | 0.99 | 0.95 | 2726 | 13832 | 0.82 / 2.20 | 2.6, 2.6, 0.0, 3.6 | liveledger_dsire_92b14d8422 |
| dsire | qwen2.5:7b | ref: iterative refinement | - | 100 | **F1 0.528** | 0.36 [0.27, 0.46] | 0.528 | 0.97 | 0.96 | 1502 | 8268 | 0.37 / 0.57 | 1.7, 1.7, 1.0, 1.0 | iterative_dsire_4ec1543b9a |
| dsire | qwen2.5:7b | ref: no retrieval (floor) | - | 100 | **F1 0.118** | 0.11 [0.05, 0.17] | 0.118 | 0.00 | 0.00 | 120 | 84 | 0.02 / 0.04 | 0.0, 0.0, 0.0, 1.0 | none_dsire_04313b4467 |
| musique | qwen2.5:3b | B1 vanilla RAG (chunk) | - | 100 | **EM 0.09** | 0.09 [0.04, 0.15] | 0.141 | 0.70 | 0.16 | 1286 | 8493 | 0.08 / 0.11 | 1.0, 1.0, 0.0, 1.0 | chunk_musique_e11612e21e |
| musique | qwen2.5:3b | B2 vanilla RAG (sentence) | - | 100 | **EM 0.14** | 0.14 [0.08, 0.21] | 0.190 | 0.65 | 0.16 | 572 | 3361 | 0.06 / 0.08 | 1.0, 1.0, 0.0, 1.0 | sentence_musique_e2fb329883 |
| musique | qwen2.5:3b | B3 GeoRAG | - | 100 | **EM 0.05** | 0.05 [0.01, 0.10] | 0.073 | 0.59 | 0.01 | 1248 | 1544 | 2.20 / 3.05 | 9.1, 0.0, 1.0, 1.0 | georag_musique_9b79d47841 |
| musique | qwen2.5:3b | B3 GeoRAG | unit=sentence | 100 | **EM 0.05** | 0.05 [0.01, 0.10] | 0.083 | 0.56 | 0.02 | 378 | 938 | 2.23 / 3.00 | 9.0, 0.0, 1.0, 1.0 | georag_musique_06fd277f03 |
| musique | qwen2.5:3b | B4 HLG-RAG | - | 100 | **EM 0.11** | 0.11 [0.05, 0.17] | 0.160 | 0.64 | 0.08 | 329 | 5668 | 0.31 / 0.45 | 1.0, 1.0, 1.0, 1.0 | hlg_musique_fad9811692 |
| musique | qwen2.5:3b | B4 HLG-RAG | retriever=topic | 100 | **EM 0.10** | 0.10 [0.05, 0.16] | 0.139 | 0.61 | 0.07 | 328 | 6987 | 0.79 / 1.23 | 1.0, 2.0, 1.0, 1.0 | hlg_musique_c675804103 |
| musique | qwen2.5:3b | B5 SURE-RAG | gate=False | 100 | **EM 0.08** | 0.08 [0.03, 0.14] | 0.135 | 0.70 | 0.16 | 1286 | 9373 | 0.19 / 0.23 | 1.0, 1.0, 1.0, 1.0 | surerag_musique_cbd4247e59 |
| musique | qwen2.5:3b | B5 SURE-RAG | gate=False, premise=concat | 100 | **EM 0.09** | 0.09 [0.04, 0.15] | 0.145 | 0.70 | 0.16 | 1286 | 9266 | 0.17 / 0.22 | 1.0, 1.0, 1.0, 1.0 | surerag_musique_ad4793b34c |
| musique | qwen2.5:3b | B6 EviMem | - | 100 | **EM 0.06** | 0.06 [0.02, 0.11] | 0.120 | 0.61 | 0.40 | 1015 | 2831 | 0.73 / 1.03 | 5.7, 0.0, 1.6, 1.0 | evimem_musique_82a52ec43a |
| musique | qwen2.5:3b | B6 EviMem | answer_context=raw | 100 | **EM 0.09** | 0.09 [0.04, 0.15] | 0.145 | 0.61 | 0.23 | 874 | 2687 | 0.72 / 1.00 | 5.7, 0.0, 1.6, 1.0 | evimem_musique_7b4df72306 |
| musique | qwen2.5:3b | B7 LycheeMemory V2 | - | 100 | **EM 0.16** | 0.16 [0.09, 0.24] | 0.242 | 0.54 | 0.19 | 2014 | 1589 | 0.72 / 1.07 | 1.1, 0.0, 1.0, 1.0 | lychee_musique_5a73c430f2 |
| musique | qwen2.5:3b | B7 LycheeMemory V2 | plan=False | 100 | **EM 0.16** | 0.16 [0.09, 0.23] | 0.262 | 0.57 | 0.23 | 1825 | 1191 | 0.13 / 0.16 | 1.0, 0.0, 0.0, 1.0 | lychee_musique_aff577a430 |
| musique | qwen2.5:3b | B7b LycheeMemory (ACL 2026, released 7B checkpoint) | - | 100 | **EM 0.00** | 0.00 [0.00, 0.00] | 0.073 | 0.00 | 0.00 | 3945 | 2900 | 8.65 / 23.52 | 0.0, 0.0, 0.0, 1.0 | lycheeacl_musique_51e5e4e7b0 |
| musique | qwen2.5:3b | B8 PyRAG | - | 100 | **EM 0.16** | 0.16 [0.09, 0.24] | 0.237 | 0.73 | 0.46 | 5385 | 43404 | 1.67 / 3.04 | 4.8, 4.8, 2.0, 6.5 | pyrag_musique_fe8aed3e81 |
| musique | qwen2.5:3b | B8 PyRAG | plan_generator=qwen2.5-coder:3b | 100 | **EM 0.10** | 0.10 [0.05, 0.17] | 0.151 | 0.73 | 0.45 | 5891 | 44108 | 1.46 / 2.80 | 4.7, 4.7, 2.0, 6.3 | pyrag_musique_fd14c47e70 |
| musique | qwen2.5:3b | B9 LiveLedger | - | 100 | **EM 0.20** | 0.20 [0.13, 0.28] | 0.277 | 0.66 | 0.26 | 2618 | 17668 | 1.61 / 3.17 | 3.8, 3.8, 1.0, 4.8 | liveledger_musique_c7b1dadec0 |
| musique | qwen2.5:3b | B9 LiveLedger | ledger=False | 100 | **EM 0.16** | 0.16 [0.09, 0.23] | 0.227 | 0.68 | 0.29 | 3059 | 12400 | 0.82 / 1.77 | 3.6, 3.6, 0.0, 4.6 | liveledger_musique_c02d64e22c |
| musique | qwen2.5:3b | ref: iterative refinement | - | 100 | **EM 0.16** | 0.16 [0.09, 0.24] | 0.216 | 0.65 | 0.24 | 1004 | 6711 | 0.31 / 0.47 | 2.3, 2.3, 1.0, 1.0 | iterative_musique_0c025d489b |
| musique | qwen2.5:3b | ref: no retrieval (floor) | - | 100 | **EM 0.00** | 0.00 [0.00, 0.00] | 0.018 | 0.00 | 0.00 | 89 | 78 | 0.02 / 0.03 | 0.0, 0.0, 0.0, 1.0 | none_musique_6f5642e66b |
| musique | qwen2.5:7b | B1 vanilla RAG (chunk) | - | 100 | **EM 0.13** | 0.13 [0.07, 0.20] | 0.236 | 0.70 | 0.16 | 1286 | 8493 | 0.12 / 0.16 | 1.0, 1.0, 0.0, 1.0 | chunk_musique_cbff8bcd4f |
| musique | qwen2.5:7b | B2 vanilla RAG (sentence) | - | 100 | **EM 0.19** | 0.19 [0.12, 0.27] | 0.298 | 0.65 | 0.16 | 572 | 3362 | 0.08 / 0.11 | 1.0, 1.0, 0.0, 1.0 | sentence_musique_05d8af55e7 |
| musique | qwen2.5:7b | B3 GeoRAG | - | 100 | **EM 0.07** | 0.07 [0.03, 0.12] | 0.120 | 0.59 | 0.01 | 1365 | 1249 | 1.92 / 2.46 | 10.7, 0.0, 1.0, 1.0 | georag_musique_f8d2dd85a4 |
| musique | qwen2.5:7b | B3 GeoRAG | unit=sentence | 100 | **EM 0.09** | 0.09 [0.04, 0.15] | 0.164 | 0.56 | 0.02 | 378 | 628 | 1.92 / 2.42 | 10.7, 0.0, 1.0, 1.0 | georag_musique_9d57172de6 |
| musique | qwen2.5:7b | B4 HLG-RAG | - | 100 | **EM 0.17** | 0.17 [0.10, 0.24] | 0.286 | 0.67 | 0.06 | 286 | 5444 | 0.29 / 0.44 | 1.0, 1.0, 1.0, 1.0 | hlg_musique_e978b336eb |
| musique | qwen2.5:7b | B4 HLG-RAG | retriever=topic | 100 | **EM 0.17** | 0.17 [0.10, 0.24] | 0.280 | 0.65 | 0.06 | 286 | 14528 | 0.72 / 1.15 | 1.0, 2.0, 1.0, 1.0 | hlg_musique_6221c745f8 |
| musique | qwen2.5:7b | B5 SURE-RAG | gate=False | 100 | **EM 0.14** | 0.14 [0.08, 0.21] | 0.237 | 0.70 | 0.16 | 1286 | 9338 | 0.23 / 0.29 | 1.0, 1.0, 1.0, 1.0 | surerag_musique_848c056400 |
| musique | qwen2.5:7b | B5 SURE-RAG | gate=False, premise=concat | 100 | **EM 0.15** | 0.15 [0.09, 0.22] | 0.247 | 0.70 | 0.16 | 1286 | 9254 | 0.19 / 0.25 | 1.0, 1.0, 1.0, 1.0 | surerag_musique_a1c47a71a0 |
| musique | qwen2.5:7b | B6 EviMem | - | 100 | **EM 0.21** | 0.21 [0.13, 0.29] | 0.306 | 0.64 | 0.42 | 1101 | 3220 | 1.65 / 2.52 | 5.2, 0.0, 2.2, 1.0 | evimem_musique_7bd9d7a7f3 |
| musique | qwen2.5:7b | B6 EviMem | answer_context=raw | 100 | **EM 0.22** | 0.22 [0.14, 0.30] | 0.306 | 0.64 | 0.28 | 1028 | 3147 | 1.51 / 2.49 | 5.2, 0.0, 2.2, 1.0 | evimem_musique_d8f902601e |
| musique | qwen2.5:7b | B7 LycheeMemory V2 | - | 100 | **EM 0.21** | 0.21 [0.13, 0.29] | 0.350 | 0.57 | 0.28 | 1850 | 1529 | 0.93 / 1.34 | 1.1, 0.0, 1.0, 1.0 | lychee_musique_ea5f761da5 |
| musique | qwen2.5:7b | B7 LycheeMemory V2 | plan=False | 100 | **EM 0.24** | 0.24 [0.16, 0.33] | 0.372 | 0.60 | 0.32 | 1850 | 1191 | 0.17 / 0.21 | 1.0, 0.0, 0.0, 1.0 | lychee_musique_ffa135d3a8 |
| musique | qwen2.5:7b | B8 PyRAG | - | 100 | **EM 0.17** | 0.17 [0.10, 0.25] | 0.267 | 0.74 | 0.48 | 6211 | 40938 | 2.62 / 4.72 | 4.6, 4.6, 2.0, 6.2 | pyrag_musique_cbfb5e6b86 |
| musique | qwen2.5:7b | B8 PyRAG | plan_generator=qwen2.5-coder:7b | 100 | **EM 0.21** | 0.21 [0.14, 0.29] | 0.283 | 0.75 | 0.49 | 4598 | 40349 | 2.58 / 4.47 | 4.6, 4.6, 2.0, 6.2 | pyrag_musique_a83ba7372e |
| musique | qwen2.5:7b | B9 LiveLedger | - | 100 | **EM 0.28** | 0.28 [0.20, 0.37] | 0.399 | 0.78 | 0.51 | 3221 | 19936 | 3.08 / 5.05 | 4.4, 4.4, 1.0, 5.4 | liveledger_musique_28fd22daca |
| musique | qwen2.5:7b | B9 LiveLedger | ledger=False | 100 | **EM 0.28** | 0.28 [0.20, 0.37] | 0.404 | 0.75 | 0.44 | 2440 | 12546 | 1.31 / 2.05 | 3.9, 3.9, 0.0, 4.8 | liveledger_musique_ac3153dc0c |
| musique | qwen2.5:7b | ref: iterative refinement | - | 100 | **EM 0.20** | 0.20 [0.13, 0.28] | 0.309 | 0.64 | 0.25 | 1104 | 6462 | 0.37 / 0.70 | 2.3, 2.3, 1.0, 1.0 | iterative_musique_982ac337ae |
| musique | qwen2.5:7b | ref: no retrieval (floor) | - | 100 | **EM 0.00** | 0.00 [0.00, 0.00] | 0.021 | 0.00 | 0.00 | 89 | 80 | 0.03 / 0.05 | 0.0, 0.0, 0.0, 1.0 | none_musique_00f2d8c0b9 |

**Notes**
- **DSIRE: use F1, not EM.** Gold answers are free text (mean 21 words); 37 of the 100 are plain yes/no, and EM is about 0.92 on those and about 0.06 on the rest, so DSIRE EM is mostly yes/no accuracy. 'primary metric' is F1 for DSIRE and EM for 2Wiki/MuSiQue.
- **B7b (LycheeMemory ACL) is not a RAG row.** The released checkpoint answers in full sentences, so strict EM is about 0 by construction. A lenient 'contains-acc' (gold appears in the raw output) is in each run's `rescore.json` (`python baselines/b7b_lychee_acl/tools/rescore.py`); it must not be put next to EM. It also does not use the qwen2.5 generator named in the generator column.
- **B5 SURE-RAG rows have the gate off**: they are B1's answers plus unfitted verifier scores, not a measurement of SURE-RAG accuracy.
- Gaps below about 4 EM points are inside run-to-run noise (same config, same questions: 0.35-0.39); n=100 gives a 95% CI of about +/- 9 points. See `FIDELITY_AUDIT.md`.
