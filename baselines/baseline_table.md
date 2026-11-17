| dataset | generator | system | variant | n | EM [95% CI] | F1 | doc R@10 | all gold in prompt | peak prompt tok | total tok/q | latency p50 / p95 (s) | calls/q (retr, rerank, comp, gen) | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2wiki | qwen2.5:3b | B1 vanilla RAG (chunk) | - | 100 | 0.35 [0.26, 0.44] | 0.419 | 0.83 | 0.53 | 1443 | 8310 | 0.63 / 0.71 | 1.0, 1.0, 0.0, 1.0 | chunk_2wiki_abdc596e68 |
| 2wiki | qwen2.5:3b | B2 vanilla RAG (sentence) | - | 100 | 0.35 [0.26, 0.45] | 0.416 | 0.75 | 0.29 | 499 | 3002 | 0.36 / 0.63 | 1.0, 1.0, 0.0, 1.0 | sentence_2wiki_ac1a2d0b75 |
| 2wiki | qwen2.5:3b | B3 GeoRAG | - | 100 | 0.28 [0.20, 0.37] | 0.339 | 0.78 | 0.05 | 1564 | 1752 | 4.60 / 6.02 | 9.1, 0.0, 1.0, 1.0 | georag_2wiki_68349c53d9 |
| 2wiki | qwen2.5:3b | B3 GeoRAG | unit=sentence | 100 | 0.26 [0.17, 0.35] | 0.289 | 0.71 | 0.00 | 372 | 911 | 4.59 / 5.87 | 9.1, 0.0, 1.0, 1.0 | georag_2wiki_d7dd56c6e1 |
| 2wiki | qwen2.5:3b | B4 HLG-RAG | - | 100 | 0.36 [0.27, 0.45] | 0.428 | 0.77 | 0.24 | 346 | 4195 | 1.31 / 1.49 | 1.0, 1.0, 1.0, 1.0 | hlg_2wiki_ca340d310f |
| 2wiki | qwen2.5:3b | B4 HLG-RAG | retriever=topic | 100 | 0.37 [0.28, 0.46] | 0.423 | 0.77 | 0.22 | 346 | 5914 | 2.57 / 3.32 | 1.0, 2.0, 1.0, 1.0 | hlg_2wiki_0c2dbbc355 |
| 2wiki | qwen2.5:3b | B5 SURE-RAG | gate=False | 100 | 0.37 [0.28, 0.46] | 0.437 | 0.83 | 0.53 | 1443 | 9121 | 1.09 / 1.39 | 1.0, 1.0, 1.0, 1.0 | surerag_2wiki_9ad1b85c0a |
| 2wiki | qwen2.5:3b | B5 SURE-RAG | gate=False, premise=concat | 100 | 0.37 [0.28, 0.46] | 0.435 | 0.83 | 0.53 | 1443 | 9042 | 1.27 / 1.40 | 1.0, 1.0, 1.0, 1.0 | surerag_2wiki_88571b8855 |
| 2wiki | qwen2.5:3b | B6 EviMem | - | 100 | 0.33 [0.24, 0.42] | 0.419 | 0.77 | 0.57 | 1196 | 2741 | 3.37 / 3.77 | 5.5, 0.0, 1.8, 1.0 | evimem_2wiki_82b289f04a |
| 2wiki | qwen2.5:3b | B6 EviMem | answer_context=raw | 100 | 0.38 [0.29, 0.48] | 0.455 | 0.74 | 0.40 | 1113 | 2540 | 3.42 / 4.20 | 5.5, 0.0, 1.8, 1.0 | evimem_2wiki_4c317ff90e |
| 2wiki | qwen2.5:3b | B7 LycheeMemory V2 | - | 100 | 0.37 [0.28, 0.47] | 0.430 | 0.73 | 0.32 | 1928 | 1567 | 1.89 / 2.47 | 1.3, 0.0, 1.0, 1.0 | lychee_2wiki_31b713ea89 |
| 2wiki | qwen2.5:3b | B7 LycheeMemory V2 | plan=False | 100 | 0.37 [0.28, 0.46] | 0.426 | 0.74 | 0.38 | 1928 | 1172 | 0.47 / 0.70 | 1.0, 0.0, 0.0, 1.0 | lychee_2wiki_d9ff2238e4 |
| 2wiki | qwen2.5:3b | B7b LycheeMemory (ACL 2026, released 7B checkpoint) | timeout_s=3600 | 100 | 0.00 [0.00, 0.00] | 0.144 | 0.00 | 0.00 | 2372 | 1193 | 220.04 / 403.83 | 0.0, 0.0, 0.0, 1.0 | lycheeacl_2wiki_0bda7466e7 |
| 2wiki | qwen2.5:3b | B8 PyRAG | - | 100 | 0.44 [0.34, 0.54] | 0.508 | 0.87 | 0.68 | 4672 | 31295 | 5.56 / 8.13 | 3.6, 3.6, 2.1, 5.3 | pyrag_2wiki_a8030ce3b1 |
| 2wiki | qwen2.5:3b | B8 PyRAG | plan_generator=qwen2.5-coder:3b | 100 | 0.40 [0.31, 0.49] | 0.479 | 0.86 | 0.67 | 4626 | 32288 | 5.20 / 7.89 | 3.8, 3.8, 2.0, 5.6 | pyrag_2wiki_7aafdc3d11 |
| 2wiki | qwen2.5:3b | B9 LiveLedger | - | 100 | 0.27 [0.19, 0.36] | 0.355 | 0.84 | 0.54 | 2754 | 13711 | 4.60 / 10.97 | 3.0, 3.0, 1.0, 4.0 | liveledger_2wiki_974c112e7e |
| 2wiki | qwen2.5:3b | B9 LiveLedger | ledger=False | 100 | 0.29 [0.21, 0.38] | 0.388 | 0.87 | 0.64 | 2463 | 7815 | 1.18 / 3.06 | 2.4, 2.4, 0.0, 3.4 | liveledger_2wiki_02323980dc |
| 2wiki | qwen2.5:3b | ref: iterative refinement | - | 100 | 0.36 [0.27, 0.45] | 0.420 | 0.75 | 0.33 | 767 | 5703 | 1.15 / 3.48 | 2.1, 2.1, 1.0, 1.0 | iterative_2wiki_d75cbcefcb |
| 2wiki | qwen2.5:3b | ref: no retrieval (floor) | - | 100 | 0.20 [0.13, 0.28] | 0.202 | 0.00 | 0.00 | 85 | 72 | 0.04 / 0.06 | 0.0, 0.0, 0.0, 1.0 | none_2wiki_7679ff0061 |
| 2wiki | qwen2.5:7b | B1 vanilla RAG (chunk) | - | 100 | 0.37 [0.28, 0.46] | 0.461 | 0.83 | 0.53 | 1443 | 8311 | 0.58 / 0.69 | 1.0, 1.0, 0.0, 1.0 | chunk_2wiki_1e8ef39c7a |
| 2wiki | qwen2.5:7b | B2 vanilla RAG (sentence) | - | 100 | 0.39 [0.30, 0.49] | 0.455 | 0.75 | 0.29 | 499 | 3003 | 0.52 / 0.89 | 1.0, 1.0, 0.0, 1.0 | sentence_2wiki_7eb0ce0e4e |
| 2wiki | qwen2.5:7b | B3 GeoRAG | - | 100 | 0.23 [0.15, 0.31] | 0.301 | 0.78 | 0.04 | 1564 | 1458 | 4.03 / 4.91 | 10.8, 0.0, 1.0, 1.0 | georag_2wiki_24f766fc74 |
| 2wiki | qwen2.5:7b | B3 GeoRAG | unit=sentence | 100 | 0.26 [0.18, 0.34] | 0.308 | 0.71 | 0.00 | 388 | 610 | 4.06 / 4.90 | 10.6, 0.0, 1.0, 1.0 | georag_2wiki_6ea09457b9 |
| 2wiki | qwen2.5:7b | B4 HLG-RAG | - | 100 | 0.27 [0.19, 0.36] | 0.350 | 0.76 | 0.21 | 369 | 4342 | 1.87 / 2.03 | 1.0, 1.0, 1.0, 1.0 | hlg_2wiki_e1e5fd151b |
| 2wiki | qwen2.5:7b | B4 HLG-RAG | retriever=topic | 100 | 0.27 [0.19, 0.36] | 0.350 | 0.75 | 0.21 | 369 | 11436 | 2.26 / 2.71 | 1.0, 2.0, 1.0, 1.0 | hlg_2wiki_70ab86bfea |
| 2wiki | qwen2.5:7b | B5 SURE-RAG | gate=False | 100 | 0.38 [0.29, 0.47] | 0.466 | 0.83 | 0.53 | 1443 | 9104 | 1.55 / 1.97 | 1.0, 1.0, 1.0, 1.0 | surerag_2wiki_bb3fccf333 |
| 2wiki | qwen2.5:7b | B5 SURE-RAG | gate=False, premise=concat | 100 | 0.38 [0.29, 0.47] | 0.466 | 0.83 | 0.53 | 1443 | 9036 | 1.84 / 2.00 | 1.0, 1.0, 1.0, 1.0 | surerag_2wiki_08b0a060f1 |
| 2wiki | qwen2.5:7b | B6 EviMem | - | 100 | 0.36 [0.27, 0.46] | 0.470 | 0.85 | 0.65 | 1312 | 2803 | 5.31 / 8.15 | 4.3, 0.0, 1.9, 1.0 | evimem_2wiki_f3c1045a0d |
| 2wiki | qwen2.5:7b | B6 EviMem | answer_context=raw | 100 | 0.41 [0.31, 0.51] | 0.520 | 0.85 | 0.52 | 1157 | 2683 | 5.21 / 8.00 | 4.3, 0.0, 1.9, 1.0 | evimem_2wiki_f2c48d736b |
| 2wiki | qwen2.5:7b | B7 LycheeMemory V2 | - | 100 | 0.39 [0.30, 0.48] | 0.468 | 0.72 | 0.43 | 1667 | 1398 | 2.44 / 3.13 | 1.0, 0.0, 1.0, 1.0 | lychee_2wiki_9946152374 |
| 2wiki | qwen2.5:7b | B7 LycheeMemory V2 | plan=False | 100 | 0.43 [0.33, 0.53] | 0.510 | 0.73 | 0.45 | 1695 | 1107 | 0.69 / 1.06 | 1.0, 0.0, 0.0, 1.0 | lychee_2wiki_5fc9a7b175 |
| 2wiki | qwen2.5:7b | B8 PyRAG | - | 100 | 0.37 [0.27, 0.47] | 0.451 | 0.87 | 0.63 | 3043 | 27026 | 7.30 / 13.00 | 3.2, 3.2, 2.0, 4.4 | pyrag_2wiki_c44791bf80 |
| 2wiki | qwen2.5:7b | B8 PyRAG | plan_generator=qwen2.5-coder:7b | 100 | 0.42 [0.32, 0.52] | 0.500 | 0.88 | 0.64 | 3236 | 27298 | 6.61 / 12.33 | 3.2, 3.2, 2.0, 4.5 | pyrag_2wiki_64f650128d |
| 2wiki | qwen2.5:7b | B9 LiveLedger | - | 100 | 0.49 [0.39, 0.59] | 0.602 | 0.95 | 0.79 | 2378 | 16889 | 11.74 / 18.24 | 3.8, 3.8, 1.0, 4.8 | liveledger_2wiki_e4ca67a421 |
| 2wiki | qwen2.5:7b | B9 LiveLedger | ledger=False | 100 | 0.44 [0.34, 0.54] | 0.530 | 0.91 | 0.70 | 1855 | 8674 | 1.84 / 3.52 | 2.8, 2.8, 0.0, 3.8 | liveledger_2wiki_6b02197e52 |
| 2wiki | qwen2.5:7b | ref: iterative refinement | - | 100 | 0.29 [0.21, 0.38] | 0.370 | 0.74 | 0.30 | 797 | 5077 | 2.04 / 2.52 | 1.9, 1.9, 1.0, 1.0 | iterative_2wiki_0993722dcd |
| 2wiki | qwen2.5:7b | ref: no retrieval (floor) | - | 100 | 0.14 [0.08, 0.21] | 0.160 | 0.00 | 0.00 | 85 | 73 | 0.05 / 0.10 | 0.0, 0.0, 0.0, 1.0 | none_2wiki_fd1aa63fca |
| musique | qwen2.5:3b | B1 vanilla RAG (chunk) | - | 100 | 0.09 [0.04, 0.15] | 0.144 | 0.70 | 0.16 | 1286 | 8493 | 0.39 / 0.47 | 1.0, 1.0, 0.0, 1.0 | chunk_musique_df1040a2ff |
| musique | qwen2.5:3b | B2 vanilla RAG (sentence) | - | 100 | 0.14 [0.08, 0.21] | 0.194 | 0.65 | 0.16 | 572 | 3361 | 0.38 / 0.65 | 1.0, 1.0, 0.0, 1.0 | sentence_musique_0d935103be |
| musique | qwen2.5:3b | B3 GeoRAG | - | 100 | 0.05 [0.01, 0.09] | 0.068 | 0.60 | 0.01 | 1266 | 1547 | 4.66 / 5.85 | 8.3, 0.0, 1.0, 1.0 | georag_musique_88953daad9 |
| musique | qwen2.5:3b | B3 GeoRAG | unit=sentence | 100 | 0.05 [0.01, 0.10] | 0.089 | 0.56 | 0.02 | 378 | 947 | 4.53 / 5.75 | 8.1, 0.0, 1.0, 1.0 | georag_musique_4b61181368 |
| musique | qwen2.5:3b | B4 HLG-RAG | - | 100 | 0.11 [0.05, 0.17] | 0.158 | 0.64 | 0.08 | 324 | 5696 | 1.17 / 1.52 | 1.0, 1.0, 1.0, 1.0 | hlg_musique_ec76328fd2 |
| musique | qwen2.5:3b | B4 HLG-RAG | retriever=topic | 100 | 0.10 [0.04, 0.16] | 0.150 | 0.64 | 0.07 | 329 | 6961 | 2.26 / 2.96 | 1.0, 2.0, 1.0, 1.0 | hlg_musique_0c482f9c36 |
| musique | qwen2.5:3b | B5 SURE-RAG | gate=False | 100 | 0.09 [0.04, 0.15] | 0.141 | 0.70 | 0.16 | 1286 | 9369 | 1.16 / 1.44 | 1.0, 1.0, 1.0, 1.0 | surerag_musique_1883594d09 |
| musique | qwen2.5:3b | B5 SURE-RAG | gate=False, premise=concat | 100 | 0.09 [0.04, 0.15] | 0.141 | 0.70 | 0.16 | 1286 | 9263 | 1.30 / 1.40 | 1.0, 1.0, 1.0, 1.0 | surerag_musique_4367c64c2c |
| musique | qwen2.5:3b | B6 EviMem | - | 100 | 0.07 [0.03, 0.12] | 0.146 | 0.62 | 0.33 | 1151 | 2863 | 3.42 / 4.03 | 5.8, 0.0, 1.6, 1.0 | evimem_musique_916f199499 |
| musique | qwen2.5:3b | B6 EviMem | answer_context=raw | 100 | 0.10 [0.05, 0.16] | 0.147 | 0.62 | 0.24 | 1159 | 2695 | 3.52 / 4.32 | 5.8, 0.0, 1.6, 1.0 | evimem_musique_7575dac5d3 |
| musique | qwen2.5:3b | B7 LycheeMemory V2 | - | 100 | 0.13 [0.07, 0.20] | 0.234 | 0.56 | 0.21 | 1652 | 1613 | 2.10 / 2.74 | 1.3, 0.0, 1.0, 1.0 | lychee_musique_7a94fcd277 |
| musique | qwen2.5:3b | B7 LycheeMemory V2 | plan=False | 100 | 0.14 [0.08, 0.21] | 0.264 | 0.58 | 0.29 | 1652 | 1211 | 0.51 / 0.72 | 1.0, 0.0, 0.0, 1.0 | lychee_musique_37a76c435b |
| musique | qwen2.5:3b | B7b LycheeMemory (ACL 2026, released 7B checkpoint) | timeout_s=3600 | 100 | 0.00 [0.00, 0.00] | 0.068 | 0.00 | 0.00 | 3945 | 2894 | 302.92 / 617.62 | 0.0, 0.0, 0.0, 1.0 | lycheeacl_musique_3d1bd68197 |
| musique | qwen2.5:3b | B8 PyRAG | - | 100 | 0.15 [0.08, 0.22] | 0.220 | 0.73 | 0.41 | 4989 | 46028 | 6.52 / 9.85 | 4.9, 4.9, 2.1, 6.6 | pyrag_musique_ed703f84ed |
| musique | qwen2.5:3b | B8 PyRAG | plan_generator=qwen2.5-coder:3b | 100 | 0.10 [0.05, 0.16] | 0.148 | 0.73 | 0.46 | 5304 | 42663 | 5.26 / 8.83 | 4.7, 4.7, 2.0, 6.2 | pyrag_musique_88a2a0d19e |
| musique | qwen2.5:3b | B9 LiveLedger | - | 100 | 0.15 [0.08, 0.22] | 0.210 | 0.68 | 0.26 | 2135 | 17097 | 7.40 / 11.37 | 3.7, 3.7, 1.0, 4.7 | liveledger_musique_28a83496f0 |
| musique | qwen2.5:3b | B9 LiveLedger | ledger=False | 100 | 0.22 [0.14, 0.31] | 0.280 | 0.68 | 0.30 | 2438 | 12267 | 1.63 / 3.24 | 3.5, 3.5, 0.0, 4.5 | liveledger_musique_950ff6e262 |
| musique | qwen2.5:3b | ref: iterative refinement | - | 100 | 0.13 [0.07, 0.20] | 0.211 | 0.65 | 0.21 | 890 | 6351 | 1.28 / 1.62 | 2.2, 2.2, 1.0, 1.0 | iterative_musique_0a6293fd27 |
| musique | qwen2.5:3b | ref: no retrieval (floor) | - | 100 | 0.00 [0.00, 0.00] | 0.024 | 0.00 | 0.00 | 89 | 79 | 0.05 / 0.19 | 0.0, 0.0, 0.0, 1.0 | none_musique_f51860b62f |
| musique | qwen2.5:7b | B1 vanilla RAG (chunk) | - | 100 | 0.15 [0.09, 0.22] | 0.252 | 0.70 | 0.16 | 1286 | 8493 | 0.59 / 0.65 | 1.0, 1.0, 0.0, 1.0 | chunk_musique_5d86dd4bba |
| musique | qwen2.5:7b | B2 vanilla RAG (sentence) | - | 100 | 0.19 [0.12, 0.27] | 0.307 | 0.65 | 0.16 | 572 | 3362 | 0.54 / 0.95 | 1.0, 1.0, 0.0, 1.0 | sentence_musique_89620719f1 |
| musique | qwen2.5:7b | B3 GeoRAG | - | 100 | 0.07 [0.03, 0.12] | 0.133 | 0.59 | 0.01 | 1365 | 1256 | 4.41 / 5.15 | 10.8, 0.0, 1.0, 1.0 | georag_musique_d7fd4e8c2e |
| musique | qwen2.5:7b | B3 GeoRAG | unit=sentence | 100 | 0.09 [0.04, 0.15] | 0.162 | 0.56 | 0.02 | 378 | 631 | 4.19 / 5.06 | 10.7, 0.0, 1.0, 1.0 | georag_musique_b6639ab9a7 |
| musique | qwen2.5:7b | B4 HLG-RAG | - | 100 | 0.17 [0.10, 0.24] | 0.282 | 0.67 | 0.06 | 298 | 5436 | 1.95 / 2.11 | 1.0, 1.0, 1.0, 1.0 | hlg_musique_c501e4b372 |
| musique | qwen2.5:7b | B4 HLG-RAG | retriever=topic | 100 | 0.17 [0.10, 0.24] | 0.277 | 0.66 | 0.06 | 298 | 14368 | 2.60 / 3.19 | 1.0, 2.0, 1.0, 1.0 | hlg_musique_8228debad1 |
| musique | qwen2.5:7b | B5 SURE-RAG | gate=False | 100 | 0.14 [0.08, 0.21] | 0.241 | 0.70 | 0.16 | 1286 | 9340 | 1.59 / 1.91 | 1.0, 1.0, 1.0, 1.0 | surerag_musique_539ac2910e |
| musique | qwen2.5:7b | B5 SURE-RAG | gate=False, premise=concat | 100 | 0.14 [0.08, 0.21] | 0.232 | 0.70 | 0.16 | 1286 | 9254 | 1.89 / 2.00 | 1.0, 1.0, 1.0, 1.0 | surerag_musique_a8a911e1d1 |
| musique | qwen2.5:7b | B6 EviMem | - | 100 | 0.22 [0.14, 0.31] | 0.291 | 0.63 | 0.38 | 1255 | 3254 | 6.98 / 8.58 | 5.2, 0.0, 2.3, 1.0 | evimem_musique_a5b17687df |
| musique | qwen2.5:7b | B6 EviMem | answer_context=raw | 100 | 0.20 [0.13, 0.28] | 0.283 | 0.62 | 0.25 | 1193 | 3165 | 6.36 / 8.23 | 5.2, 0.0, 2.3, 1.0 | evimem_musique_778bec6c37 |
| musique | qwen2.5:7b | B7 LycheeMemory V2 | - | 100 | 0.21 [0.13, 0.29] | 0.344 | 0.58 | 0.28 | 1892 | 1521 | 2.77 / 3.64 | 1.0, 0.0, 1.0, 1.0 | lychee_musique_ff32896580 |
| musique | qwen2.5:7b | B7 LycheeMemory V2 | plan=False | 100 | 0.23 [0.15, 0.31] | 0.360 | 0.59 | 0.28 | 1904 | 1190 | 0.67 / 0.86 | 1.0, 0.0, 0.0, 1.0 | lychee_musique_7da640a3c4 |
| musique | qwen2.5:7b | B8 PyRAG | - | 100 | 0.16 [0.09, 0.23] | 0.265 | 0.76 | 0.53 | 5405 | 38771 | 9.35 / 14.95 | 4.3, 4.3, 2.0, 5.9 | pyrag_musique_cdb729a73a |
| musique | qwen2.5:7b | B8 PyRAG | plan_generator=qwen2.5-coder:7b | 100 | 0.20 [0.12, 0.28] | 0.281 | 0.76 | 0.50 | 4734 | 39260 | 8.80 / 14.51 | 4.4, 4.4, 2.0, 6.0 | pyrag_musique_c4b125dcb9 |
| musique | qwen2.5:7b | B9 LiveLedger | - | 100 | 0.29 [0.20, 0.38] | 0.401 | 0.73 | 0.41 | 3141 | 19816 | 14.67 / 18.55 | 4.4, 4.4, 1.0, 5.4 | liveledger_musique_4412435e9e |
| musique | qwen2.5:7b | B9 LiveLedger | ledger=False | 100 | 0.30 [0.21, 0.39] | 0.416 | 0.75 | 0.45 | 2435 | 12966 | 2.65 / 4.25 | 4.0, 4.0, 0.0, 5.0 | liveledger_musique_f00cb18167 |
| musique | qwen2.5:7b | ref: iterative refinement | - | 100 | 0.18 [0.11, 0.25] | 0.276 | 0.65 | 0.26 | 973 | 6481 | 2.11 / 2.73 | 2.3, 2.3, 1.0, 1.0 | iterative_musique_c08c0f0d74 |
| musique | qwen2.5:7b | ref: no retrieval (floor) | - | 100 | 0.00 [0.00, 0.00] | 0.028 | 0.00 | 0.00 | 89 | 79 | 0.06 / 0.10 | 0.0, 0.0, 0.0, 1.0 | none_musique_2b4457a8b1 |
