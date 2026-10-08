# Benchmark history

One file per measurement, newest last. Every entry is produced by `scripts/bench.py run` on the same machine with the same prompts; see each file for the exact method. `ollama-reference.md` holds the llama.cpp/Ollama numbers the baseline was compared against.

Throughput columns show the median of the warm runs and, in parentheses, the best run; the best run is the better estimate when other processes were competing for the CPU (see the load average in each file).

| # | label | commit | binary | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | chat prompt tok/s | decode tok/s | peak RSS (MiB) | load |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [baseline](01-baseline.md) | `bf6f95b` | tinyllama_q4km | 0.199 | 652 | 135 | 685 (-) | 128.6 (-) | 103.5 (107.3) | 608 | 5.73 |
| 1 | [baseline](01-baseline.md) | `bf6f95b` | tinyllama_q40 | 0.220 | 656 | 125 | 1071 (-) | 82.2 (-) | 73.7 (98.7) | 578 | 5.73 |
| 1 | [baseline](01-baseline.md) | `bf6f95b` | llama32_1b_q4km | 0.295 | 1030 | 121 | 777 (-) | 90.1 (-) | 60.7 (66.6) | 772 | 5.73 |
| 2 | [batched-prompt](02-batched-prompt.md) | `8c36d40` | tinyllama_q4km | 0.231 | 584 | 91 | 567 (-) | 155.3 (-) | 112.4 (114.6) | 609 | 7.21 |
| 2 | [batched-prompt](02-batched-prompt.md) | `8c36d40` | tinyllama_q40 | 0.235 | 545 | 114 | 752 (-) | 117.1 (-) | 111.6 (112.3) | 580 | 7.21 |
| 2 | [batched-prompt](02-batched-prompt.md) | `8c36d40` | llama32_1b_q4km | 0.324 | 719 | 114 | 583 (-) | 120.1 (-) | 61.2 (81.8) | 773 | 7.21 |
| 3 | [batched-x4](03-batched-x4.md) | `89784d4` | tinyllama_q4km | 0.232 | 590 | 65 | 424 (351) | 207.4 (250.6) | 115.7 (116.5) | 609 | 2.9 |
| 3 | [batched-x4](03-batched-x4.md) | `89784d4` | tinyllama_q40 | 0.225 | 546 | 91 | 571 (469) | 154.0 (187.7) | 76.9 (112.5) | 580 | 2.9 |
| 3 | [batched-x4](03-batched-x4.md) | `89784d4` | llama32_1b_q4km | 0.290 | 711 | 66 | 488 (283) | 143.4 (247.8) | 82.8 (93.8) | 773 | 2.9 |
| 4 | [prefault](04-prefault.md) | `89784d4` | tinyllama_q4km | 0.473 | 642 | 67 | 375 (362) | 234.9 (243.0) | 114.4 (115.8) | 644 | 6.15 |
| 4 | [prefault](04-prefault.md) | `89784d4` | tinyllama_q40 | 0.455 | 593 | 71 | 485 (373) | 181.4 (235.8) | 95.7 (113.6) | 614 | 6.15 |
| 4 | [prefault](04-prefault.md) | `89784d4` | llama32_1b_q4km | 0.738 | 711 | 70 | 302 (286) | 231.5 (244.8) | 93.0 (93.8) | 773 | 6.15 |
| 5 | [decode-tuning](05-decode-tuning.md) | `89784d4` | tinyllama_q4km | 0.440 | 606 | 68 | 436 (355) | 201.7 (247.8) | 127.7 (131.1) | 644 | 7.61 |
| 5 | [decode-tuning](05-decode-tuning.md) | `89784d4` | tinyllama_q40 | 0.418 | 580 | 60 | 341 (296) | 257.9 (296.9) | 120.7 (142.7) | 614 | 7.61 |
| 5 | [decode-tuning](05-decode-tuning.md) | `89784d4` | llama32_1b_q4km | 2.003 | 704 | 64 | 298 (280) | 235.1 (250.2) | 113.2 (113.3) | 773 | 7.61 |
