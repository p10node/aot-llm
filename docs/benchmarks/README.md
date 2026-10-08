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
