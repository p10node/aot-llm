# Benchmark history

One file per measurement, newest last. Every entry is produced by `scripts/bench.py run` on the same machine with the same prompts; see each file for the exact method. `ollama-reference.md` holds the llama.cpp/Ollama numbers the baseline was compared against.

| # | label | commit | binary | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | prompt tok/s | decode tok/s | peak RSS (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [baseline](01-baseline.md) | `bf6f95b` | tinyllama_q4km | 0.199 | 652 | 135 | 685 | 96.3 | 103.5 | 608 |
| 1 | [baseline](01-baseline.md) | `bf6f95b` | tinyllama_q40 | 0.220 | 656 | 125 | 1071 | 104.3 | 73.7 | 578 |
| 1 | [baseline](01-baseline.md) | `bf6f95b` | llama32_1b_q4km | 0.295 | 1030 | 121 | 777 | 91.2 | 60.7 | 772 |
