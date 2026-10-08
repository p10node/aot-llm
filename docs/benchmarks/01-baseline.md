# Benchmark: baseline

* Date: 2026-10-09
* Commit: `bf6f95b` README with architecture diagram and benchmarks; license files (dirty tree)
* Machine: Apple M1 Pro, 10 cores, 16 GiB, macOS 27.0.1, rustc 1.99.0 (b940084d7 2026-09-28)
* Load average at start: [5.73, 5.19, 5.28]
* Method: `scripts/bench.py` — prompt `'The quick brown fox jumps over the lazy dog because'`, 64 generated tokens, greedy, 5 warm runs (medians), cold = F_NOCACHE copy of the binary run once. Chat = `--chat --system <~60 tokens>`, 16 tokens.

## Startup and time to first token

| binary          | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | chat prompt tokens |
|-----------------|--------------|-----------------------|-----------------------|-----------------------|--------------------|
| tinyllama_q4km  | 0.199        | 652                   | 135                   | 685                   | 88                 |
| tinyllama_q40   | 0.220        | 656                   | 125                   | 1071                  | 88                 |
| llama32_1b_q4km | 0.295        | 1030                  | 121                   | 777                   | 70                 |

## Throughput and memory

| binary          | prompt tok/s (warm) | chat prompt tok/s | decode tok/s (median) | decode min–max | peak RSS (MiB) | binary (MiB) | kernels / threads |
|-----------------|---------------------|-------------------|-----------------------|----------------|----------------|--------------|-------------------|
| tinyllama_q4km  | 96.3                | 128.6             | 103.5                 | 51.6–107.3     | 608            | 642.3        | neon+dotprod / 8  |
| tinyllama_q40   | 104.3               | 82.2              | 73.7                  | 33.0–98.7      | 578            | 612.4        | neon+dotprod / 8  |
| llama32_1b_q4km | 91.2                | 90.1              | 60.7                  | 47.7–66.6      | 772            | 776.3        | neon+dotprod / 8  |

## Cold run detail

| binary          | cold startup (ms) | cold prompt tok/s | cold decode tok/s |
|-----------------|-------------------|-------------------|-------------------|
| tinyllama_q4km  | 1.307             | 19.9              | 79.6              |
| tinyllama_q40   | 0.743             | 19.8              | 71.9              |
| llama32_1b_q4km | 1.502             | 10.7              | 22.0              |

## Notes

Baseline: token-by-token prompt eval, no prefetch, static row split. Machine was running the user's desktop apps (see load average), so decode figures carry ±15% noise.

Sample output (first warm run): `it is a funny story.

2. The Tale of the Three Little Pigs:
` / `it's a funny poem.

2. "The Grapes of Wrath" by John Steinbe` / `the fox is faster than the dog. This is a classic example of`

Raw data: `01-baseline.json`. Reference numbers for Ollama/llama.cpp: `ollama-reference.md`.
