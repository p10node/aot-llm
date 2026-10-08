# Benchmark: decode-tuning

* Date: 2026-10-09
* Commit: `89784d4` README: batched prompt, baked prefix, prefault, new flags (dirty tree)
* Machine: Apple M1 Pro, 10 cores, 16 GiB, macOS 27.0.1, rustc 1.99.0 (b940084d7 2026-09-28)
* Load average at start: [7.61, 6.87, 8.6]
* Method: `scripts/bench.py` — prompt `'The quick brown fox jumps over the lazy dog because'`, 64 generated tokens, greedy, 5 warm runs (medians), cold = F_NOCACHE copy of the binary run once. Chat = `--chat --system <~60 tokens>`, 16 tokens.

## Startup and time to first token

| binary | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | chat prompt tokens |
|---|---|---|---|---|---|
| tinyllama_q4km | 0.440 | 606 | 68 | 436 | 88 |
| tinyllama_q40 | 0.418 | 580 | 60 | 341 | 88 |
| llama32_1b_q4km | 2.003 | 704 | 64 | 298 | 70 |

## Throughput and memory

| binary | prompt tok/s (warm) | chat prompt tok/s | decode tok/s (median) | decode min–max | peak RSS (MiB) | binary (MiB) | kernels / threads |
|---|---|---|---|---|---|---|---|
| tinyllama_q4km | 190.1 | 201.7 | 127.7 | 124.8–131.1 | 644 | 642.4 | neon+dotprod / 8 |
| tinyllama_q40 | 218.4 | 257.9 | 120.7 | 76.7–142.7 | 614 | 612.6 | neon+dotprod / 8 |
| llama32_1b_q4km | 171.3 | 235.1 | 113.2 | 98.5–113.3 | 773 | 776.4 | neon+dotprod / 8 |

## Cold run detail

| binary | cold startup (ms) | cold prompt tok/s | cold decode tok/s |
|---|---|---|---|
| tinyllama_q4km | 15.465 | 21.5 | 126.2 |
| tinyllama_q40 | 10.055 | 22.4 | 135.3 |
| llama32_1b_q4km | 24.130 | 15.6 | 99.0 |

## Notes

Step D: q/k/v projections fused into one parallel region, rows handed out dynamically in chunks (atomic counter) instead of a static split, half-float scales converted with the fcvt instruction instead of software (one scale per 32 elements for Q4_0/Q8_0). Prefault on, no baked prefix.

Sample output (first warm run): `it is a funny story.

2. The Tale of the Three Little Pigs:
` / `it's a funny poem.

2. "The Grapes of Wrath" by John Steinbe` / `the fox is faster than the dog. This is a classic example of`

Raw data: `05-decode-tuning.json`. Reference numbers for Ollama/llama.cpp: `ollama-reference.md`.
