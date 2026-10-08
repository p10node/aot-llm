# Benchmark: prefault

* Date: 2026-10-09
* Commit: `89784d4` README: batched prompt, baked prefix, prefault, new flags (dirty tree)
* Machine: Apple M1 Pro, 10 cores, 16 GiB, macOS 27.0.1, rustc 1.99.0 (b940084d7 2026-09-28)
* Load average at start: [6.15, 6.61, 8.65]
* Method: `scripts/bench.py` — prompt `'The quick brown fox jumps over the lazy dog because'`, 64 generated tokens, greedy, 5 warm runs (medians), cold = F_NOCACHE copy of the binary run once. Chat = `--chat --system <~60 tokens>`, 16 tokens.

## Startup and time to first token

| binary | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | chat prompt tokens |
|---|---|---|---|---|---|
| tinyllama_q4km | 0.473 | 642 | 67 | 375 | 88 |
| tinyllama_q40 | 0.455 | 593 | 71 | 485 | 88 |
| llama32_1b_q4km | 0.738 | 711 | 70 | 302 | 70 |

## Throughput and memory

| binary | prompt tok/s (warm) | chat prompt tok/s | decode tok/s (median) | decode min–max | peak RSS (MiB) | binary (MiB) | kernels / threads |
|---|---|---|---|---|---|---|---|
| tinyllama_q4km | 193.6 | 234.9 | 114.4 | 77.1–115.8 | 644 | 642.5 | neon+dotprod / 8 |
| tinyllama_q40 | 183.6 | 181.4 | 95.7 | 61.2–113.6 | 614 | 612.6 | neon+dotprod / 8 |
| llama32_1b_q4km | 158.7 | 231.5 | 93.0 | 83.5–93.8 | 773 | 776.4 | neon+dotprod / 8 |

## Cold run detail

| binary | cold startup (ms) | cold prompt tok/s | cold decode tok/s |
|---|---|---|---|
| tinyllama_q4km | 5.243 | 20.3 | 47.4 |
| tinyllama_q40 | 6.164 | 21.9 | 109.1 |
| llama32_1b_q4km | 18.611 | 15.5 | 108.4 |

## Notes

Step C: at startup the weights are paged in by background threads (one MADV_WILLNEED + page touch per chunk, one thread per worker) while the main thread tokenizes; --no-prefetch disables it. Only the cold columns are expected to move. Binaries have no baked prefix (control for the chat columns).

Sample output (first warm run): `it is a funny story.

2. The Tale of the Three Little Pigs:
` / `it's a funny poem.

2. "The Grapes of Wrath" by John Steinbe` / `the fox is faster than the dog. This is a classic example of`

Raw data: `04-prefault.json`. Reference numbers for Ollama/llama.cpp: `ollama-reference.md`.
