# Benchmark: batched-x4

* Date: 2026-10-09
* Commit: `89784d4` README: batched prompt, baked prefix, prefault, new flags (dirty tree)
* Machine: Apple M1 Pro, 10 cores, 16 GiB, macOS 27.0.1, rustc 1.99.0 (b940084d7 2026-09-28)
* Load average at start: [2.9, 6.38, 8.74]
* Method: `scripts/bench.py` — prompt `'The quick brown fox jumps over the lazy dog because'`, 64 generated tokens, greedy, 5 warm runs (medians), cold = F_NOCACHE copy of the binary run once. Chat = `--chat --system <~60 tokens>`, 16 tokens.

## Startup and time to first token

| binary | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | chat prompt tokens |
|---|---|---|---|---|---|
| tinyllama_q4km | 0.232 | 590 | 65 | 424 | 88 |
| tinyllama_q40 | 0.225 | 546 | 91 | 571 | 88 |
| llama32_1b_q4km | 0.290 | 711 | 66 | 488 | 70 |

## Throughput and memory

| binary | prompt tok/s (warm) | chat prompt tok/s | decode tok/s (median) | decode min–max | peak RSS (MiB) | binary (MiB) | kernels / threads |
|---|---|---|---|---|---|---|---|
| tinyllama_q4km | 200.9 | 207.4 | 115.7 | 62.9–116.5 | 609 | 642.4 | neon+dotprod / 8 |
| tinyllama_q40 | 142.2 | 154.0 | 76.9 | 51.9–112.5 | 580 | 612.6 | neon+dotprod / 8 |
| llama32_1b_q4km | 166.9 | 143.4 | 82.8 | 40.7–93.8 | 773 | 776.4 | neon+dotprod / 8 |

## Cold run detail

| binary | cold startup (ms) | cold prompt tok/s | cold decode tok/s |
|---|---|---|---|
| tinyllama_q4km | 0.651 | 22.0 | 58.7 |
| tinyllama_q40 | 0.656 | 23.8 | 84.7 |
| llama32_1b_q4km | 1.618 | 15.5 | 104.4 |

## Notes

Step B, part 2: 1x4 micro-kernels (one weight row against four activations, block decoded once) for Q4_0/Q8_0/Q4_K/Q6_K on NEON and AVX2, plus 8-row L1 blocking in the batched driver. Results stay bit-identical to the token-by-token path.

Sample output (first warm run): `it is a funny story.

2. The Tale of the Three Little Pigs:
` / `it's a funny poem.

2. "The Grapes of Wrath" by John Steinbe` / `the fox is faster than the dog. This is a classic example of`

Raw data: `03-batched-x4.json`. Reference numbers for Ollama/llama.cpp: `ollama-reference.md`.
