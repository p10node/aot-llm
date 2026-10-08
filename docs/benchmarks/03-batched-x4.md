# Benchmark: batched-x4

* Date: 2026-10-09
* Commit: `703bdf4` Batched prompt processing with 1x4 micro-kernels (dirty tree)
* Machine: Apple M1 Pro, 10 cores, 16 GiB, macOS 27.0.1, rustc 1.99.0 (b940084d7 2026-09-28)
* Load average at start: [3.53, 9.05, 9.58]
* Method: `scripts/bench.py` — prompt `'The quick brown fox jumps over the lazy dog because'`, 64 generated tokens, greedy, 5 warm runs (medians), cold = F_NOCACHE copy of the binary run once. Chat = `--chat --system <~60 tokens>`, 16 tokens.

## Startup and time to first token

| binary | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | chat prompt tokens |
|---|---|---|---|---|---|
| tinyllama_q4km | 0.223 | 636 | 127 | 568 | 88 |
| tinyllama_q40 | 0.233 | 597 | 77 | 473 | 88 |
| llama32_1b_q4km | 0.301 | 756 | 117 | 379 | 70 |

## Throughput and memory

| binary | prompt tok/s (warm) | chat prompt tok/s | decode tok/s (median) | decode min–max | peak RSS (MiB) | binary (MiB) | kernels / threads |
|---|---|---|---|---|---|---|---|
| tinyllama_q4km | 102.4 | 154.9 | 51.4 | 20.5–73.7 | 609 | 642.4 | neon+dotprod / 8 |
| tinyllama_q40 | 168.2 | 186.0 | 66.9 | 43.7–110.0 | 580 | 612.6 | neon+dotprod / 8 |
| llama32_1b_q4km | 94.3 | 185.0 | 51.8 | 30.6–58.2 | 773 | 776.4 | neon+dotprod / 8 |

## Cold run detail

| binary | cold startup (ms) | cold prompt tok/s | cold decode tok/s |
|---|---|---|---|
| tinyllama_q4km | 0.655 | 20.5 | 42.6 |
| tinyllama_q40 | 0.707 | 21.8 | 89.6 |
| llama32_1b_q4km | 1.492 | 14.6 | 19.2 |

## Notes

Step B, part 2: 1x4 micro-kernels (one weight row against four activations, block decoded once) for Q4_0/Q8_0/Q4_K/Q6_K on NEON and AVX2, plus 8-row L1 blocking in the batched driver. Results stay bit-identical to the token-by-token path.

Sample output (first warm run): `it is a funny story.

2. The Tale of the Three Little Pigs:
` / `it's a funny poem.

2. "The Grapes of Wrath" by John Steinbe` / `the fox is faster than the dog. This is a classic example of`

Raw data: `03-batched-x4.json`. Reference numbers for Ollama/llama.cpp: `ollama-reference.md`.
