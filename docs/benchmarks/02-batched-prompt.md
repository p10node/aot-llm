# Benchmark: batched-prompt

* Date: 2026-10-09
* Commit: `8c36d40` docs: benchmark harness and baseline measurements (dirty tree)
* Machine: Apple M1 Pro, 10 cores, 16 GiB, macOS 27.0.1, rustc 1.99.0 (b940084d7 2026-09-28)
* Load average at start: [7.21, 7.73, 7.4]
* Method: `scripts/bench.py` — prompt `'The quick brown fox jumps over the lazy dog because'`, 64 generated tokens, greedy, 5 warm runs (medians), cold = F_NOCACHE copy of the binary run once. Chat = `--chat --system <~60 tokens>`, 16 tokens.

## Startup and time to first token

| binary | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | chat prompt tokens |
|---|---|---|---|---|---|
| tinyllama_q4km | 0.231 | 584 | 91 | 567 | 88 |
| tinyllama_q40 | 0.235 | 545 | 114 | 752 | 88 |
| llama32_1b_q4km | 0.324 | 719 | 114 | 583 | 70 |

## Throughput and memory

| binary | prompt tok/s (warm) | chat prompt tok/s | decode tok/s (median) | decode min–max | peak RSS (MiB) | binary (MiB) | kernels / threads |
|---|---|---|---|---|---|---|---|
| tinyllama_q4km | 142.6 | 155.3 | 112.4 | 95.3–114.6 | 609 | 642.4 | neon+dotprod / 8 |
| tinyllama_q40 | 114.1 | 117.1 | 111.6 | 56.4–112.3 | 580 | 612.5 | neon+dotprod / 8 |
| llama32_1b_q4km | 96.4 | 120.1 | 61.2 | 55.5–81.8 | 773 | 776.4 | neon+dotprod / 8 |

## Cold run detail

| binary | cold startup (ms) | cold prompt tok/s | cold decode tok/s |
|---|---|---|---|
| tinyllama_q4km | 0.699 | 22.3 | 51.0 |
| tinyllama_q40 | 0.871 | 23.8 | 96.3 |
| llama32_1b_q4km | 1.821 | 15.3 | 102.1 |

## Notes

Step B: prompt tokens are processed in batches of up to 64 (`--batch`); each weight row is read once per batch instead of once per token. Decode path unchanged. The batched and token-by-token paths are bit-identical (rope table codegen pinned with #[inline(never)]).

Sample output (first warm run): `it is a funny story.

2. The Tale of the Three Little Pigs:
` / `it's a funny poem.

2. "The Grapes of Wrath" by John Steinbe` / `the fox is faster than the dog. This is a classic example of`

Raw data: `02-batched-prompt.json`. Reference numbers for Ollama/llama.cpp: `ollama-reference.md`.
