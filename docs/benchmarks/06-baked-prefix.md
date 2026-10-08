# Benchmark: baked-prefix

* Date: 2026-10-09
* Commit: `bbdef96` docs: benchmarks for batched x4, prefault and decode tuning (dirty tree)
* Machine: Apple M1 Pro, 10 cores, 16 GiB, macOS 27.0.1, rustc 1.99.0 (b940084d7 2026-09-28)
* Load average at start: [4.72, 3.87, 3.06]
* Method: `scripts/bench.py` — prompt `'The quick brown fox jumps over the lazy dog because'`, 64 generated tokens, greedy, 5 warm runs (medians), cold = F_NOCACHE copy of the binary run once. Chat = `--chat --system <~60 tokens>`, 16 tokens.

## Startup and time to first token

| binary | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | chat prompt tokens (cached) |
|---|---|---|---|---|---|
| tinyllama_q4km | 0.525 | 624 | 78 | 164 | 88 (66) |
| tinyllama_q40 | 0.438 | 606 | 59 | 95 | 88 (66) |
| llama32_1b_q4km | 0.784 | 721 | 74 | 90 | 70 (55) |

## Throughput and memory

| binary | prompt tok/s (warm) | chat prompt tok/s | decode tok/s (median) | decode min–max | peak RSS (MiB) | binary (MiB) | kernels / threads |
|---|---|---|---|---|---|---|---|
| tinyllama_q4km | 167.0 | 134.2 | 112.1 | 100.3–118.1 | 644 | 645.3 | neon+dotprod / 8 |
| tinyllama_q40 | 219.3 | 232.6 | 135.6 | 114.9–138.3 | 614 | 615.4 | neon+dotprod / 8 |
| llama32_1b_q4km | 148.9 | 167.9 | 92.4 | 74.2–102.9 | 773 | 779.8 | neon+dotprod / 8 |

## Cold run detail

| binary | cold startup (ms) | cold prompt tok/s | cold decode tok/s |
|---|---|---|---|
| tinyllama_q4km | 4.151 | 20.8 | 70.1 |
| tinyllama_q40 | 5.434 | 21.5 | 80.6 |
| llama32_1b_q4km | 24.020 | 15.3 | 98.8 |

## Notes

Step A on top of B+C+D: binaries compiled with --system "$(cat scripts/chat_system.txt)", i.e. the KV cache of the benchmark's own system prompt is baked in. Only the chat columns change: the cached prefix (66 / 66 / 55 tokens) is skipped and just the user turn is evaluated; chat prompt tok/s counts evaluated tokens only. Raw-prompt columns are unaffected and serve as a control.

Sample output (first warm run): `it is a funny story.

2. The Tale of the Three Little Pigs:
` / `it's a funny poem.

2. "The Grapes of Wrath" by John Steinbe` / `the fox is faster than the dog. This is a classic example of`

Raw data: `06-baked-prefix.json`. Reference numbers for Ollama/llama.cpp: `ollama-reference.md`.
