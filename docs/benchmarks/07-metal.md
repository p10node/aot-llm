# Benchmark: metal

* Date: 2026-10-20
* Commit: `c032656` chore: update gitignore and remove target build artifacts (dirty tree)
* Machine: Apple M1 Pro, 10 cores, 16 GiB, macOS 27.0.1, rustc 1.99.1 (58cf8f9c3 2026-10-13)
* Load average at start: [4.55, 15.08, 40.19]
* Method: `scripts/bench.py` — prompt `'The quick brown fox jumps over the lazy dog because'`, 64 generated tokens, greedy, 5 warm runs (medians), cold = F_NOCACHE copy of the binary run once. Chat = `--chat --system <~60 tokens>`, 16 tokens.

## Startup and time to first token

| binary                | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | chat prompt tokens (cached) |
|-----------------------|--------------|-----------------------|-----------------------|-----------------------|-----------------------------|
| tinyllama_q4km        | 0.411        | 646                   | 128                   | 189                   | 88 (66)                     |
| tinyllama_q4km_metal  | 30.083       | 109                   | 69                    | 64                    | 88 (66)                     |
| llama32_1b_q4km       | 0.697        | 744                   | 89                    | 110                   | 70 (55)                     |
| llama32_1b_q4km_metal | 30.004       | 132                   | 66                    | 74                    | 70 (55)                     |

## Throughput and memory

| binary                | prompt tok/s (warm) | chat prompt tok/s | decode tok/s (median) | decode min–max | peak RSS (MiB) | binary (MiB) | kernels / threads        |
|-----------------------|---------------------|-------------------|-----------------------|----------------|----------------|--------------|--------------------------|
| tinyllama_q4km        | 101.8               | 116.4             | 69.6                  | 64.5–81.4      | 644            | 645.4        | neon+dotprod / 8         |
| tinyllama_q4km_metal  | 190.2               | 341.5             | 105.9                 | 103.7–109.1    | 655            | 645.4        | metal (Apple M1 Pro) / 8 |
| llama32_1b_q4km       | 124.2               | 136.5             | 79.3                  | 63.6–82.1      | 773            | 779.9        | neon+dotprod / 8         |
| llama32_1b_q4km_metal | 166.0               | 203.8             | 92.2                  | 92.1–92.3      | 783            | 779.9        | metal (Apple M1 Pro) / 8 |

## Cold run detail

| binary                | cold startup (ms) | cold prompt tok/s | cold decode tok/s |
|-----------------------|-------------------|-------------------|-------------------|
| tinyllama_q4km        | 7.201             | 20.1              | 53.9              |
| tinyllama_q4km_metal  | 836.588           | 119.4             | 108.6             |
| llama32_1b_q4km       | 21.136            | 14.8              | 77.8              |
| llama32_1b_q4km_metal | 607.770           | 83.7              | 92.4              |

## Notes

Step E: Metal GPU backend (`aot-llm compile --gpu metal`). The *_metal binaries run the transformer on the GPU: weights mapped zero-copy from the executable's own section, activations and KV cache shared with the CPU, one command buffer per token with 8 dispatches per layer (RoPE, the SwiGLU product and the residual adds fused into the quantized matrix products), and prompt batches as a simdgroup-matrix GEMM. The other two binaries are the same models compiled without a GPU backend, as a control; all four have the benchmark's system prompt baked in. Startup of the GPU binaries includes Metal initialisation (device, kernels from the OS shader cache, mapping the weights); the first run on a machine also compiles the kernels (~0.3 s) and is not shown. Cold first token for the GPU binaries includes paging the weights in through the GPU mapping. Decode on the GPU is bound by the matrix-vector kernels (75-107 GB/s against llama.cpp's ~115 GB/s on this chip). The CPU control binaries ran while other processes still loaded the machine (15-minute load average 15; their decode is 70-79 tok/s against 113-128 in entries 5-6), so compare the GPU columns with the earlier entries rather than with the control rows of this one.

Sample output (first warm run): `it is a funny story.

2. The Tale of the Three Little Pigs:
` / `it is a funny story.

2. The Gingerbread Man by Kate DiCamil` / `the fox is faster than the dog. This is a classic example of` / `the fox is faster than the dog. The fox is brown, the dog is`

Raw data: `07-metal.json`. Reference numbers for Ollama/llama.cpp: `ollama-reference.md`.
