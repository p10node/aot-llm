# Reference: Ollama 0.34.4 (llama.cpp backend), same GGUF files

Measured 2026-10-09 on the baseline machine (Apple M1 Pro, 8P+2E cores, 16 GiB). The GGUFs compiled by
aot-llm were imported unchanged with `ollama create <name> -f Modelfile` (`FROM <file>.gguf`) and driven
through `POST /api/generate` with `raw: true`, `temperature: 0`, `num_predict: 64`, the same prompt as
`scripts/bench.py` ("The quick brown fox jumps over the lazy dog because"). "Cold" = request issued right
after `ollama stop <model>`; "warm" = model already resident. CPU mode = `num_gpu: 0, num_thread: 8`.

## Latency

| model | mode | cold: model load | cold: total request (64 tok) | warm: total request (64 tok) |
|---|---|---|---|---|
| TinyLlama-1.1B-Chat Q4_K_M | CPU, 8 threads | 1.57 s | 2.33 s | 0.58 s |
| TinyLlama-1.1B-Chat Q4_K_M | Metal GPU | 0.54 s | 1.34 s | 0.47 s |
| Llama-3.2-1B-Instruct Q4_K_M | CPU, 8 threads | 2.84 s | 4.61 s | 0.78 s |
| Llama-3.2-1B-Instruct Q4_K_M | Metal GPU | 0.83 s | 1.48 s | 0.50 s |

The server process itself (`ollama serve`) was already running; these numbers do not include starting it.

## Throughput (warm)

| model | mode | prompt eval tok/s | decode tok/s |
|---|---|---|---|
| TinyLlama-1.1B-Chat Q4_K_M | CPU, 8 threads | 510 | 117 |
| TinyLlama-1.1B-Chat Q4_K_M | Metal GPU | 340 | 150 |
| Llama-3.2-1B-Instruct Q4_K_M | CPU, 8 threads | 412 | 86 |
| Llama-3.2-1B-Instruct Q4_K_M | Metal GPU | 388 | 139 |

## Memory (RSS of `llama-server` while the model is resident)

| model | CPU mode | GPU mode |
|---|---|---|
| TinyLlama-1.1B-Chat Q4_K_M | 798 MiB | 766 MiB |
| Llama-3.2-1B-Instruct Q4_K_M | 1586 MiB | 1420 MiB |

Plus `ollama serve` (22–27 MiB) and the menu-bar app (12–20 MiB).

## Output equivalence

Greedy generations of both models from aot-llm binaries match these Ollama runs token for token
(e.g. `The capital of France is` → ` Paris.\n\n2. B.C. The capital of ancient Rome was Rome.` for TinyLlama).
