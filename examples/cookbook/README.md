# Cookbook

Small, finished tools built on one compiled model binary. Every script reads
the binary from `$MODEL` (default `./model`); build one first, for example

```sh
aot-llm compile --model hf://unsloth/Qwen3-0.6B-GGUF:Q4_K_M --output ./model --gpu metal   # drop --gpu off Apple Silicon
export MODEL=./model
```

| script                  | what it shows                                                                  |
|-------------------------|--------------------------------------------------------------------------------|
| `classify.sh`           | `--choices`: a label from a fixed set, nothing to parse. Sentiment / log level  |
| `ask.sh`                | `--stop '\n'`: natural language -> one shell command, confirm before running     |
| `summarize.sh`          | stdin or clipboard -> three bullets; stop string closes the list                 |
| `extract.sh`            | fields -> JSON line per input, one process per line, 30 ms start each            |
| `chat.sh`               | multi-turn conversation across processes with `--save-kv` / `--load-kv`           |
| `batch_label.py`        | the C library from Python: label 1 000 lines in one process (ctypes, no server)  |

The scripts are deliberately short; copy the one closest to your tool and edit
the prompt. `../commit-msg/` is the same idea as a git hook with the
instructions baked into the binary at compile time.

Model size matters more than the prompt: `classify.sh`, `summarize.sh`,
`extract.sh` and `chat.sh` work with a 0.6B model (the examples above were run
on Qwen3-0.6B); `ask.sh` (free-form command synthesis) needs a 1.7B+ model to
be reliable, smaller ones echo the request or invent flags.
