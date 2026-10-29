#!/usr/bin/env sh
# Compile a commit-message generator: one binary with the model, the
# tokenizer and the KV cache of the instructions already baked in.
#
#   examples/commit-msg/build.sh path/to/Qwen3-0.6B-Q4_K_M.gguf [./commit-msg]
set -eu
MODEL=${1:?usage: build.sh MODEL.gguf [OUTPUT]}
OUT=${2:-./commit-msg}
HERE=$(cd "$(dirname "$0")" && pwd)
SYSTEM=$(cat "$HERE/system-prompt.txt")
GPU=""
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then GPU="--gpu metal"; fi
# shellcheck disable=SC2086
exec "${AOT_LLM:-aot-llm}" compile --model "$MODEL" --output "$OUT" --system "$SYSTEM" $GPU --quiet
