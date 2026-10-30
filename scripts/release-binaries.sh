#!/usr/bin/env bash
# Build ready-to-run model binaries for the release and (optionally) upload
# them to a Hugging Face model repository.
#
#   scripts/release-binaries.sh build            # -> dist/<model>-<target>
#   ONLY=phi-3 TARGETS=x86_64-unknown-linux-musl scripts/release-binaries.sh build
#   scripts/release-binaries.sh upload p10node/aot-llm-binaries
#
# Models are fetched from the Hub (hf:// references, cached in
# ~/.cache/aot-llm). Each model is built for the host (Apple Silicon with the
# Metal backend) and for x86_64 Linux as a static musl executable. Needs the
# `x86_64-unknown-linux-musl` target (`rustup target add ...`) and, for
# upload, `hf auth login` (https://huggingface.co/docs/huggingface_hub/guides/cli).
set -euo pipefail
cd "$(dirname "$0")/.."
AOT=${AOT_LLM:-./target/release/aot-llm}
DIST=${DIST:-./dist}
TARGETS=${TARGETS:-"host x86_64-unknown-linux-musl"}
ONLY=${ONLY:-}   # build only the models whose name matches this pattern

# name | hf reference
MODELS=(
  "qwen3-0.6b          hf://unsloth/Qwen3-0.6B-GGUF:Q4_K_M"
  "qwen3-1.7b          hf://unsloth/Qwen3-1.7B-GGUF:Q4_K_M"
  "qwen2.5-1.5b        hf://Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m"
  "llama-3.2-1b        hf://bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"
  "llama-3.2-3b        hf://bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
  "gemma-3-1b          hf://unsloth/gemma-3-1b-it-GGUF:Q4_K_M"
  "phi-3-mini-4k       hf://microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf"
)

host_triple() {
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) echo aarch64-apple-darwin ;;
    Darwin-x86_64) echo x86_64-apple-darwin ;;
    Linux-x86_64) echo x86_64-unknown-linux-gnu ;;
    Linux-aarch64) echo aarch64-unknown-linux-gnu ;;
    *) echo unknown ;;
  esac
}

build() {
  [ -x "$AOT" ] || cargo build --release
  mkdir -p "$DIST"
  for entry in "${MODELS[@]}"; do
    name=$(echo "$entry" | awk '{print $1}')
    ref=$(echo "$entry" | awk '{print $2}')
    if [ -n "$ONLY" ] && [[ "$name" != *$ONLY* ]]; then continue; fi
    for target in $TARGETS; do
      if [ "$target" = host ]; then
        triple=$(host_triple)
        extra=""
        if [ "$triple" = aarch64-apple-darwin ]; then extra="--gpu metal"; fi
        out="$DIST/$name-$triple"
        # shellcheck disable=SC2086
        "$AOT" compile --model "$ref" --output "$out" --build-dir "$DIST/build-$name-$triple" --name "$(echo "$name" | tr '.-' '__')" $extra --quiet
      else
        out="$DIST/$name-$target"
        "$AOT" compile --model "$ref" --output "$out" --build-dir "$DIST/build-$name-$target" --name "$(echo "$name" | tr '.-' '__')" --target "$target" --quiet
      fi
      rm -rf "$DIST/build-$name-"*
      ls -la "$out" | awk '{print $5, $9}'
    done
  done
  echo "binaries in $DIST"
}

upload() {
  repo=${1:?usage: release-binaries.sh upload ORG/REPO}
  hf auth whoami >/dev/null || { echo "run: hf auth login" >&2; exit 1; }
  hf repo create "$repo" --type model 2>/dev/null || true
  cat > "$DIST/README.md" <<'MD'
---
license: apache-2.0
tags: [aot-llm, gguf, llama, qwen, gemma, phi]
---

# aot-llm prebuilt model binaries

Each file is **one executable with the model inside**, built with
[aot-llm](https://github.com/p10node/aot-llm) from the GGUF named in the
file. No runtime, no model download, nothing to install:

```sh
curl -LO https://huggingface.co/REPO/resolve/main/qwen3-0.6b-aarch64-apple-darwin
chmod +x qwen3-0.6b-aarch64-apple-darwin
./qwen3-0.6b-aarch64-apple-darwin --chat --prompt "Why is the sky blue?"
./qwen3-0.6b-aarch64-apple-darwin --help
```

`*-aarch64-apple-darwin` binaries use the Metal GPU when present (`--device cpu`
to force the CPU); `*-x86_64-unknown-linux-musl` are static Linux executables
(AVX2 when available). Quantization: Q4_K_M for every model. The models keep
their own licenses (Qwen: Apache-2.0, Llama 3.2: Llama 3.2 Community License,
Gemma: Gemma Terms of Use, Phi-3: MIT).

To build your own, with a baked system prompt or as a C library:
`aot-llm compile --model hf://org/repo:Q4_K_M --output ./model --system "..."`.
MD
  sed -i.bak "s|REPO|$repo|" "$DIST/README.md" && rm -f "$DIST/README.md.bak"
  hf upload "$repo" "$DIST" . --exclude "build-*" --commit-message "aot-llm binaries $(git describe --tags --always)"
}

case "${1:-build}" in
  build) build ;;
  upload) upload "${2:-}" ;;
  *) echo "usage: $0 build | upload ORG/REPO" >&2; exit 2 ;;
esac
