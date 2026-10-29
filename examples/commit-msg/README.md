# commit-msg: a commit message generator in one binary

The kind of tool aot-llm is for: a process that starts, answers once and exits.
The model, the tokenizer and the KV cache of the instructions are compiled into
one executable, so a commit message costs one process spawn and the evaluation
of the diff, with no server and no model load.

```sh
# 1. build (Qwen3-0.6B-Q4_K_M works well; Llama-3.2-1B-Instruct too)
examples/commit-msg/build.sh ~/models/Qwen3-0.6B-Q4_K_M.gguf ./commit-msg

# 2. try it
git diff --cached | ./commit-msg --chat --system "$(cat examples/commit-msg/system-prompt.txt)" --stop '\n' -n 40 -q

# 3. install as a hook: an empty `git commit` opens the editor with a proposal
cp examples/commit-msg/prepare-commit-msg .git/hooks/ && chmod +x .git/hooks/prepare-commit-msg
```

The system prompt is baked in with `--system`, so passing the same text at run
time (which the hook does) skips its evaluation (69 tokens for Qwen3): the first
token depends only on the size of the diff. The hook feeds at most 4 KB of diff
(about 1 000 tokens); on an M1 Pro with `--gpu metal` that is roughly 2 s of
prompt evaluation plus ~30 ms of process start, and the message itself streams
at about 50 tok/s.
