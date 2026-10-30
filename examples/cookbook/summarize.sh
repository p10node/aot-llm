#!/usr/bin/env sh
# Summarize stdin (or the clipboard on macOS) in three bullets.
#
#   pbpaste | examples/cookbook/summarize.sh
#   examples/cookbook/summarize.sh < article.txt
#   examples/cookbook/summarize.sh            # macOS: reads the clipboard
#
# Long inputs: the prompt is evaluated in batches (fast on Metal); the system
# prompt could be baked in with `aot-llm compile --system "..."` to skip it.
set -eu
MODEL=${MODEL:-./model}
if [ -t 0 ] && command -v pbpaste >/dev/null 2>&1; then TEXT=$(pbpaste); else TEXT=$(cat); fi
[ -n "$TEXT" ] || { echo "nothing to summarize" >&2; exit 1; }
printf '%s\n' "$TEXT" | "$MODEL" --chat --system "Summarize the text in exactly three short bullet points starting with '- '. Then stop." --stop '\n\n' -n 160 -q
