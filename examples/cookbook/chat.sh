#!/usr/bin/env sh
# A multi-turn chat where every turn is a separate process.
#
#   examples/cookbook/chat.sh            # type messages, Ctrl-D to quit
#
# Each process loads the conversation's KV cache from the previous one
# (`--load-kv`), evaluates only the new message and saves the cache again
# (`--save-kv`), so turn N costs the same as turn 1. Delete the file to start
# over. This is the "no server" multi-turn story: state lives in a file.
set -eu
MODEL=${MODEL:-./model}
KV=${KV:-/tmp/aot-chat.kv}
SYSTEM=${SYSTEM:-"You are a concise assistant."}
rm -f "$KV"
while printf '> ' && IFS= read -r msg; do
  [ -n "$msg" ] || continue
  if [ -f "$KV" ]; then
    "$MODEL" --chat --prompt "$msg" --load-kv "$KV" --save-kv "$KV" -n 256 -q
  else
    "$MODEL" --chat --system "$SYSTEM" --prompt "$msg" --save-kv "$KV" -n 256 -q
  fi
done
echo
