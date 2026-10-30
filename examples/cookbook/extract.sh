#!/usr/bin/env sh
# Pull fields out of free text, one JSON object per input line.
#
#   printf 'Meeting with Ana on 2026-11-03 at 14:00 about the Q4 budget\n' | examples/cookbook/extract.sh person,date,time,topic
#
# One process per line: the model starts in ~1 ms warm, so this is fine for a
# few hundred lines; for thousands use batch_label.py (one process).
set -eu
MODEL=${MODEL:-./model}
FIELDS=${1:?usage: extract.sh field,field,... < lines}
SYSTEM="Extract the fields $FIELDS from the text and answer with one JSON object on a single line with exactly those keys (use null when unknown). No explanation."
while IFS= read -r line; do
  [ -n "$line" ] || continue
  # '{' primes the JSON; the stop string ends it at the first closing brace.
  obj=$("$MODEL" --chat --system "$SYSTEM" --prompt "$line" --stop '}' -n 120 -q)
  printf '%s}\n' "$obj"
done
