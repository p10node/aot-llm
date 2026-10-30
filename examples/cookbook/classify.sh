#!/usr/bin/env sh
# Classify each input line into one of a fixed set of labels.
#
#   echo "The build broke again after the update" | examples/cookbook/classify.sh positive,negative,neutral
#   tail -n 100 app.log | examples/cookbook/classify.sh info,warning,error,critical
#
# `--choices` masks the model's logits to the given answers, so the output is
# always exactly one label: no parsing, no retries.
set -eu
MODEL=${MODEL:-./model}
LABELS=${1:?usage: classify.sh LABEL,LABEL,... < lines}
while IFS= read -r line; do
  [ -n "$line" ] || continue
  label=$("$MODEL" --chat --system "Classify the text into exactly one of these labels: $LABELS." --prompt "$line" --choices "$LABELS" -q)
  printf '%s\t%s\n' "$label" "$line"
done
