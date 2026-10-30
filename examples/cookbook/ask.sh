#!/usr/bin/env sh
# Natural language -> one shell command; shows it and asks before running.
#
#   examples/cookbook/ask.sh "list the 5 largest files under this directory"
#
# `--stop '\n'` ends the reply at the first newline, so the model cannot
# append an explanation; the stop string itself is never printed.
set -eu
MODEL=${MODEL:-./model}
Q=${*:?usage: ask.sh "what you want to do"}
SYSTEM="You are a $(uname -s) shell expert. The user describes a task; answer with one $(basename "${SHELL:-sh}") command that does exactly that task, on one line, nothing else: no explanation, no code fences, no echo of the request."
CMD=$("$MODEL" --chat --system "$SYSTEM" --prompt "Task: $Q" --stop '\n' --stop '```' -n 80 -q | sed -e 's/^ *//' -e 's/ *$//' -e 's/^Command: *//')
[ -n "$CMD" ] || { echo "no command produced" >&2; exit 1; }
printf '\033[1m%s\033[0m\n' "$CMD"
printf 'run? [y/N] '
read -r ans
case "$ans" in y|Y|yes) eval "$CMD" ;; *) echo "not run" ;; esac
