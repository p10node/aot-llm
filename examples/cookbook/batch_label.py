#!/usr/bin/env python3
"""Label many lines with one process through the C library (ctypes binding).

    aot-llm compile --model hf://unsloth/Qwen3-0.6B-GGUF:Q4_K_M --output ./libmodel --lib shared --name model
    python3 examples/cookbook/batch_label.py ./libmodel/libmodel.dylib positive,negative,neutral < reviews.txt

The model is opened once; every line is a fresh context (reset) with the
answer constrained to the labels, so the output is one label per line and the
per-line cost is the prompt evaluation only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from aot_llm import Model  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    lib, labels = sys.argv[1], sys.argv[2].split(",")
    m = Model(lib)
    system = f"Classify the text into exactly one of these labels: {', '.join(labels)}."
    n = 0
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        m.reset()
        label = m.generate(line, chat=True, system=system, choices=labels, max_tokens=8)
        print(f"{label}\t{line}")
        n += 1
    print(f"{n} lines", file=sys.stderr)
    m.close()


if __name__ == "__main__":
    main()
