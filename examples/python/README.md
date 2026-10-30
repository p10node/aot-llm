# Python

`aot_llm.py` is a ctypes wrapper over `aot_llm.h`; it needs a shared library:

```sh
aot-llm compile --model Qwen3-0.6B-Q4_K_M.gguf --output ./libqwen3 --lib shared --name qwen3
python3 examples/python/aot_llm.py ./libqwen3/libqwen3.dylib "Why is the sky blue?"
```

```python
from aot_llm import Model
m = Model("./libqwen3/libqwen3.dylib")
print(m.generate("Why is the sky blue?", chat=True, max_tokens=64))
print(m.generate("Say it in five words.", chat=True))      # continues the conversation
m.reset()
print(m.generate("Is the sky blue?", chat=True, choices=["yes", "no"]))
m.save("conversation.kv")                                    # resume later with m.load(...)
```

No dependencies beyond the standard library. Two models in one process: build
each with `--symbol-prefix` and pass `prefix=` to `Model`.
