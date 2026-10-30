"""ctypes binding for a model library built with `aot-llm compile --lib shared`.

    from aot_llm import Model
    m = Model("./libqwen3/libqwen3.dylib")
    print(m.generate("Why is the sky blue?", chat=True, max_tokens=64))
    for piece in m.stream("Tell me a joke.", chat=True):
        print(piece, end="", flush=True)

The functions mirror aot_llm.h; a library built with --symbol-prefix is
opened with Model(path, prefix="qwen3_").
"""
import ctypes
import ctypes.util
import sys
from typing import Callable, Iterator, Optional, Sequence

AOT_DEVICE_AUTO, AOT_DEVICE_CPU, AOT_DEVICE_GPU = 0, 1, 2


class ModelParams(ctypes.Structure):
    _fields_ = [
        ("n_threads", ctypes.c_int),
        ("device", ctypes.c_int),
        ("prefault", ctypes.c_int),
        ("weights_path", ctypes.c_char_p),
        ("kernels", ctypes.c_char_p),
    ]


class ModelInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("arch", ctypes.c_char_p),
        ("file_type", ctypes.c_char_p),
        ("backend", ctypes.c_char_p),
        ("compiler", ctypes.c_char_p),
        ("n_params", ctypes.c_uint64),
        ("n_vocab", ctypes.c_uint32),
        ("n_ctx_train", ctypes.c_uint32),
        ("n_layer", ctypes.c_uint32),
        ("dim", ctypes.c_uint32),
        ("n_prefix_tokens", ctypes.c_uint32),
    ]


class SamplerParams(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("repeat_last_n", ctypes.c_int),
        ("seed", ctypes.c_uint64),
    ]


class GenerateParams(ctypes.Structure):
    _fields_ = [
        ("chat", ctypes.c_int),
        ("system", ctypes.c_char_p),
        ("max_tokens", ctypes.c_int),
        ("add_bos", ctypes.c_int),
        ("sampler", SamplerParams),
        ("stop", ctypes.POINTER(ctypes.c_char_p)),
        ("n_stop", ctypes.c_size_t),
        ("choices", ctypes.POINTER(ctypes.c_char_p)),
        ("n_choices", ctypes.c_size_t),
    ]


TOKEN_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_void_p)


class Model:
    """One compiled model (the library is the model)."""

    def __init__(self, path: str, prefix: str = "", threads: int = 0, device: int = AOT_DEVICE_AUTO):
        self.lib = ctypes.CDLL(path)
        self._p = prefix
        f = self._fn
        f("model_default_params", [], ModelParams)
        f("open", [ctypes.POINTER(ModelParams)], ctypes.c_void_p)
        f("close", [ctypes.c_void_p], None)
        f("last_error", [], ctypes.c_char_p)
        f("model_info_get", [ctypes.c_void_p, ctypes.POINTER(ModelInfo)], None)
        f("tokenize", [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t], ctypes.c_int32)
        f("detokenize", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int32)
        f("chat_prompt", [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int32)
        f("ctx_new", [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32], ctypes.c_void_p)
        f("ctx_free", [ctypes.c_void_p], None)
        f("ctx_reset", [ctypes.c_void_p], None)
        f("ctx_n_past", [ctypes.c_void_p], ctypes.c_uint32)
        f("ctx_save", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int)
        f("ctx_load", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int)
        f("eval", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.c_int], ctypes.c_int)
        f("logits", [ctypes.c_void_p], ctypes.POINTER(ctypes.c_float))
        f("generate_default_params", [], GenerateParams)
        f("generate", [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(GenerateParams), TOKEN_CB, ctypes.c_void_p], ctypes.c_int32)
        params = self._c("model_default_params")()
        params.n_threads = threads
        params.device = device
        self.model = self._c("open")(ctypes.byref(params))
        if not self.model:
            raise RuntimeError(self.error())
        self.ctx = None

    def _fn(self, name, argtypes, restype):
        fn = getattr(self.lib, f"{self._p}aot_{name}")
        fn.argtypes = argtypes
        fn.restype = restype

    def _c(self, name):
        return getattr(self.lib, f"{self._p}aot_{name}")

    def error(self) -> str:
        return (self._c("last_error")() or b"").decode(errors="replace")

    def info(self) -> dict:
        i = ModelInfo()
        self._c("model_info_get")(self.model, ctypes.byref(i))
        return {k: (getattr(i, k).decode() if isinstance(getattr(i, k), bytes) else getattr(i, k)) for k, _ in ModelInfo._fields_}

    def tokenize(self, text: str, add_bos: Optional[bool] = None, parse_special: bool = True) -> list:
        b = text.encode()
        n = self._c("tokenize")(self.model, b, len(b), -1 if add_bos is None else int(add_bos), int(parse_special), None, 0)
        if n < 0:
            raise RuntimeError(self.error())
        buf = (ctypes.c_uint32 * n)()
        self._c("tokenize")(self.model, b, len(b), -1 if add_bos is None else int(add_bos), int(parse_special), buf, n)
        return list(buf)

    def detokenize(self, ids: Sequence[int]) -> str:
        arr = (ctypes.c_uint32 * len(ids))(*ids)
        n = self._c("detokenize")(self.model, arr, len(ids), None, 0)
        buf = ctypes.create_string_buffer(max(n, 1))
        self._c("detokenize")(self.model, arr, len(ids), buf, n)
        return buf.raw[:n].decode(errors="replace")

    def context(self, n_ctx: int = 0, n_batch: int = 0):
        """Create (or return) the context used by generate / stream."""
        if self.ctx is None:
            self.ctx = self._c("ctx_new")(self.model, n_ctx, n_batch)
            if not self.ctx:
                raise RuntimeError(self.error())
        return self.ctx

    def reset(self):
        if self.ctx:
            self._c("ctx_reset")(self.ctx)

    def save(self, path: str):
        if self._c("ctx_save")(self.context(), path.encode()) != 0:
            raise RuntimeError(self.error())

    def load(self, path: str):
        if self._c("ctx_load")(self.context(), path.encode()) != 0:
            raise RuntimeError(self.error())

    def _params(self, chat, system, max_tokens, temperature, stop, choices, seed):
        p = self._c("generate_default_params")()
        p.chat = int(chat)
        p.system = system.encode() if system else None
        p.max_tokens = max_tokens
        p.sampler.temperature = temperature
        p.sampler.seed = seed
        keep = []
        if stop:
            arr = (ctypes.c_char_p * len(stop))(*[s.encode() for s in stop])
            p.stop, p.n_stop = arr, len(stop)
            keep.append(arr)
        if choices:
            arr = (ctypes.c_char_p * len(choices))(*[s.encode() for s in choices])
            p.choices, p.n_choices = arr, len(choices)
            keep.append(arr)
        return p, keep

    def stream(self, prompt: str, chat: bool = False, system: Optional[str] = None, max_tokens: int = 128, temperature: float = 0.0,
               stop: Optional[Sequence[str]] = None, choices: Optional[Sequence[str]] = None, seed: int = 42) -> Iterator[str]:
        """Generate and yield text pieces. A second call continues the conversation; reset() starts over."""
        pieces = []

        def cb(piece, length, token, user):
            pieces.append(ctypes.string_at(piece, length).decode(errors="replace"))
            return 0

        p, keep = self._params(chat, system, max_tokens, temperature, stop, choices, seed)
        b = prompt.encode()
        n = self._c("generate")(self.context(), b, len(b), ctypes.byref(p), TOKEN_CB(cb), None)
        if n < 0:
            raise RuntimeError(self.error())
        yield from pieces

    def generate(self, prompt: str, **kw) -> str:
        return "".join(self.stream(prompt, **kw))

    def close(self):
        if self.ctx:
            self._c("ctx_free")(self.ctx)
            self.ctx = None
        if self.model:
            self._c("close")(self.model)
            self.model = None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 aot_llm.py path/to/libmodel.{dylib,so} [prompt]")
        sys.exit(2)
    m = Model(sys.argv[1])
    print(m.info(), file=sys.stderr)
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Why is the sky blue?"
    for piece in m.stream(prompt, chat=True, max_tokens=96):
        print(piece, end="", flush=True)
    print()
    print("yes/no:", m.generate("Is water wet?", chat=True, choices=["yes", "no"]) if m.reset() is None else "")
    m.close()
