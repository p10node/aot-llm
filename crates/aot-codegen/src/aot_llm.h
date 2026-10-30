/*
 * aot_llm.h - C interface of a model library produced by `aot-llm compile --lib`.
 *
 * The library contains one model: its weights, tokenizer and kernels. Nothing
 * is loaded at run time; `aot_open` maps the embedded weights (or a sidecar
 * file) and starts the worker threads. All functions are safe to call from
 * any thread; evaluations of different contexts are serialised internally.
 *
 *   aot_model *m = aot_open(NULL);
 *   aot_ctx *c = aot_ctx_new(m, 0, 0);
 *   aot_generate_params p = aot_generate_default_params();
 *   p.chat = 1; p.max_tokens = 64;
 *   aot_generate(c, "Why is the sky blue?", 0, &p, print_piece, NULL);
 *   aot_ctx_free(c); aot_close(m);
 *
 * Link: `cc app.c libmodel.a` on macOS, `cc app.c libmodel.a -lpthread -ldl -lm`
 * on Linux. Only one model library can be linked into a program (the symbols
 * are not prefixed by model name).
 */
#ifndef AOT_LLM_H
#define AOT_LLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* AOT_COMMON_BEGIN: shared by every model library linked into a program. */
#ifndef AOT_LLM_COMMON_H
#define AOT_LLM_COMMON_H
enum aot_device {
    AOT_DEVICE_AUTO = 0, /* GPU when the library was built with --gpu and one exists, else CPU */
    AOT_DEVICE_CPU = 1,
    AOT_DEVICE_GPU = 2   /* fail if the GPU is unavailable */
};
#endif
/* AOT_COMMON_END */

/* ---------------------------------------------------------------- model --- */

/* The compiled model: weights, tokenizer, thread pool. */
typedef struct aot_model aot_model;

typedef struct aot_model_params {
    int n_threads;            /* worker threads, 0 = min(cores, 8) */
    int device;               /* enum aot_device */
    int prefault;             /* 1: page the weights in from background threads right away */
    const char *weights_path; /* sidecar builds: path of the weights file, NULL = <library dir>/<name>.weights */
    const char *kernels;      /* CPU kernel set override ("scalar", "neon", "avx2", ...), NULL = auto */
} aot_model_params;

aot_model_params aot_model_default_params(void);

/* Open the model. Returns NULL on failure (see aot_last_error). May be called
 * more than once; every call returns the same model until aot_close. */
aot_model *aot_open(const aot_model_params *params);
void aot_close(aot_model *m);

/* Message of the last failure on the calling thread ("" when none). */
const char *aot_last_error(void);

typedef struct aot_model_info {
    const char *name;
    const char *arch;        /* "llama", "qwen2", "qwen3" */
    const char *file_type;   /* e.g. "MOSTLY_Q4_K_M" */
    const char *backend;     /* active backend: kernel set name or "metal (<device>)" */
    const char *compiler;    /* aot-llm version */
    uint64_t n_params;
    uint32_t n_vocab;
    uint32_t n_ctx_train;
    uint32_t n_layer;
    uint32_t dim;
    uint32_t n_prefix_tokens; /* tokens of the compile-time KV prefix, 0 when none */
} aot_model_info;

void aot_model_info_get(const aot_model *m, aot_model_info *out);

/* ------------------------------------------------------------ tokenizer --- */

/* Tokenize `text` (`text_len` bytes, or NUL-terminated when text_len == 0).
 * Writes up to `cap` ids and returns the number of ids the text produces
 * (which may exceed `cap`), or -1 on error. */
int32_t aot_tokenize(const aot_model *m, const char *text, size_t text_len, int add_bos, int parse_special, uint32_t *out, size_t cap);

/* Decode ids to UTF-8 bytes (no terminating NUL). Returns the number of
 * bytes needed; writes up to `cap`. Control tokens produce no text. */
int32_t aot_detokenize(const aot_model *m, const uint32_t *ids, size_t n, char *out, size_t cap);

/* Bytes of one token (same conventions as aot_detokenize). */
int32_t aot_token_piece(const aot_model *m, uint32_t id, char *out, size_t cap);

/* 1 when `id` ends a reply (EOS / EOT / ...). */
int aot_token_is_stop(const aot_model *m, uint32_t id);

/* Wrap a user message (and optional system prompt, NULL for none) in the
 * model's chat template, ending with the assistant header. Returns the
 * number of bytes needed (no NUL), -1 when the model has no known template. */
int32_t aot_chat_prompt(const aot_model *m, const char *system, const char *user, char *out, size_t cap);

/* The text that continues a conversation already in a context: closes the
 * last reply, opens the next user turn and the assistant header. */
int32_t aot_chat_continuation(const aot_model *m, const char *user, char *out, size_t cap);

/* -------------------------------------------------------------- context --- */

/* One sequence: activations and KV cache for up to n_ctx tokens. */
typedef struct aot_ctx aot_ctx;

/* n_ctx 0 = the model's training context; n_batch 0 = 64 prompt tokens per
 * step. Returns NULL on failure. */
aot_ctx *aot_ctx_new(aot_model *m, uint32_t n_ctx, uint32_t n_batch);
void aot_ctx_free(aot_ctx *c);

/* Forget every token (the KV cache is reused). */
void aot_ctx_reset(aot_ctx *c);
uint32_t aot_ctx_n_past(const aot_ctx *c); /* tokens evaluated so far */
uint32_t aot_ctx_n_ctx(const aot_ctx *c);

/* Append `n` tokens to the sequence and run them through the model. With
 * want_logits the logits of the last token become available through
 * aot_logits. When the context is empty and the tokens start with the
 * compile-time KV prefix, that part is loaded from the binary instead of
 * being computed. Returns 0, or -1 on error (context full, bad token id). */
int aot_eval(aot_ctx *c, const uint32_t *tokens, size_t n, int want_logits);

/* Logits of the last evaluated token (n_vocab floats), valid until the next
 * aot_eval / aot_sample on this context. */
const float *aot_logits(const aot_ctx *c);

/* Save the sequence (tokens and KV rows) to a file, and restore one into an
 * empty context: a conversation can continue in a later process. Loading
 * re-evaluates only the last token, so aot_logits / aot_sample are valid
 * right after it. The file is specific to this model. Returns 0 or -1. */
int aot_ctx_save(const aot_ctx *c, const char *path);
int aot_ctx_load(aot_ctx *c, const char *path);

/* ------------------------------------------------------------- sampling --- */

typedef struct aot_sampler_params {
    float temperature;    /* 0 = greedy */
    int top_k;            /* 0 = off */
    float top_p;          /* 1 = off */
    float repeat_penalty; /* 1 = off */
    int repeat_last_n;    /* window of the repeat penalty */
    uint64_t seed;
} aot_sampler_params;

aot_sampler_params aot_sampler_default_params(void);

typedef struct aot_sampler aot_sampler;
aot_sampler *aot_sampler_new(const aot_sampler_params *params);
void aot_sampler_free(aot_sampler *s);

/* Pick the next token from the context's current logits (modifies them). The
 * repeat penalty looks at the tokens evaluated in the context. */
uint32_t aot_sample(aot_sampler *s, aot_ctx *c);

/* ----------------------------------------------------------- generation --- */

/* Called for every generated token with its text; return non-zero to stop. */
typedef int (*aot_token_cb)(const char *piece, size_t len, uint32_t token, void *user);

typedef struct aot_generate_params {
    int chat;                   /* 1: wrap the prompt in the chat template */
    const char *system;         /* system prompt for chat (NULL = none; ignored when continuing) */
    int max_tokens;             /* 0 = 128 */
    int add_bos;                /* -1 = model default */
    aot_sampler_params sampler;
    const char *const *stop;    /* stop strings: generation ends before the first one that appears */
    size_t n_stop;
    const char *const *choices; /* answer constraint: the output is exactly one of these strings */
    size_t n_choices;
} aot_generate_params;

aot_generate_params aot_generate_default_params(void);

/* Tokenize the prompt (optionally chat-wrapped), append it to the context,
 * evaluate it and generate tokens until a stop token, a stop string, a
 * complete choice, max_tokens, or the callback asks to stop. On a context
 * that already holds tokens the prompt continues the conversation (chat:
 * as the next user turn); call aot_ctx_reset for a fresh one. Returns the
 * number of generated tokens, or -1 on error. */
int32_t aot_generate(aot_ctx *c, const char *prompt, size_t prompt_len, const aot_generate_params *params, aot_token_cb cb, void *user);

#ifdef __cplusplus
}
#endif
#endif /* AOT_LLM_H */
