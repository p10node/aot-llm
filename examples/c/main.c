/*
 * Minimal program against a model library built with `aot-llm compile --lib`.
 *
 *   aot-llm compile --model model.gguf --output ./libmodel --lib
 *   cc -O2 examples/c/main.c -I./libmodel ./libmodel/libmodel.a -o chat      # macOS
 *   cc -O2 examples/c/main.c -I./libmodel ./libmodel/libmodel.a -lpthread -ldl -lm -o chat   # Linux
 *   ./chat "Why is the sky blue?"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "aot_llm.h"

static int print_piece(const char *piece, size_t len, uint32_t token, void *user) {
    (void)token;
    (void)user;
    fwrite(piece, 1, len, stdout);
    fflush(stdout);
    return 0;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const char *prompt = argc > 1 ? argv[1] : "Write one sentence about the moon.";
    double t0 = now_ms();
    aot_model *m = aot_open(NULL);
    if (!m) {
        fprintf(stderr, "cannot open the model: %s\n", aot_last_error());
        return 1;
    }
    aot_model_info info;
    aot_model_info_get(m, &info);
    fprintf(stderr, "%s (%s, %s) on %s, ready in %.1f ms\n", info.name, info.arch, info.file_type, info.backend, now_ms() - t0);

    aot_ctx *c = aot_ctx_new(m, 512, 0);
    if (!c) {
        fprintf(stderr, "cannot create a context: %s\n", aot_last_error());
        return 1;
    }
    aot_generate_params p = aot_generate_default_params();
    p.chat = 1;
    p.system = "You are a concise assistant.";
    p.max_tokens = 96;
    p.sampler.temperature = 0.0f;
    double t1 = now_ms();
    int32_t n = aot_generate(c, prompt, 0, &p, print_piece, NULL);
    double t2 = now_ms();
    printf("\n");
    if (n < 0) {
        fprintf(stderr, "generation failed: %s\n", aot_last_error());
        return 1;
    }
    fprintf(stderr, "%d tokens in %.0f ms\n", n, t2 - t1);

    /* The low-level API: tokens in, logits out. */
    uint32_t ids[64];
    int32_t n_ids = aot_tokenize(m, "The capital of France is", 0, -1, 1, ids, 64);
    aot_ctx_reset(c);
    if (n_ids > 0 && aot_eval(c, ids, (size_t)n_ids, 1) == 0) {
        const float *logits = aot_logits(c);
        uint32_t best = 0;
        for (uint32_t i = 1; i < info.n_vocab; i++)
            if (logits[i] > logits[best]) best = i;
        char piece[64];
        int32_t len = aot_token_piece(m, best, piece, sizeof piece);
        fprintf(stderr, "argmax after \"The capital of France is\": token %u = \"%.*s\"\n", best, len > 0 ? (int)len : 0, piece);
    }
    aot_ctx_free(c);
    aot_close(m);
    return 0;
}
