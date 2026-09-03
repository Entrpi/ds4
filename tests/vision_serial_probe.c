/* Track B inc 3+4 gate (a): token identity of an image prompt, same source for
 * the fork and the upstream oracle.  Opens BASE + SIDECAR on CUDA, encodes IMAGE,
 * builds a user turn "TEXT_BEFORE <image> TEXT_AFTER", disables thinking, syncs
 * the serial session (passing the image span on every sync), and greedy-decodes
 * N tokens.  Prints one JSON object: prompt ids, the span, a sha256 of the
 * scattered block rows, the continuation.  Only API shared by both trees is
 * used; the identity line is fork-only (guarded on DS4_VISION_PREPROCESS_VERSION).
 * Run with DS4_CUDA_FP8_KV=0 on the fork (visual attention needs F32 comp KV). */
#include "ds4.h"
#include "ds4_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void hex(const uint8_t *b, size_t n, char *out) { for (size_t i = 0; i < n; i++) sprintf(out + 2 * i, "%02x", b[i]); out[2 * n] = 0; }

int main(int argc, char **argv) {
    if (argc < 8 || (argc - 5) % 3 != 0) {
        fprintf(stderr, "usage: %s BASE.gguf SIDECAR.gguf N_TOKENS CTX IMAGE 'TEXT_BEFORE' 'TEXT_AFTER' [IMAGE 'BEFORE' 'AFTER' ...]\n", argv[0]);
        return 2;
    }
    const int n_gen = atoi(argv[3]);
    const int ctx = atoi(argv[4]);
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.vision_path = argv[2];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.power_percent = 100;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) { fprintf(stderr, "probe: engine open failed\n"); return 1; }
    int failed = 0;
    for (int ai = 5; ai + 2 < argc; ai += 3) {
    const char *image_path = argv[ai], *text_before = argv[ai + 1], *text_after = argv[ai + 2];
    ds4_vision_embedding emb;
    char err[512] = {0};
    if (!ds4_engine_vision_encode_file(e, image_path, &emb, err, sizeof(err))) { fprintf(stderr, "probe: %s: encode failed: %s\n", image_path, err); failed++; continue; }
    char fp[65]; hex(emb.fingerprint, 32, fp);
    const uint32_t grid_w = emb.grid_width, grid_h = emb.grid_height, natural_rows = emb.token_count;

    ds4_tokens toks; memset(&toks, 0, sizeof(toks));
    ds4_chat_begin(e, &toks);
    const char *parts[2] = { text_before, text_after };
    ds4_vision_span span; memset(&span, 0, sizeof(span));
    if (!ds4_chat_append_multimodal_message(e, &toks, "user", parts, &emb, 1, &span, err, sizeof(err))) {
        fprintf(stderr, "probe: multimodal append failed: %s\n", err); ds4_vision_embedding_free(&emb); ds4_tokens_free(&toks); failed++; continue;
    }
    ds4_chat_append_assistant_prefix(e, &toks, DS4_THINK_NONE);
    const int prompt_len = toks.len;
    uint8_t block_sha[32]; char block_hex[65];
    ds4_image_sha256(span.embedding.data, (size_t)span.embedding.token_count * 4096u * sizeof(float), block_sha);
    hex(block_sha, 32, block_hex);

    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, ctx) != 0 || !s) { fprintf(stderr, "probe: session create failed\n"); failed++; continue; }
    int *gen = calloc((size_t)(n_gen > 0 ? n_gen : 1), sizeof(int));
    int produced = 0;
    for (int i = 0; i < n_gen; i++) {
        if (ds4_session_sync_multimodal(s, &toks, &span, 1, err, sizeof(err)) != 0) {
            fprintf(stderr, "probe: sync failed at step %d: %s\n", i, err);
            break;
        }
        const int t = ds4_session_argmax(s);
        gen[produced++] = t;
        ds4_tokens_push(&toks, t);
    }
    printf("{\"image\":\"%s\",\"fingerprint\":\"%s\",\"grid\":[%u,%u],\"natural_rows\":%u,", image_path, fp, grid_h, grid_w, natural_rows);
#ifdef DS4_VISION_PREPROCESS_VERSION
    { char idh[65]; hex(span.embedding.identity, 32, idh); printf("\"identity\":\"%s\",", idh); }
#endif
    printf("\"span\":{\"token_start\":%u,\"token_count\":%u},\"block_sha256\":\"%s\",\"prompt_len\":%d,\"prompt_ids\":[",
           span.token_start, span.embedding.token_count, block_hex, prompt_len);
    for (int i = 0; i < prompt_len; i++) printf("%s%d", i ? "," : "", toks.v[i]);
    printf("],\"generated\":[");
    for (int i = 0; i < produced; i++) printf("%s%d", i ? "," : "", gen[i]);
    printf("],\"produced\":%d,\"requested\":%d}\n", produced, n_gen);
    free(gen);
    ds4_session_free(s);
    ds4_vision_embedding_free(&span.embedding);
    ds4_tokens_free(&toks);
    if (produced != n_gen) failed++;
    fflush(stdout);
    }
    ds4_engine_close(e);
    return failed == 0 ? 0 : 1;
}
