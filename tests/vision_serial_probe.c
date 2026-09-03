/* Track B inc 3+4 gate (a): token identity of an image prompt, same source for
 * the fork and the upstream oracle.  Opens BASE + SIDECAR on CUDA, encodes IMAGE,
 * builds a user turn "TEXT_BEFORE <image> TEXT_AFTER", disables thinking, syncs
 * the serial session (passing the image span on every sync), and greedy-decodes
 * N tokens.  Prints one JSON object: prompt ids, the span, an FNV-1a hash of the
 * scattered block rows, the continuation.  Only API shared by both trees is
 * used; the identity line is fork-only (guarded on DS4_VISION_PREPROCESS_VERSION).
 * Run with DS4_CUDA_FP8_KV=0 on the fork (visual attention needs F32 comp KV). */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void hex(const uint8_t *b, size_t n, char *out) { for (size_t i = 0; i < n; i++) sprintf(out + 2 * i, "%02x", b[i]); out[2 * n] = 0; }
/* FNV-1a 64 over raw bytes: an equality hash both trees compute identically
 * (the fork's sha256 helper is not part of upstream's public surface). */
static uint64_t fnv1a64(const void *data, size_t n) {
    const uint8_t *p = data; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

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
    const bool text_only = !strcmp(image_path, "-");   /* control: the same turn without an image */
    ds4_vision_embedding emb; memset(&emb, 0, sizeof(emb));
    char err[512] = {0};
    char fp[65] = "-";
    uint32_t grid_w = 0, grid_h = 0, natural_rows = 0;
    if (!text_only) {
        if (!ds4_engine_vision_encode_file(e, image_path, &emb, err, sizeof(err))) { fprintf(stderr, "probe: %s: encode failed: %s\n", image_path, err); failed++; continue; }
        hex(emb.fingerprint, 32, fp);
        grid_w = emb.grid_width; grid_h = emb.grid_height; natural_rows = emb.token_count;
    }

    ds4_tokens toks; memset(&toks, 0, sizeof(toks));
    ds4_chat_begin(e, &toks);
    const char *parts[2] = { text_before, text_after };
    ds4_vision_span span; memset(&span, 0, sizeof(span));
    if (text_only) {
        char *joined = malloc(strlen(text_before) + strlen(text_after) + 1);
        strcpy(joined, text_before); strcat(joined, text_after);
        ds4_chat_append_message(e, &toks, "user", joined);
        free(joined);
    } else if (!ds4_chat_append_multimodal_message(e, &toks, "user", parts, &emb, 1, &span, err, sizeof(err))) {
        fprintf(stderr, "probe: multimodal append failed: %s\n", err); ds4_vision_embedding_free(&emb); ds4_tokens_free(&toks); failed++; continue;
    }
    ds4_chat_append_assistant_prefix(e, &toks, DS4_THINK_NONE);
    const int prompt_len = toks.len;
    char block_hex[17] = "-";
    if (!text_only)
        snprintf(block_hex, sizeof(block_hex), "%016llx",
                 (unsigned long long)fnv1a64(span.embedding.data, (size_t)span.embedding.token_count * 4096u * sizeof(float)));

    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, ctx) != 0 || !s) { fprintf(stderr, "probe: session create failed\n"); failed++; continue; }
    int *gen = calloc((size_t)(n_gen > 0 ? n_gen : 1), sizeof(int));
    int *top4 = calloc((size_t)(n_gen > 0 ? n_gen : 1) * 4u, sizeof(int));
#ifdef DS4_VISION_PREPROCESS_VERSION
    /* fork only: the top-8 (id, logit) at every step, for logit-level adjudication */
    int *t8id = calloc((size_t)(n_gen > 0 ? n_gen : 1) * 8u, sizeof(int));
    float *t8lg = calloc((size_t)(n_gen > 0 ? n_gen : 1) * 8u, sizeof(float));
#endif
    int produced = 0;
    for (int i = 0; i < n_gen; i++) {
        const int rc = text_only ? ds4_session_sync(s, &toks, err, sizeof(err))
                                 : ds4_session_sync_multimodal(s, &toks, &span, 1, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "probe: sync failed at step %d: %s\n", i, err);
            break;
        }
        const int t = ds4_session_argmax(s);
        /* the top-4 ORDER at this position: a near-tie flip vs the oracle shows
         * its top-1 among these, a wrong forward does not */
        top4[4 * i] = t;
        top4[4 * i + 1] = ds4_session_argmax_excluding(s, t);
        {   /* third and fourth by two more exclusions (argmax_excluding takes one id) */
            int a = top4[4 * i], b = top4[4 * i + 1];
            int c = -1, d = -1;
            /* cheap: the API excludes one id; emulate a 2-exclusion by picking the
             * best of argmax_excluding(a) and argmax_excluding(b) that is neither */
            int ca = ds4_session_argmax_excluding(s, a), cb = ds4_session_argmax_excluding(s, b);
            c = (ca != b) ? ca : ((cb != a) ? cb : -1);
            (void)d;
            top4[4 * i + 2] = c;
            top4[4 * i + 3] = -1;
        }
#ifdef DS4_VISION_PREPROCESS_VERSION
        {
            const float *lg = ds4_session_logits(s); const int nv = ds4_session_n_vocab(s);
            int used[8]; int nu = 0;
            for (int k = 0; k < 8; k++) {
                int best = -1; float bv = -1e30f;
                for (int v = 0; v < nv; v++) {
                    bool skip = false;
                    for (int u = 0; u < nu; u++) if (used[u] == v) { skip = true; break; }
                    if (!skip && lg[v] > bv) { bv = lg[v]; best = v; }
                }
                used[nu++] = best; t8id[8 * i + k] = best; t8lg[8 * i + k] = bv;
            }
        }
#endif
        gen[produced++] = t;
        ds4_tokens_push(&toks, t);
    }
    printf("{\"image\":\"%s\",\"fingerprint\":\"%s\",\"grid\":[%u,%u],\"natural_rows\":%u,", image_path, fp, grid_h, grid_w, natural_rows);
#ifdef DS4_VISION_PREPROCESS_VERSION
    { char idh[65]; hex(span.embedding.identity, 32, idh); printf("\"identity\":\"%s\",", idh); }
#endif
    printf("\"span\":{\"token_start\":%u,\"token_count\":%u},\"block_fnv1a64\":\"%s\",\"prompt_len\":%d,\"prompt_ids\":[",
           span.token_start, span.embedding.token_count, block_hex, prompt_len);
    for (int i = 0; i < prompt_len; i++) printf("%s%d", i ? "," : "", toks.v[i]);
    printf("],\"generated\":[");
    for (int i = 0; i < produced; i++) printf("%s%d", i ? "," : "", gen[i]);
    printf("],\"top3\":[");
    for (int i = 0; i < produced; i++) printf("%s[%d,%d,%d]", i ? "," : "", top4[4 * i], top4[4 * i + 1], top4[4 * i + 2]);
    printf("]");
#ifdef DS4_VISION_PREPROCESS_VERSION
    printf(",\"top8\":[");
    for (int i = 0; i < produced; i++) {
        printf("%s[", i ? "," : "");
        for (int k = 0; k < 8; k++) printf("%s[%d,%.4f]", k ? "," : "", t8id[8 * i + k], (double)t8lg[8 * i + k]);
        printf("]");
    }
    printf("]");
    free(t8id); free(t8lg);
#endif
    printf(",\"produced\":%d,\"requested\":%d}\n", produced, n_gen);
    free(gen); free(top4);
    ds4_session_free(s);
    if (!text_only) ds4_vision_embedding_free(&span.embedding);
    ds4_tokens_free(&toks);
    if (produced != n_gen) failed++;
    fflush(stdout);
    }
    ds4_engine_close(e);
    return failed == 0 ? 0 : 1;
}
