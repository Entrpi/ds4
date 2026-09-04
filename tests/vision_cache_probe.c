/* Track B inc 8 (cache safety) on-box probe: image-conditioned session state
 * through the kvstore, keyed by identity markers.
 *
 *   vision_cache_probe BASE SIDECAR CTX KVDIR store  N_GEN IMAGES 'BEFORE' 'AFTER'
 *   vision_cache_probe BASE SIDECAR CTX KVDIR load   N_GEN IMAGES 'BEFORE' 'AFTER' ['SUFFIX']
 *   vision_cache_probe BASE SIDECAR CTX KVDIR swap   N_GEN IMAGES 'BEFORE' 'AFTER' IMAGES2
 *
 * IMAGES = "-" (text control) or "a.png[+b.png...]".  store syncs the prompt
 * and writes one record; load renders the same key (markers from the spans),
 * restores the record into a FRESH session, syncs with REFERENCE spans
 * (no embedding data: the block must come from the cache), then decodes
 * N_GEN greedy tokens and compares them against a cache-less TWIN that
 * takes the same path without the cache: sync the record's prompt (real
 * spans), then extend by the suffix -- identical means the restored state
 * IS the state.  A one-shot session (prompt+suffix synced at once) is
 * reported as information only: its chunk boundaries differ, and the text
 * path drifts across chunk boundaries with or without a cache.  swap syncs
 * IMAGES, then IMAGES2 at the same spans (identity mismatch: the session
 * must invalidate and rebuild), then IMAGES again, and reports the three
 * argmaxes.  Output is one JSON line. */
#include "ds4.h"
#include "ds4_kvstore.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

enum { MAXIMG = 4 };

static void probe_log(void *ud, ds4_kvstore_log_type type, const char *msg) {
    (void)ud; (void)type;
    fprintf(stderr, "%s\n", msg);
}

static void hex(const uint8_t *p, size_t n, char *out) {
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) { out[2 * i] = d[p[i] >> 4]; out[2 * i + 1] = d[p[i] & 15]; }
    out[2 * n] = '\0';
}

static void json_escape(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') printf("\\%c", c);
        else if (c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
}

typedef struct {
    ds4_tokens toks;
    ds4_vision_span spans[MAXIMG];
    int n_img;
    ds4_vision_lineage lineage;   /* identities from the spans (the key's markers) */
} probe_prompt;

/* Encode IMAGES and build one user turn BEFORE [img...] AFTER + assistant prefix. */
static bool build_prompt(ds4_engine *e, const char *images, const char *before, const char *after,
                         probe_prompt *p, char *err, size_t errlen) {
    memset(p, 0, sizeof(*p));
    const bool text_only = !strcmp(images, "-");
    char *pathbuf = strdup(images);
    char *paths[MAXIMG];
    if (!text_only)
        for (char *tok = strtok(pathbuf, "+"); tok && p->n_img < MAXIMG; tok = strtok(NULL, "+")) paths[p->n_img++] = tok;
    ds4_vision_embedding emb[MAXIMG];
    memset(emb, 0, sizeof(emb));
    for (int k = 0; k < p->n_img; k++) {
        if (!ds4_engine_vision_encode_file(e, paths[k], &emb[k], err, errlen)) {
            for (int j = 0; j < k; j++) ds4_vision_embedding_free(&emb[j]);
            free(pathbuf);
            return false;
        }
    }
    ds4_chat_begin(e, &p->toks);
    if (text_only) {
        char *joined = malloc(strlen(before) + strlen(after) + 1);
        strcpy(joined, before); strcat(joined, after);
        ds4_chat_append_message(e, &p->toks, "user", joined);
        free(joined);
    } else {
        const char *parts[MAXIMG + 1] = { before, "", "", "", "" };
        parts[p->n_img] = after;
        if (!ds4_chat_append_multimodal_message(e, &p->toks, "user", parts, emb, (size_t)p->n_img,
                                                p->spans, err, errlen)) {
            for (int k = 0; k < p->n_img; k++) ds4_vision_embedding_free(&emb[k]);
            free(pathbuf);
            return false;
        }
    }
    ds4_chat_append_assistant_prefix(e, &p->toks, DS4_THINK_NONE);
    ds4_vision_lineage_store(&p->lineage, p->spans, (size_t)p->n_img);
    free(pathbuf);
    return true;
}

static void free_prompt(probe_prompt *p) {
    for (int k = 0; k < p->n_img; k++) ds4_vision_embedding_free(&p->spans[k].embedding);
    ds4_tokens_free(&p->toks);
    ds4_vision_lineage_clear(&p->lineage);
}

static int sync_any(ds4_session *s, const ds4_tokens *toks, const ds4_vision_span *spans, int n_img,
                    char *err, size_t errlen) {
    return n_img ? ds4_session_sync_multimodal(s, toks, spans, (size_t)n_img, err, errlen)
                 : ds4_session_sync(s, toks, err, errlen);
}

/* Greedy decode n tokens from the synced session, extending `toks` with the
 * spans given (real or reference).  Returns the count produced, -1 on error. */
static int decode(ds4_session *s, ds4_tokens *toks, const ds4_vision_span *spans, int n_img,
                  int n, int *out, char *err, size_t errlen) {
    int produced = 0;
    for (int i = 0; i < n; i++) {
        if (sync_any(s, toks, spans, n_img, err, errlen) != 0) return -1;
        const int t = ds4_session_argmax(s);
        out[produced++] = t;
        ds4_tokens_push(toks, t);
        if (t == 1) break;
    }
    return produced;
}

static void print_ids(const char *key, const int *v, int n) {
    printf("\"%s\":[", key);
    for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", v[i]);
    printf("]");
}

static int lcp_ids(const int *a, int na, const int *b, int nb) {
    int n = 0;
    while (n < na && n < nb && a[n] == b[n]) n++;
    return n;
}

int main(int argc, char **argv) {
    if (argc < 10) {
        fprintf(stderr, "usage: %s BASE SIDECAR CTX KVDIR store|load|swap N_GEN IMAGES 'BEFORE' 'AFTER' [SUFFIX|IMAGES2]\n", argv[0]);
        return 2;
    }
    const int ctx = atoi(argv[3]);
    const char *kvdir = argv[4], *mode = argv[5];
    const int n_gen = atoi(argv[6]);
    const char *images = argv[7], *before = argv[8], *after = argv[9];
    const char *extra = argc > 10 ? argv[10] : NULL;

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.vision_path = argv[2];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.power_percent = 100;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) { fprintf(stderr, "probe: engine open failed\n"); return 1; }
    char err[512] = {0};

    probe_prompt p;
    if (!build_prompt(e, images, before, after, &p, err, sizeof(err))) {
        fprintf(stderr, "probe: prompt build failed: %s\n", err);
        return 1;
    }
    size_t key_len = 0;
    char *key = ds4_kvstore_render_tokens_text_lineage(e, &p.toks, p.lineage.v, p.lineage.n, &key_len, err, sizeof(err));
    if (!key) { fprintf(stderr, "probe: key render failed: %s\n", err); return 1; }
    char key_sha[41];
    ds4_kvstore_sha1_bytes_hex(key, key_len, key_sha);

    ds4_kvstore kc;
    memset(&kc, 0, sizeof(kc));
    ds4_kvstore_options kopt = ds4_kvstore_default_options();
    kopt.min_tokens = 1;   /* the fixtures are short prompts; the default 512 floor is a policy, not a format rule */
    if (!ds4_kvstore_open(&kc, kvdir, 8192, false, kopt, "probe", probe_log, NULL)) {
        fprintf(stderr, "probe: kvstore open failed: %s\n", kvdir);
        return 1;
    }
    int rc = 0;
    printf("{\"mode\":\"%s\",\"images\":\"%s\",\"n_images\":%d,\"prompt_len\":%d,\"key_len\":%zu,\"key_sha\":\"%s\",\"markers\":%s,",
           mode, images, p.n_img, p.toks.len, key_len, key_sha,
           ds4_kvstore_text_has_image_marker(key, key_len) ? "true" : "false");
    if (p.n_img) { char idh[65]; hex(p.spans[0].embedding.identity, 32, idh); printf("\"identity0\":\"%s\",", idh); }

    if (!strcmp(mode, "store")) {
        ds4_session *s = NULL;
        if (ds4_session_create(&s, e, ctx) != 0 || !s) { fprintf(stderr, "probe: session create failed\n"); return 1; }
        const int prompt_len = p.toks.len;
        int gen[256]; const int n = n_gen > 256 ? 256 : n_gen;
        /* sync the prompt first (the record = the prompt), then store, then decode */
        if (sync_any(s, &p.toks, p.spans, p.n_img, err, sizeof(err)) != 0) { fprintf(stderr, "probe: sync failed: %s\n", err); return 1; }
        const ds4_vision_identity *ids = NULL;
        const size_t n_ids = ds4_session_vision_identities(s, &ids);
        const bool stored = ds4_kvstore_store_live_prefix_text(&kc, e, s, ctx, &p.toks, prompt_len, "cold",
                                                               NULL, 0, NULL, NULL, NULL, NULL, 0, err, sizeof(err));
        printf("\"session_images\":%zu,\"stored\":%s,\"store_err\":\"", n_ids, stored ? "true" : "false");
        json_escape(err, strlen(err));
        printf("\",");
        const int produced = decode(s, &p.toks, p.spans, p.n_img, n, gen, err, sizeof(err));
        if (produced < 0) { fprintf(stderr, "probe: decode failed: %s\n", err); rc = 1; }
        print_ids("generated", gen, produced > 0 ? produced : 0);
        printf(",\"generation\":%llu", (unsigned long long)ds4_session_generation(s));
        ds4_session_free(s);
        if (!stored) rc = 1;
    } else if (!strcmp(mode, "load")) {
        /* the prompt text = key + optional text suffix; the record must byte-prefix it */
        size_t text_len = key_len + (extra ? strlen(extra) : 0);
        char *text = malloc(text_len + 1);
        memcpy(text, key, key_len);
        if (extra) memcpy(text + key_len, extra, strlen(extra));
        text[text_len] = '\0';
        ds4_session *s = NULL;
        if (ds4_session_create(&s, e, ctx) != 0 || !s) { fprintf(stderr, "probe: session create failed\n"); return 1; }
        ds4_tokens effective; memset(&effective, 0, sizeof(effective));
        ds4_kvstore_load_result lr; memset(&lr, 0, sizeof(lr));
        const int loaded = ds4_kvstore_try_load_text(&kc, e, s, text, &effective, &lr, NULL, false);
        printf("\"loaded_tokens\":%d,\"effective_len\":%d,\"load_ms\":%.1f,\"path\":\"%s\",",
               loaded, effective.len, lr.load_ms, lr.path ? lr.path : "");
        const ds4_vision_identity *ids = NULL;
        const size_t n_ids = ds4_session_vision_identities(s, &ids);
        printf("\"restored_images\":%zu,", n_ids);
        int gen_r[256], gen_f[256], gen_o[256]; int nr = 0, nf = 0, no = 0;
        const int n = n_gen > 256 ? 256 : n_gen;
        if (loaded > 0) {
            /* reference spans: same start/count/identity, NO data */
            ds4_vision_span ref[MAXIMG];
            memcpy(ref, p.spans, sizeof(ref));
            for (int k = 0; k < p.n_img; k++) ref[k].embedding.data = NULL;
            ds4_tokens toks_r; memset(&toks_r, 0, sizeof(toks_r));
            ds4_tokens_copy(&toks_r, &effective);
            nr = decode(s, &toks_r, ref, p.n_img, n, gen_r, err, sizeof(err));
            printf("\"restored_err\":\""); json_escape(err, strlen(err)); printf("\",");
            print_ids("restored", gen_r, nr > 0 ? nr : 0);
            printf(",");
            ds4_tokens_free(&toks_r);
            /* the cache-less TWIN: the same two steps without the cache -- sync the
             * record's prompt with real spans, then extend by the suffix */
            ds4_session *f = NULL;
            if (ds4_session_create(&f, e, ctx) != 0 || !f) { fprintf(stderr, "probe: session create failed\n"); return 1; }
            err[0] = '\0';
            if (sync_any(f, &p.toks, p.spans, p.n_img, err, sizeof(err)) != 0) { fprintf(stderr, "probe: twin sync failed: %s\n", err); return 1; }
            ds4_tokens toks_f; memset(&toks_f, 0, sizeof(toks_f));
            ds4_tokens_copy(&toks_f, &effective);
            nf = decode(f, &toks_f, p.spans, p.n_img, n, gen_f, err, sizeof(err));
            printf("\"fresh_err\":\""); json_escape(err, strlen(err)); printf("\",");
            print_ids("fresh", gen_f, nf > 0 ? nf : 0);
            printf(",");
            ds4_tokens_free(&toks_f);
            ds4_session_free(f);
            /* information: the one-shot session (prompt+suffix in one sync; different
             * chunk boundaries, so the text path itself may drift from the twin) */
            ds4_session *o = NULL;
            if (ds4_session_create(&o, e, ctx) != 0 || !o) { fprintf(stderr, "probe: session create failed\n"); return 1; }
            ds4_tokens toks_o; memset(&toks_o, 0, sizeof(toks_o));
            ds4_tokens_copy(&toks_o, &effective);
            err[0] = '\0';
            no = decode(o, &toks_o, p.spans, p.n_img, n, gen_o, err, sizeof(err));
            print_ids("oneshot", gen_o, no > 0 ? no : 0);
            printf(",\"oneshot_vs_fresh_lcp\":%d,", lcp_ids(gen_o, no > 0 ? no : 0, gen_f, nf > 0 ? nf : 0));
            ds4_tokens_free(&toks_o);
            ds4_session_free(o);
            const bool same = nr == nf && nr > 0 && memcmp(gen_r, gen_f, (size_t)nr * sizeof(int)) == 0;
            printf("\"restored_equals_fresh\":%s,\"restored_vs_fresh_lcp\":%d,", same ? "true" : "false",
                   lcp_ids(gen_r, nr > 0 ? nr : 0, gen_f, nf > 0 ? nf : 0));
            if (!same) rc = 1;
        } else {
            printf("\"restored\":[],\"fresh\":[],\"oneshot\":[],\"restored_equals_fresh\":false,");
        }
        printf("\"hit\":%s", loaded > 0 ? "true" : "false");
        ds4_kvstore_load_result_free(&lr);
        ds4_tokens_free(&effective);
        ds4_session_free(s);
        free(text);
    } else if (!strcmp(mode, "swap")) {
        if (!extra) { fprintf(stderr, "probe: swap needs IMAGES2\n"); return 1; }
        probe_prompt q;
        if (!build_prompt(e, extra, before, after, &q, err, sizeof(err))) {
            fprintf(stderr, "probe: prompt 2 build failed: %s\n", err);
            return 1;
        }
        ds4_session *s = NULL;
        if (ds4_session_create(&s, e, ctx) != 0 || !s) { fprintf(stderr, "probe: session create failed\n"); return 1; }
        int a1 = -1, a2 = -1, a3 = -1;
        uint64_t g1 = 0, g2 = 0, g3 = 0;
        const bool same_shape = q.toks.len == p.toks.len && q.n_img == p.n_img;
        if (ds4_session_sync_multimodal(s, &p.toks, p.spans, (size_t)p.n_img, err, sizeof(err)) != 0) { fprintf(stderr, "probe: sync 1 failed: %s\n", err); return 1; }
        a1 = ds4_session_argmax(s); g1 = ds4_session_generation(s);
        const bool match2 = ds4_session_vision_state_matches(s, q.spans, (size_t)q.n_img);
        if (ds4_session_sync_multimodal(s, &q.toks, q.spans, (size_t)q.n_img, err, sizeof(err)) != 0) { fprintf(stderr, "probe: sync 2 failed: %s\n", err); return 1; }
        a2 = ds4_session_argmax(s); g2 = ds4_session_generation(s);
        const bool match3 = ds4_session_vision_state_matches(s, p.spans, (size_t)p.n_img);
        if (ds4_session_sync_multimodal(s, &p.toks, p.spans, (size_t)p.n_img, err, sizeof(err)) != 0) { fprintf(stderr, "probe: sync 3 failed: %s\n", err); return 1; }
        a3 = ds4_session_argmax(s); g3 = ds4_session_generation(s);
        printf("\"same_shape\":%s,\"images2\":\"%s\",\"prompt2_len\":%d,\"argmax\":[%d,%d,%d],\"generation\":[%llu,%llu,%llu],"
               "\"state_matches_before_swap\":%s,\"state_matches_before_restore\":%s,\"swap_invalidated\":%s,\"restore_invalidated\":%s",
               same_shape ? "true" : "false", extra, q.toks.len, a1, a2, a3,
               (unsigned long long)g1, (unsigned long long)g2, (unsigned long long)g3,
               match2 ? "true" : "false", match3 ? "true" : "false",
               g2 != g1 ? "true" : "false", g3 != g2 ? "true" : "false");
        ds4_session_free(s);
        free_prompt(&q);
    } else {
        fprintf(stderr, "probe: unknown mode %s\n", mode);
        rc = 2;
    }
    printf("}\n");
    ds4_kvstore_close(&kc);
    free(key);
    free_prompt(&p);
    ds4_engine_close(e);
    return rc;
}
