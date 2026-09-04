/* Track B inc 6+9 gate: the batched bank path serves image prompts.
 *
 *   vision_cont_probe BASE SIDECAR CTX N_GEN IMAGES 'BEFORE' 'AFTER' [IMAGES 'BEFORE' 'AFTER' ...]
 *
 * IMAGES = "-" (a text control) or "a.png[+b.png...]".  Up to 8 prompts, each
 * one user turn BEFORE [image...] AFTER + the assistant prefix, thinking off.
 *
 * Legs (one JSON line each, then PASS/FAIL per leg, exit = failures):
 *   serial    the serial lane's greedy N_GEN tokens per prompt (the oracle
 *             the batched lane must reproduce; the identity legs proved the
 *             serial lane against upstream).
 *   batch     every prompt admitted into ONE persistent batch ctx (width =
 *             prompt count), cold, greedy; each row's tokens vs its serial
 *             twin.  Image rows engage the visual path ("image prefill
 *             engaged" on stderr).
 *   text      the text prompts alone in a second batch: byte-identical to
 *             their rows of the mixed batch (image banks never perturb text
 *             banks).
 *   warm      the first image prompt re-admitted on its own bank with the
 *             committed history + a text suffix and REFERENCE spans (no
 *             embedding data): the engine must report the history as cached
 *             (n_cached == committed) and the continuation must equal the
 *             serial lane's sync+extend of the same tokens with real spans
 *             (the inc 8 bar: restore+extend vs sync+extend).
 *   payload   the same bank staged to a file and restored into ANOTHER bank:
 *             lineage preserved, and the same warm re-admit there gives the
 *             same continuation.
 *   reject    reference spans whose identity was flipped: the engine degrades
 *             the reuse to cold and, with no data to prefill, REJECTS the
 *             admission (on_done with NULL tokens); a cold admit with a
 *             data-less span rejects too.
 */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

enum { MAXP = 8, MAXIMG = 4, MAXGEN = 256 };
typedef struct { int scored, agree, neartie, flagged, first_flag, flag_rank; float max_gap; } tf_score;

typedef struct {
    const char *images, *before, *after;
    bool text_only;
    int n_img;
    ds4_vision_embedding emb[MAXIMG];
    ds4_vision_span spans[MAXIMG];
    ds4_tokens toks;
    int serial[MAXGEN]; int n_serial;
    tf_score self_sc; int self_ok;   /* the serial lane's own chunk-boundary noise (one-shot vs sync+extend) */
} prompt_t;

typedef struct {
    const int *tokens; int n;
    const ds4_vision_span *images; size_t image_count;
    int place_bank, n_cached;
    int max_new;
    /* results */
    int out[MAXGEN]; int n_out; int finish;
    bool done, rejected;
    int bank_used, adm_cached, adm_computed;
} creq_t;

typedef struct { creq_t *q; int n, next; } cqueue_t;

static int cb_admit(void *ud, ds4_cont_request *req) {
    cqueue_t *cq = ud;
    if (cq->next >= cq->n) return 0;
    creq_t *r = &cq->q[cq->next++];
    req->tokens = r->tokens; req->n = r->n; req->max_new = r->max_new; req->eos = -1;
    req->user = r; req->images = r->images; req->image_count = r->image_count;
    req->place_bank = r->place_bank; req->n_cached = r->n_cached;
    req->bank_used = &r->bank_used;
    req->temperature = 0.0f;   /* greedy */
    return 1;
}
static int cb_on_admitted(void *ud, void *user, int n_cached, int n_computed, int bank) {
    (void)ud; creq_t *r = user;
    r->adm_cached = n_cached; r->adm_computed = n_computed; r->bank_used = bank;
    return 1;
}
static void cb_on_done(void *ud, void *user, const int *tokens, int n, int finish) {
    (void)ud; creq_t *r = user;
    if (!r) return;
    r->done = true;
    if (!tokens) { r->rejected = true; return; }
    r->n_out = n > MAXGEN ? MAXGEN : n;
    memcpy(r->out, tokens, (size_t)r->n_out * sizeof(int));
    r->finish = finish;
}
/* the admit path sets on_admitted through a global (the request struct is filled per admit) */
static int cb_admit_full(void *ud, ds4_cont_request *req) {
    if (!cb_admit(ud, req)) return 0;
    req->on_admitted = cb_on_admitted;
    return 1;
}

static int run_batch(ds4_batch_ctx *ctx, creq_t *q, int n, char *err, size_t errlen) {
    cqueue_t cq = { q, n, 0 };
    for (int i = 0; i < n; i++) { q[i].done = false; q[i].rejected = false; q[i].n_out = 0; q[i].bank_used = -1; q[i].adm_cached = -1; q[i].adm_computed = -1; }
    return ds4_engine_continuous_generate(ctx, cb_admit_full, NULL, cb_on_done, &cq, err, errlen);
}

static int eos_cut(const int *t, int n, int eos) {
    for (int i = 0; i < n; i++) if (t[i] == eos) return i + 1;
    return n;
}
/* compare two greedy streams up to the shorter one's EOS; returns the first divergence or -1 */
static int first_diff(const int *a, int na, const int *b, int nb, int eos) {
    const int ca = eos_cut(a, na, eos), cb = eos_cut(b, nb, eos);
    const int n = ca < cb ? ca : cb;
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return i;
    return ca != cb ? n : -1;
}
static void print_ids(const char *key, const int *t, int n) {
    printf("\"%s\":[", key);
    for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", t[i]);
    printf("]");
}

static int serial_greedy(ds4_engine *e, int ctx, const ds4_tokens *prompt,
                         const ds4_vision_span *spans, int n_img, int n_gen, int *out, char *err, size_t errlen) {
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, ctx) != 0 || !s) { snprintf(err, errlen, "session create failed"); return -1; }
    ds4_tokens toks = {0};
    for (int i = 0; i < prompt->len; i++) ds4_tokens_push(&toks, prompt->v[i]);
    const int eos = ds4_token_eos(e);
    int produced = 0;
    for (int i = 0; i < n_gen; i++) {
        const int rc = n_img ? ds4_session_sync_multimodal(s, &toks, spans, (size_t)n_img, err, errlen)
                             : ds4_session_sync(s, &toks, err, errlen);
        if (rc != 0) { produced = -1; break; }
        const int t = ds4_session_argmax(s);
        out[produced++] = t;
        ds4_tokens_push(&toks, t);
        if (t == eos) break;
    }
    ds4_tokens_free(&toks);
    ds4_session_free(s);
    return produced;
}

/* Teacher-forced adjudication (the V2 bar): feed the serial lane the batched
 * row's own prefix step by step and score each batched pick against the
 * serial logits at that position -- agree (serial argmax), near-tie (the
 * serial top-1 leads the pick by <= band logits), flagged (a larger gap).
 * The batched lane is FP-noise-different from serial by construction (the
 * text control diverges too), so byte identity is not the bar; a flagged
 * position is. */
static int teacher_force(ds4_engine *e, int ctx, const ds4_tokens *prompt, const ds4_vision_span *spans, int n_img,
                         const int *forced, int n_forced, float band, tf_score *sc, char *err, size_t errlen) {
    memset(sc, 0, sizeof(*sc));
    sc->first_flag = -1;
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, ctx) != 0 || !s) { snprintf(err, errlen, "session create failed"); return -1; }
    ds4_tokens toks = {0};
    for (int i = 0; i < prompt->len; i++) ds4_tokens_push(&toks, prompt->v[i]);
    const int eos = ds4_token_eos(e);
    int rc = 0;
    for (int i = 0; i < n_forced; i++) {
        rc = n_img ? ds4_session_sync_multimodal(s, &toks, spans, (size_t)n_img, err, errlen)
                   : ds4_session_sync(s, &toks, err, errlen);
        if (rc != 0) break;
        const float *lg = ds4_session_logits(s);
        const int nv = ds4_session_n_vocab(s);
        const int top = ds4_session_argmax(s);
        const int pick = forced[i];
        if (pick < 0 || pick >= nv) { rc = 1; snprintf(err, errlen, "forced id out of range"); break; }
        const float gap = lg[top] - lg[pick];
        sc->scored++;
        if (top == pick) sc->agree++;
        else if (gap <= band) sc->neartie++;
        else {
            sc->flagged++;
            if (sc->first_flag < 0) {
                sc->first_flag = i;
                int rank = 1;
                for (int v = 0; v < nv; v++) if (lg[v] > lg[pick]) rank++;
                sc->flag_rank = rank;
            }
        }
        if (gap > sc->max_gap) sc->max_gap = gap;
        ds4_tokens_push(&toks, pick);
        if (pick == eos) break;
    }
    ds4_tokens_free(&toks);
    ds4_session_free(s);
    return rc == 0 ? 0 : -1;
}
static void print_score(const char *key, const tf_score *sc) {
    printf("\"%s\":{\"scored\":%d,\"agree\":%d,\"neartie\":%d,\"flagged\":%d,\"first_flag\":%d,\"flag_rank\":%d,\"max_gap\":%.4f}",
           key, sc->scored, sc->agree, sc->neartie, sc->flagged, sc->first_flag, sc->flag_rank, sc->max_gap);
}

/* The serial lane against ITSELF: greedy tokens produced after a different
 * chunk boundary (sync the prompt minus its last 8 tokens, then extend) are
 * teacher-forced through a one-shot session.  The gaps this produces are the
 * engine's own FP noise envelope (inc 8 saw lcp 15/24 on a text control from
 * exactly this shape); a cross-lane gap inside it is not a lane defect. */
static int self_noise_split(ds4_engine *e, int ctx, const ds4_tokens *prompt, const ds4_vision_span *spans, int n_img,
                            int n_gen, float band, int back, tf_score *sc, char *err, size_t errlen) {
    memset(sc, 0, sizeof(*sc));
    sc->first_flag = -1;
    /* back < 0: cut right BEFORE the first image block (the serving shape a
     * text prefix + image turn produces on the serial lane). */
    int cut;
    if (back < 0) {
        if (n_img == 0 || spans[0].token_start < 8u) return 1;
        cut = (int)spans[0].token_start;
    } else {
        if (prompt->len <= back + 8) return 1;   /* nothing to split */
        cut = prompt->len - back;
    }
    for (int k = 0; k < n_img; k++)
        if ((uint64_t)spans[k].token_start + spans[k].embedding.token_count > (uint64_t)cut) return 1;   /* block past the cut: skip */
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, ctx) != 0 || !s) { snprintf(err, errlen, "session create failed"); return -1; }
    ds4_tokens toks = {0};
    for (int i = 0; i < cut; i++) ds4_tokens_push(&toks, prompt->v[i]);
    int rc = n_img ? ds4_session_sync_multimodal(s, &toks, spans, (size_t)n_img, err, errlen)
                   : ds4_session_sync(s, &toks, err, errlen);
    for (int i = cut; rc == 0 && i < prompt->len; i++) ds4_tokens_push(&toks, prompt->v[i]);
    int gen[MAXGEN]; int n = 0;
    const int eos = ds4_token_eos(e);
    for (int i = 0; rc == 0 && i < n_gen; i++) {
        rc = n_img ? ds4_session_sync_multimodal(s, &toks, spans, (size_t)n_img, err, errlen)
                   : ds4_session_sync(s, &toks, err, errlen);
        if (rc != 0) break;
        const int t = ds4_session_argmax(s);
        gen[n++] = t;
        ds4_tokens_push(&toks, t);
        if (t == eos) break;
    }
    ds4_tokens_free(&toks);
    ds4_session_free(s);
    if (rc != 0) return -1;
    return teacher_force(e, ctx, prompt, spans, n_img, gen, n, band, sc, err, errlen) == 0 ? 2 : -1;
}
/* Pool several chunk boundaries (8, 24, 64 tokens before the end, and right
 * before the first image block): the
 * envelope is the max gap any split produces; one split can land on a
 * position where the noise happens to cancel. */
static int self_noise(ds4_engine *e, int ctx, const ds4_tokens *prompt, const ds4_vision_span *spans, int n_img,
                      int n_gen, float band, tf_score *sc, char *err, size_t errlen) {
    static const int backs[4] = { 8, 24, 64, -1 };
    memset(sc, 0, sizeof(*sc));
    sc->first_flag = -1;
    int measured = 0;
    for (int k = 0; k < 4; k++) {
        tf_score one;
        const int rc = self_noise_split(e, ctx, prompt, spans, n_img, n_gen, band, backs[k], &one, err, errlen);
        if (rc < 0) return -1;
        if (rc != 2) continue;
        measured++;
        sc->scored += one.scored; sc->agree += one.agree; sc->neartie += one.neartie; sc->flagged += one.flagged;
        if (one.max_gap > sc->max_gap) { sc->max_gap = one.max_gap; sc->first_flag = one.first_flag; sc->flag_rank = one.flag_rank; }
    }
    return measured ? 2 : 1;
}

static bool build_prompt(ds4_engine *e, prompt_t *p, char *err, size_t errlen) {
    p->text_only = !strcmp(p->images, "-");
    char *pathbuf = strdup(p->images);
    char *paths[MAXIMG];
    p->n_img = 0;
    if (!p->text_only)
        for (char *tok = strtok(pathbuf, "+"); tok && p->n_img < MAXIMG; tok = strtok(NULL, "+")) paths[p->n_img++] = tok;
    for (int k = 0; k < p->n_img; k++)
        if (!ds4_engine_vision_encode_file(e, paths[k], &p->emb[k], err, errlen)) { free(pathbuf); return false; }
    ds4_chat_begin(e, &p->toks);
    if (p->text_only) {
        char *joined = malloc(strlen(p->before) + strlen(p->after) + 1);
        strcpy(joined, p->before); strcat(joined, p->after);
        ds4_chat_append_message(e, &p->toks, "user", joined);
        free(joined);
    } else {
        const char *parts[MAXIMG + 1] = { p->before, "", "", "", "" };
        parts[p->n_img] = p->after;
        if (!ds4_chat_append_multimodal_message(e, &p->toks, "user", parts, p->emb, (size_t)p->n_img, p->spans, err, errlen)) {
            free(pathbuf); return false;
        }
    }
    ds4_chat_append_assistant_prefix(e, &p->toks, DS4_THINK_NONE);
    free(pathbuf);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 8 || (argc - 5) % 3 != 0) {
        fprintf(stderr, "usage: %s BASE.gguf SIDECAR.gguf CTX N_GEN IMAGES 'BEFORE' 'AFTER' [IMAGES 'BEFORE' 'AFTER' ...]\n", argv[0]);
        return 2;
    }
    const int ctx = atoi(argv[3]);
    int n_gen = atoi(argv[4]);
    if (n_gen < 1) n_gen = 1;
    if (n_gen > MAXGEN) n_gen = MAXGEN;
    const int np = (argc - 5) / 3;
    if (np > MAXP) { fprintf(stderr, "probe: at most %d prompts\n", MAXP); return 2; }
    const char *band_env = getenv("CONT_NEAR_TIE_BAND");
    const float band = band_env && band_env[0] ? (float)atof(band_env) : 0.34f;   /* the V2 text drift band */
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.vision_path = argv[2];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.power_percent = 100;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) { fprintf(stderr, "probe: engine open failed\n"); return 1; }
    const int eos = ds4_token_eos(e);
    char err[512] = {0};
    int failures = 0;
    prompt_t P[MAXP];
    memset(P, 0, sizeof(P));
    for (int i = 0; i < np; i++) {
        P[i].images = argv[5 + 3 * i]; P[i].before = argv[6 + 3 * i]; P[i].after = argv[7 + 3 * i];
        if (!build_prompt(e, &P[i], err, sizeof(err))) { fprintf(stderr, "probe: prompt %d: %s\n", i, err); return 1; }
    }

    /* CONT_DUMP_SERIAL_LOGITS=path: the serial lane's logits at the seed position
     * of every prompt (u32 index, u32 prompt length, float[n_vocab]); the engine's
     * DS4_CONT_DUMP_SEED_LOGITS twin records the batched lane's, matched by length. */
    const char *serial_dump = getenv("CONT_DUMP_SERIAL_LOGITS");
    if (serial_dump && serial_dump[0]) {
        FILE *df = fopen(serial_dump, "wb");
        for (int i = 0; df && i < np; i++) {
            ds4_session *s = NULL;
            if (ds4_session_create(&s, e, ctx) != 0 || !s) break;
            const int rc = P[i].n_img ? ds4_session_sync_multimodal(s, &P[i].toks, P[i].spans, (size_t)P[i].n_img, err, sizeof(err))
                                      : ds4_session_sync(s, &P[i].toks, err, sizeof(err));
            if (rc == 0) {
                const uint32_t hdr[2] = { (uint32_t)i, (uint32_t)P[i].toks.len };
                fwrite(hdr, sizeof(uint32_t), 2, df);
                fwrite(ds4_session_logits(s), sizeof(float), (size_t)ds4_session_n_vocab(s), df);
            }
            ds4_session_free(s);
        }
        if (df) fclose(df);
    }
    /* ---- serial: the greedy reference per prompt (information + the EOS cut) ---- */
    for (int i = 0; i < np; i++) {
        P[i].n_serial = serial_greedy(e, ctx, &P[i].toks, P[i].spans, P[i].n_img, n_gen, P[i].serial, err, sizeof(err));
        if (P[i].n_serial < 0) { fprintf(stderr, "probe: serial %d failed: %s\n", i, err); return 1; }
        P[i].self_ok = self_noise(e, ctx, &P[i].toks, P[i].spans, P[i].n_img, n_gen, band, &P[i].self_sc, err, sizeof(err));
        if (P[i].self_ok < 0) { fprintf(stderr, "probe: self-noise %d failed: %s\n", i, err); return 1; }
        printf("{\"leg\":\"serial\",\"prompt\":%d,\"images\":%d,\"prompt_len\":%d,\"self_measured\":%s,", i, P[i].n_img, P[i].toks.len,
               P[i].self_ok == 2 ? "true" : "false");
        print_score("self_noise", &P[i].self_sc); printf(",");
        print_ids("tokens", P[i].serial, P[i].n_serial);
        printf("}\n");
    }
    float self_env = band;
    for (int i = 0; i < np; i++) if (P[i].self_ok == 2 && P[i].self_sc.max_gap > self_env) self_env = P[i].self_sc.max_gap;
    printf("{\"leg\":\"envelope\",\"band\":%.4f,\"self_env\":%.4f}\n", band, self_env);

    /* ---- batch: every prompt at once, cold; each row teacher-forced through the serial lane ---- */
    ds4_batch_ctx *bctx = NULL;
    if (ds4_batch_ctx_create(e, ctx, np < 2 ? 2 : np, 4096, &bctx, err, sizeof(err)) != 0 || !bctx) {
        fprintf(stderr, "probe: batch ctx create failed: %s\n", err); return 1;
    }
    creq_t Q[MAXP];
    memset(Q, 0, sizeof(Q));
    for (int i = 0; i < np; i++) {
        Q[i].tokens = P[i].toks.v; Q[i].n = P[i].toks.len;
        Q[i].images = P[i].n_img ? P[i].spans : NULL; Q[i].image_count = (size_t)P[i].n_img;
        Q[i].max_new = n_gen;
    }
    if (run_batch(bctx, Q, np, err, sizeof(err)) != 0) { fprintf(stderr, "probe: batch failed: %s\n", err); return 1; }
    int batch_fail = 0, batch_identical = 0;
    for (int i = 0; i < np; i++) {
        tf_score sc; memset(&sc, 0, sizeof(sc)); sc.first_flag = -1;
        const int d = Q[i].rejected ? -2 : first_diff(Q[i].out, Q[i].n_out, P[i].serial, P[i].n_serial, eos);
        bool ok = !Q[i].rejected;
        if (ok && teacher_force(e, ctx, &P[i].toks, P[i].spans, P[i].n_img, Q[i].out, Q[i].n_out, band, &sc, err, sizeof(err)) != 0) {
            fprintf(stderr, "probe: teacher-force %d failed: %s\n", i, err); ok = false;
        }
        if (ok) ok = sc.scored > 0 && (sc.flagged == 0 || sc.max_gap <= self_env);
        if (d == -1) batch_identical++;
        printf("{\"leg\":\"batch\",\"prompt\":%d,\"images\":%d,\"bank\":%d,\"rejected\":%s,\"cached\":%d,\"computed\":%d,\"first_diff_vs_serial\":%d,",
               i, P[i].n_img, Q[i].bank_used, Q[i].rejected ? "true" : "false", Q[i].adm_cached, Q[i].adm_computed, d);
        print_score("teacher_forced", &sc); printf(",");
        print_ids("tokens", Q[i].out, Q[i].n_out);
        printf("}\n");
        if (!ok) batch_fail++;
    }
    printf("%s batch rows adjudicate against serial (%d of %d rows inside the serial self-noise envelope %.2f, band %.2f; %d byte-identical)\n",
           batch_fail ? "FAIL" : "PASS", np - batch_fail, np, self_env, band, batch_identical);
    failures += batch_fail != 0;

    /* ---- lineage / payload / warm / reject on the first image prompt ---- */
    int pi = -1;
    for (int i = 0; i < np; i++) if (!P[i].text_only && !Q[i].rejected) { pi = i; break; }
    if (pi < 0) {
        printf("SKIP lineage/payload/warm/reject legs (no image prompt)\n");
    } else {
        const int bank = Q[pi].bank_used;
        const int *hist = NULL;
        const int hl = ds4_batch_ctx_bank_committed(bctx, bank, &hist);
        const ds4_vision_identity *lin = NULL;
        const size_t lin_n = ds4_batch_ctx_bank_lineage(bctx, bank, &lin);
        printf("{\"leg\":\"lineage\",\"bank\":%d,\"committed\":%d,\"lineage\":%zu,\"spans\":%d}\n", bank, hl, lin_n, P[pi].n_img);
        const bool lineage_ok = hl > 0 && lin_n == (size_t)P[pi].n_img;
        printf("%s bank lineage records every image block\n", lineage_ok ? "PASS" : "FAIL");
        failures += !lineage_ok;
        if (lineage_ok) {
            /* payload: stage the bank NOW (history = hl) and restore it into another idle bank */
            const int other = (bank + 1) % ds4_batch_ctx_max_seq(bctx);
            ds4_session_payload_file staged = {0};
            bool pay_ok = ds4_cont_bank_stage_payload(bctx, (uint32_t)bank, &staged, err, sizeof(err)) == 0;
            if (!pay_ok) fprintf(stderr, "probe: stage failed: %s\n", err);
            FILE *fp = pay_ok ? fopen(staged.path, "rb") : NULL;
            if (pay_ok && !fp) { pay_ok = false; fprintf(stderr, "probe: cannot reopen staged payload\n"); }
            if (pay_ok && ds4_cont_bank_restore_payload(bctx, (uint32_t)other, fp, staged.bytes, err, sizeof(err)) != 0) {
                pay_ok = false; fprintf(stderr, "probe: restore failed: %s\n", err);
            }
            if (fp) fclose(fp);
            ds4_session_payload_file_free(&staged);
            const ds4_vision_identity *lin2 = NULL;
            const size_t lin2_n = pay_ok ? ds4_batch_ctx_bank_lineage(bctx, other, &lin2) : 0;
            const int hl2 = pay_ok ? ds4_batch_ctx_bank_committed(bctx, other, NULL) : 0;
            bool lin_same = pay_ok && lin2_n == lin_n && hl2 == hl;
            for (size_t k = 0; lin_same && k < lin_n; k++)
                lin_same = lin2[k].token_start == lin[k].token_start && lin2[k].token_count == lin[k].token_count &&
                           memcmp(lin2[k].identity, lin[k].identity, 32) == 0;
            printf("{\"leg\":\"payload\",\"src\":%d,\"dst\":%d,\"restored\":%s,\"committed_src\":%d,\"committed_dst\":%d,\"lineage_src\":%zu,\"lineage_dst\":%zu,\"lineage_same\":%s}\n",
                   bank, other, pay_ok ? "true" : "false", hl, hl2, lin_n, lin2_n, lin_same ? "true" : "false");

            /* the extension: committed history + a text suffix, REFERENCE spans */
            ds4_tokens ext = {0};
            for (int i = 0; i < hl; i++) ds4_tokens_push(&ext, hist[i]);
            ds4_tokens suf = {0};
            ds4_tokenize_text(e, " Also, in one more sentence:", &suf);
            for (int i = 0; i < suf.len; i++) ds4_tokens_push(&ext, suf.v[i]);
            ds4_tokens_free(&suf);
            const int n_gen2 = n_gen < 24 ? n_gen : 24;
            ds4_vision_span ref[MAXIMG];
            memcpy(ref, P[pi].spans, sizeof(ref));
            for (int k = 0; k < P[pi].n_img; k++) ref[k].embedding.data = NULL;

            /* warm on the live bank */
            creq_t W; memset(&W, 0, sizeof(W));
            W.tokens = ext.v; W.n = ext.len; W.images = ref; W.image_count = (size_t)P[pi].n_img;
            W.place_bank = bank + 1; W.n_cached = hl; W.max_new = n_gen2;
            if (run_batch(bctx, &W, 1, err, sizeof(err)) != 0) { fprintf(stderr, "probe: warm batch failed: %s\n", err); return 1; }
            tf_score wsc; memset(&wsc, 0, sizeof(wsc)); wsc.first_flag = -1;
            bool warm_ok = !W.rejected && W.adm_cached == hl;
            if (warm_ok && teacher_force(e, ctx, &ext, P[pi].spans, P[pi].n_img, W.out, W.n_out, band, &wsc, err, sizeof(err)) != 0) {
                fprintf(stderr, "probe: warm teacher-force failed: %s\n", err); warm_ok = false;
            }
            if (warm_ok) warm_ok = wsc.scored > 0 && (wsc.flagged == 0 || wsc.max_gap <= self_env);
            printf("{\"leg\":\"warm\",\"bank\":%d,\"rejected\":%s,\"cached\":%d,\"computed\":%d,",
                   W.bank_used, W.rejected ? "true" : "false", W.adm_cached, W.adm_computed);
            print_score("teacher_forced", &wsc); printf(",");
            print_ids("tokens", W.out, W.n_out); printf("}\n");
            printf("%s warm re-admit with reference spans (cached=%d of %d; continuation inside the envelope vs serial sync+extend, max gap %.2f)\n",
                   warm_ok ? "PASS" : "FAIL", W.adm_cached, hl, wsc.max_gap);
            failures += !warm_ok;

            /* the same warm re-admit on the RESTORED bank: same state, so byte-identical to the live warm */
            creq_t R; memset(&R, 0, sizeof(R));
            R.tokens = ext.v; R.n = ext.len; R.images = ref; R.image_count = (size_t)P[pi].n_img;
            R.place_bank = other + 1; R.n_cached = hl2; R.max_new = n_gen2;
            int dr = -3;
            if (lin_same) {
                if (run_batch(bctx, &R, 1, err, sizeof(err)) != 0) { fprintf(stderr, "probe: restored warm batch failed: %s\n", err); return 1; }
                dr = R.rejected ? -2 : first_diff(R.out, R.n_out, W.out, W.n_out, eos);
            }
            printf("{\"leg\":\"payload_warm\",\"dst\":%d,\"rejected\":%s,\"cached\":%d,\"first_diff_vs_live_warm\":%d,",
                   other, R.rejected ? "true" : "false", R.adm_cached, dr);
            print_ids("tokens", R.out, R.n_out); printf("}\n");
            const bool pok = lin_same && !R.rejected && R.adm_cached == hl && dr == -1;
            printf("%s bank payload round trip (lineage preserved; restored warm continuation == live warm, byte for byte)\n", pok ? "PASS" : "FAIL");
            failures += !pok;

            /* reject: flipped identity -> cold -> nothing to prefill -> rejected; data-less cold admit -> rejected */
            {
                ds4_vision_span bad[MAXIMG];
                memcpy(bad, ref, sizeof(bad));
                bad[0].embedding.identity[0] ^= 0xff;
                creq_t XY[2]; memset(XY, 0, sizeof(XY));
                XY[0].tokens = ext.v; XY[0].n = ext.len; XY[0].images = bad; XY[0].image_count = (size_t)P[pi].n_img;
                XY[0].place_bank = bank + 1; XY[0].n_cached = ds4_batch_ctx_bank_committed(bctx, bank, NULL); XY[0].max_new = 4;
                XY[1].tokens = P[pi].toks.v; XY[1].n = P[pi].toks.len; XY[1].images = ref; XY[1].image_count = (size_t)P[pi].n_img;
                XY[1].max_new = 4;
                if (run_batch(bctx, XY, 2, err, sizeof(err)) != 0) { fprintf(stderr, "probe: reject batch failed: %s\n", err); return 1; }
                printf("{\"leg\":\"reject\",\"flipped_rejected\":%s,\"cold_dataless_rejected\":%s}\n",
                       XY[0].rejected ? "true" : "false", XY[1].rejected ? "true" : "false");
                const bool rok = XY[0].rejected && XY[1].rejected;
                printf("%s identity mismatch and data-less cold admits are rejected\n", rok ? "PASS" : "FAIL");
                failures += !rok;
            }
            ds4_tokens_free(&ext);
        }
    }
    /* ---- text (last: it re-uses banks by the engine's free-bank pick): the text prompts alone
     * must equal their rows of the mixed batch, byte for byte ---- */
    {
        creq_t T[MAXP]; int nt = 0, map[MAXP];
        memset(T, 0, sizeof(T));
        for (int i = 0; i < np; i++) if (P[i].text_only) {
            T[nt].tokens = P[i].toks.v; T[nt].n = P[i].toks.len; T[nt].max_new = n_gen; map[nt] = i; nt++;
        }
        if (nt > 0) {
            if (run_batch(bctx, T, nt, err, sizeof(err)) != 0) { fprintf(stderr, "probe: text batch failed: %s\n", err); return 1; }
            int tf = 0;
            for (int k = 0; k < nt; k++) {
                const int i = map[k];
                const int d = T[k].rejected ? -2 : first_diff(T[k].out, T[k].n_out, Q[i].out, Q[i].n_out, eos);
                printf("{\"leg\":\"text\",\"prompt\":%d,\"first_diff_vs_mixed\":%d}\n", i, d);
                if (d != -1) tf++;
            }
            printf("%s text rows identical in all-text and mixed batches (%d of %d)\n", tf ? "FAIL" : "PASS", nt - tf, nt);
            failures += tf != 0;
        } else {
            printf("SKIP text leg (no text prompt)\n");
        }
    }

    ds4_batch_ctx_destroy(bctx);
    for (int i = 0; i < np; i++) {
        ds4_tokens_free(&P[i].toks);
        for (int k = 0; k < P[i].n_img; k++) ds4_vision_embedding_free(&P[i].spans[k].embedding);
    }
    ds4_engine_close(e);
    printf("%s\n", failures ? "CONT_LEG_FAIL" : "CONT_LEG_PASS");
    return failures ? 1 : 0;
}
