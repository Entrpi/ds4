#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

/* Reasoning effort, mirroring the three levels the DeepSeek-V4-Flash-0731 model
 * card and its reference encoder (encoding/encoding_dsv4.py) define.  All three
 * are PROMPT PREFIXES prepended at the very start of the conversation in
 * thinking mode -- the checkpoint carries no per-level control token, and its
 * embedded chat template has no reasoning_effort at all, so this is the whole
 * mechanism, not an approximation of one.
 *   DS4_THINK_LOW   upstream "low"  -- adds nothing.  The reference encoder's
 *                                      DEFAULT_REASONING_EFFORT, and ours.
 *   DS4_THINK_HIGH  upstream "high" -- the "Absolute maximum" prefix.
 *   DS4_THINK_MAX   upstream "max"  -- the "Beyond maximum" prefix.
 * Before this change the engine exposed only two non-zero levels and labelled
 * them one tier too low: what it called MAX emitted upstream's *high* string,
 * and upstream's max had no representation at all. */
typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_LOW,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);

typedef enum {
    DS4_DISTRIBUTED_NONE = 0,
    DS4_DISTRIBUTED_COORDINATOR,
    DS4_DISTRIBUTED_WORKER,
} ds4_distributed_role;

typedef struct {
    uint32_t start;
    uint32_t end;
    bool has_output;
    bool set;
} ds4_distributed_layers;

typedef struct {
    ds4_distributed_role role;
    ds4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool replay_check;
    bool debug;
} ds4_distributed_options;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *dspark_path;   /* DSpark/dflash block-drafter GGUF (optional) */
    ds4_backend backend;
    int n_threads;
    int mtp_draft_tokens;
    float mtp_margin;
    const char *directional_steering_file;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    bool warm_weights;
    bool quality;
    bool inspect_only;
    bool load_slice;
    uint32_t load_layer_start;
    uint32_t load_layer_end;
    bool load_output;
    /* inc-14b follow-up: skip the boot prewarm inside ds4_engine_open; the
       caller runs it later via ds4_engine_boot_prewarm.  Servers that budget
       bank placement from free memory must defer so the placement fit reads
       memory BEFORE the prewarm consumes one-time driver costs out of the
       fit's headroom (the prewarm's footprint is post-placement growth by
       design, exactly like the lazy first-forward costs it replaces). */
    bool defer_boot_prewarm;
    ds4_distributed_options distributed;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
/* inc-14b boot prewarm: pay the process's one-time driver costs (graph
   subsystem init, module loads, cuBLAS) with a throwaway two-chunk session
   sync.  Runs inside ds4_engine_open unless opt->defer_boot_prewarm; deferred
   callers invoke this after bank placement.  Idempotent; no-op for CPU
   backends, distributed coordinators, capture-dump boots, and under
   DS4_NO_BOOT_PREWARM=1. */
void ds4_engine_boot_prewarm(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
int ds4_engine_vocab_size(ds4_engine *e);
int ds4_engine_power(ds4_engine *e);
int ds4_engine_set_power(ds4_engine *e, int power_percent);
const char *ds4_engine_model_name(ds4_engine *e);
int ds4_engine_layer_count(ds4_engine *e);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
/* Number of hyper-connection lanes (n_hc); hidden_f32_values / n_hc == n_embd. */
int ds4_engine_n_hc(ds4_engine *e);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
uint32_t ds4_think_effort_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
const char *ds4_think_high_prefix(void);
const char *ds4_think_max_prefix(void);
/* The prefix a mode injects, or "" for NONE/LOW.  Never NULL. */
const char *ds4_think_effort_prefix(ds4_think_mode mode);
/* Uses the active model shape selected by ds4_engine_open(); call after opening
 * the GGUF so Flash/Pro dimensions are known. */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);

/* Phase 2 W1: batched greedy generation.  Ragged-prefills `n` prompts in one
 * forward, then batch-decodes all sequences with compact-on-finish.  Each
 * out[i].tokens is a malloc'd stream the CALLER must free(); out[i].finish is 1
 * when the sequence hit EOS, 0 when it hit the token budget. */
typedef struct {
    int max_new_tokens;   /* per-sequence decode budget (>=1) */
    int eos_id;           /* sequence-ending token; <0 => engine default */
} ds4_batch_gen_options;

typedef struct {
    int *tokens;          /* malloc'd generated tokens; caller frees */
    int  n_tokens;        /* count generated (<= max_new_tokens) */
    int  finish;          /* 1 = hit EOS, 0 = hit budget */
} ds4_batch_gen_result;

int ds4_engine_batched_generate(ds4_engine *e, const ds4_tokens *prompts, int n,
                                int ctx_size, const ds4_batch_gen_options *opts,
                                ds4_batch_gen_result *out,
                                char *err, size_t errlen);
/* Per-sequence variant: max_new_tokens[i]/eos_ids[i] are length-n arrays
 * (max_new_tokens entry <=0 => 1; eos_ids may be NULL or an entry <0 => engine
 * default EOS).  Used by the server's request-coalescing path. */
int ds4_engine_batched_generate_ex(ds4_engine *e, const ds4_tokens *prompts, int n,
                                   int ctx_size,
                                   const int *max_new_tokens, const int *eos_ids,
                                   ds4_batch_gen_result *out,
                                   char *err, size_t errlen);

/* Phase 2 W4: persistent batched-generation context.  Allocates the graph + N KV
 * bank slabs ONCE (sized for up to max_seq sequences and max_total_tokens packed
 * prompt tokens at the given ctx_size) and reuses them across batches, removing
 * the per-batch graph/slab alloc from the server's hot path.  Opaque handle. */
typedef struct ds4_batch_ctx ds4_batch_ctx;
int  ds4_batch_ctx_create(ds4_engine *e, int ctx_size, int max_seq, int max_total_tokens,
                          ds4_batch_ctx **out, char *err, size_t errlen);
/* R5 Inc1a: like ds4_batch_ctx_create, but treats max_seq as a CAP and sizes
 * the bank count DOWN to (free device memory - headroom) / per-bank-bytes
 * before allocating, instead of the caller probing by failing whole creates
 * (on unified memory those probes can summon the OOM killer before they
 * fail).  Residual slab failures descend by 3/4 internally.  Knobs:
 * DS4_BATCH_FIT=0 keeps caller-driven sizing, DS4_BATCH_FIT_HEADROOM_MB
 * (default 8192) reserves runtime growth room.  Backends with no memory
 * query (Metal) skip the budget.  Read the chosen width back with
 * ds4_batch_ctx_max_seq.
 * R5 Inc1b: on backends with VMM (CUDA) the ctx-scaled compressed/indexer
 * cache slabs are demand-mapped virtual reservations by default -- they cost
 * the bank-count budget nothing at boot and map physical pages only as
 * conversations grow, gated per-admission against the remaining free memory
 * (rejected admits report a comp-cache-budget error; mapping is grow-only, so
 * bank reuse rides previously mapped pages).  DS4_BATCH_VMM_COMP=0 forces
 * eager ctx-sized slabs, bit-identical to pre-Inc1b sizing. */
int  ds4_batch_ctx_create_fit(ds4_engine *e, int ctx_size, int max_seq, int max_total_tokens,
                              ds4_batch_ctx **out, char *err, size_t errlen);
void ds4_batch_ctx_destroy(ds4_batch_ctx *ctx);
/* deepmem lite-4 (minimal reclaim): trim FREE banks' demand-mapped cache
   pages to recover up to want_bytes for a non-batch consumer (the serial
   lane).  Context-owned mechanics; the CALLER owns exclusion (the server
   calls under gen_mu with no continuous pass running).  Victim order:
   invalid history first, then shortest committed history; a trimmed bank's
   history is invalidated, so warm admits cannot reuse it (its next tenant
   cold-resets).  Returns bytes released (also counted in the ctx trim
   counters and the "batch vmm: trimmed" log line); 0 when nothing is
   trimmable or DS4_BATCH_VMM_TRIM=0. */
uint64_t ds4_batch_ctx_trim_free(ds4_batch_ctx *ctx, uint64_t want_bytes);

/* deepmem D-1c: page-interval union across per-bank credited spans -- the
 * pure arithmetic under the continuous-admission credit projection
 * (credit_union_family in ds4.c), extracted header-inline so unit tests can
 * drive it without a GPU and production cannot drift from the tested code.
 * Walks banks in index order (offsets b*stride are monotonic), rounds each
 * credited span to whole pages, merges runs that touch (neighbor banks can
 * share an edge page because strides are not page-aligned), and calls
 * run_need(user, run_off_bytes, run_span_bytes) once per merged page-aligned
 * run, summing its returns.  run_need reports the bytes of the run still to
 * be charged (production: span - resident, floored at 0).  Bank `cand`
 * contributes cand_len instead of credit[cand]; a zero-length span
 * contributes nothing. */
static inline uint64_t ds4_credit_union_runs(
        uint64_t stride, uint64_t row_bytes, uint64_t cap_rows, uint32_t ratio,
        uint64_t page, const uint64_t *credit, uint32_t nbanks,
        uint32_t cand, uint64_t cand_len,
        uint64_t (*run_need)(void *user, uint64_t off, uint64_t span),
        void *user) {
    if (page == 0 || ratio == 0 || row_bytes == 0 || !run_need) return 0;
    uint64_t need = 0, run_p0 = 0, run_p1 = 0;
    bool run_open = false;
    for (uint32_t b = 0; b < nbanks; b++) {
        const uint64_t tlen = b == cand ? cand_len : credit[b];
        if (tlen == 0) continue;
        uint64_t rows = tlen / ratio + 2u;
        if (rows > cap_rows) rows = cap_rows;
        if (rows == 0) continue;
        const uint64_t off = (uint64_t)b * stride;
        const uint64_t len = rows * row_bytes;
        const uint64_t p0 = off / page, p1 = (off + len - 1u) / page;
        if (run_open && p0 <= run_p1) {
            if (p1 > run_p1) run_p1 = p1;
        } else {
            if (run_open)
                need += run_need(user, run_p0 * page,
                                 (run_p1 - run_p0 + 1u) * page);
            run_p0 = p0;
            run_p1 = p1;
            run_open = true;
        }
    }
    if (run_open)
        need += run_need(user, run_p0 * page, (run_p1 - run_p0 + 1u) * page);
    return need;
}
/* Bank count of the persistent ctx (create_fit may size it below the
 * requested cap).  Returns 0 if ctx is NULL. */
int  ds4_batch_ctx_max_seq(const ds4_batch_ctx *ctx);
/* SWA raw ring rows per bank of the persistent ctx.  R2: this is the RING size
 * only -- the STATIC batch path (ds4_engine_batched_generate_ctx) still bounds
 * prompt+budget by it (one-shot prefill, no wrap), but the continuous path's
 * per-sequence bound is ds4_batch_ctx_seq_cap.  Returns 0 if ctx is NULL. */
int  ds4_batch_ctx_raw_cap(const ds4_batch_ctx *ctx);
/* Per-sequence committed-token bound (prompt + generation) of the CONTINUOUS
 * path: the admit pre-check + decode budget cap.  With chunked admission
 * (DS4_CONT_PREFILL_CHUNK > 0, the default) the raw ring wraps and this is the
 * ctx size; legacy one-shot admission (=0) keeps the historical raw-ring bound.
 * Returns 0 if ctx is NULL. */
int  ds4_batch_ctx_seq_cap(const ds4_batch_ctx *ctx);
/* Batched generation over a persistent context (W4); same semantics as
 * ds4_engine_batched_generate_ex but reuses ctx's graph + slabs.  n <= ctx max_seq,
 * Σ(prompt len) <= ctx max_total_tokens. Returns 0 on success. */
int  ds4_engine_batched_generate_ctx(ds4_batch_ctx *ctx, const ds4_tokens *prompts, int n,
                                     const int *max_new_tokens, const int *eos_ids,
                                     ds4_batch_gen_result *out, char *err, size_t errlen);

/* Phase 2 W5: continuous batching (mid-flight admit/evict) over a persistent ctx.
 * The scheduler maintains a rolling active set of up to ctx max_seq sequences: each
 * step it admits waiting requests into freed KV banks (ragged-prefill the prompt)
 * and evicts finished ones, so short requests don't wait for long ones.  CUDA
 * backend only (the Metal path ignores per-seq bank ids).
 * W7: per-sequence sampling -- each request carries its own temperature/top-k/
 * top-p/min-p/seed, sampled with an independent RNG stream so concurrent rows in
 * one batch do not perturb each other.  A zeroed sampling block (temperature<=0)
 * is greedy argmax, bit-identical to the W5/W6 default. */
typedef struct {
    const int *tokens;   /* prompt tokens (caller-owned; must outlive the admit) */
    int        n;        /* prompt length (>0, <= ctx raw_cap) */
    int        max_new;  /* per-seq decode budget (>=1) */
    int        eos;      /* per-seq EOS token; <0 => engine default */
    void      *user;     /* opaque handle echoed back to on_done */
    /* W7 per-seq sampling (zeroed => greedy argmax, the W5/W6 default):       */
    float      temperature; /* <= 0 => greedy argmax (ignores the rest)        */
    int        top_k;       /* <= 0 => full vocab                              */
    float      top_p;       /* nucleus; <=0 or >1 treated as 1.0               */
    float      min_p;       /* relative floor; <0 => 0                         */
    uint64_t   seed;        /* per-seq RNG seed (caller resolves 0 if it wants */
                            /* distinct streams; 0 is a fixed, valid sequence) */
    /* Per-token sampling override (NULL = none).  When set, the engine calls
     * this immediately before sampling EACH of the row's tokens (seed token
     * and every decode/accept step); a nonzero return forces that one token
     * to greedy argmax (temperature 0) while the rest of the sampling block
     * still applies to non-overridden tokens.  This is how a caller samples
     * structural tool-call syntax deterministically while payload keeps the
     * request's own params (the serial path's per-token DSML override).
     * Argmax consumes NO RNG draw, and the caller's decision may depend only
     * on tokens already reported via on_token -- so seeded streams stay
     * aligned between the plain and speculative decode paths.  `ud`/`user`
     * are the same handles on_token receives. */
    int      (*sample_override)(void *ud, void *user);
    /* v0.5.2: liveness probe for the ADMISSION PREFILL phase (may be NULL).
     * Polled between prefill chunks; return 0 when the request's client is
     * gone -- the engine abandons the pending admission (bank reset to free)
     * and calls on_done(user, tokens, 0, 0) with a non-NULL empty tokens
     * array (aborted, NOT rejected: the slot must not fall to the serial
     * path).  Decode-phase aborts stay on_token's business. */
    int      (*alive)(void *ud, void *user);
    /* v0.5.6 Inc 4a: transport-after-admission signal (may be NULL).  Called
     * EXACTLY ONCE, on the generate thread, after this request's bank install
     * fully succeeded -- placement final, warm/fork validation resolved,
     * lifetime credit installed, bank state mutated -- and BEFORE any of its
     * prefill chunks run.  n_cached/n_computed are the engine's authoritative
     * split of the prompt (n_cached + n_computed == n; a degraded warm admit
     * reports n_cached = 0), bank is the placement.  This is the earliest
     * moment a caller may commit client-visible transport (SSE headers /
     * protocol start events): a request the engine REJECTS never gets this
     * call, so rejection fallback paths stay transport-clean.  Return 1 to
     * proceed; 0 to cancel the admission before any prefill is spent -- the
     * bank unwinds exactly like an alive() abort (a warm bank keeps its
     * committed prefix, a cold install resets to free) and on_done reports an
     * abort (non-NULL empty tokens, n=0, finish=0), never a reject. */
    int      (*on_admitted)(void *ud, void *user, int n_cached,
                            int n_computed, int bank);
    /* A2a warm start.  Zero-init = engine-managed cold admit (the W5..W7
     * behavior, unchanged).  place_bank is a bank id + 1 placement directive
     * (0 = engine picks the first free bank); it lets the caller route a
     * request to a specific FREE bank -- warm continuation, or directed cold
     * placement away from valuable retired banks.  n_cached > 0 requests a
     * WARM admit into bank place_bank-1: tokens[0..n_cached) must equal that
     * bank's committed history exactly (ENGINE-VALIDATED against its own
     * per-bank record; any mismatch degrades to a cold reset, never reuses a
     * non-matching cache), and only tokens[n_cached..n) are prefilled. */
    int        place_bank;  /* bank id + 1; 0 = engine's choice               */
    int        n_cached;    /* committed prefix length in bank place_bank-1;  */
                            /* 0 = cold admit                                  */
    int       *bank_used;   /* OUT (optional): engine writes the bank id this */
                            /* request was placed in, at admit time           */
    /* A2b fork-by-copy.  fork_bank = source bank id + 1 (0 = no fork): the
     * request's tokens[0..n_cached) must equal the SOURCE bank's committed
     * history (ENGINE-VALIDATED, like warm; the source must also be idle --
     * not generating).  The engine D2D-copies the source bank's committed
     * state into the target bank (place_bank directive or engine's pick) and
     * prefills only tokens[n_cached..n), leaving the source bank untouched --
     * N requests sharing a long prefix pay one prefill + N cheap copies.
     * Any validation failure degrades to a cold admit.  When fork_bank > 0,
     * n_cached describes the SOURCE bank (warm matching is skipped); if the
     * target resolves to the source itself the fork becomes a plain warm
     * admit (no copy).
     * P1 (partial-prefix fork): n_cached BELOW the source's committed length
     * requests reuse of just the shared prefix -- the request DIVERGES from
     * the source mid-prompt.  The engine rewinds to the replay base
     * R = (n_cached - 4) aligned down to the model's largest compress ratio
     * (128 on Flash: the boundary where every layer's in-progress pooling
     * group is rebuildable), validates tokens[0..R+4) against the source
     * history, clones only the state below R, and re-prefills the shared
     * tail [R, n_cached) together with the divergent suffix -- so the
     * admit's pos_base is R, not n_cached, and cuts below ~(align + 4)
     * tokens degrade to cold (no reuse worth having there anyway).  src ==
     * target is an in-place truncate-reuse (no copies; the bank's committed
     * state rewinds to R).  Unsatisfiable cuts (too short, or a wrapped
     * source ring that no longer covers the replay's attention window)
     * degrade to cold like any other validation failure. */
    int        fork_bank;   /* source bank id + 1; 0 = no fork                */
} ds4_cont_request;
/* A2a: a bank's committed token history (engine-authoritative bookkeeping for
 * warm start).  *toks points at ctx-owned storage, valid until the next admit
 * or reset that touches the bank; returns the committed length, 0 when the
 * bank is out of range or its state is not reuse-trustworthy (engine failure,
 * static-path reuse of the slabs, deferred-commit MTP path). */
int  ds4_batch_ctx_bank_committed(const ds4_batch_ctx *ctx, int bank,
                                  const int **toks);
/* v0.5.6 Inc 5a (continuation registry, plan §4.6): engine-authoritative bank
 * lineage generation.  Advances on every event after which previously
 * committed bank content may no longer be what a past reader saw: lineage
 * replacement (new admission reset), fork-by-copy into the bank, payload
 * restore, trim/reclaim, quarantine, history-capacity overflow, and the
 * deferred-commit invalidate.  Pure committed extension does NOT advance.
 * A reader that recorded (generation, committed length) revalidates the
 * exact content with equality checks alone -- no memcmp, no ABA.  Returns 0
 * for NULL ctx or out-of-range bank (0 is never a live generation). */
uint64_t ds4_batch_ctx_bank_generation(const ds4_batch_ctx *ctx, int bank);
/* admit: fill *req for the next waiting request and return 1; return 0 when none is
 *   available right now (the loop keeps decoding the active set and ends once the
 *   active set is empty AND admit returns 0).
 * on_token (may be NULL): called once per newly sampled NON-EOS token, in order,
 *   for the sequence identified by `user` (seed token then each decode step).
 *   Return 1 to keep generating, 0 to ABORT that sequence now (e.g. its client
 *   disconnected) -- the engine evicts it this step and still calls on_done
 *   (finish=0).  NULL disables streaming (pure buffer-then-on_done, the W5 path).
 * on_done: a sequence finished -- tokens[0..n) is its full generation (caller must
 *   NOT free; valid only during the call), finish=1 if it hit EOS (0 = budget/abort).
 * Returns 0 on success. */
int  ds4_engine_continuous_generate(ds4_batch_ctx *ctx,
                                    int (*admit)(void *ud, ds4_cont_request *req),
                                    int (*on_token)(void *ud, void *user, int token),
                                    void (*on_done)(void *ud, void *user,
                                                    const int *tokens, int n, int finish),
                                    void *ud, char *err, size_t errlen);

/* =========================================================================
 * Live serving metrics (v0.2.x observability): ONE registry, THREE porcelains.
 *
 * A single global registry of monotonic counters + gauges, incremented by the
 * engine and the server at event sites and rendered by the server's three
 * user-facing surfaces (per-response `timings`, GET /metrics, GET /v1/stats).
 * No surface computes its own numbers -- everything reads this registry.
 *
 * Concurrency model: every hot writer runs under the server's gen_mu (the
 * continuous loop, the serial path, the static batch path), so writes are
 * effectively single-threaded; the cold writers (request accounting on client
 * threads) and the readers (HTTP threads rendering /metrics -- which must
 * NEVER block on generation: gen_mu is held for minutes at deep ctx) are
 * cross-thread.  All accesses go through the relaxed-atomic helpers below --
 * one relaxed add per decode step is free next to a multi-ms GPU step.
 *
 * The rolling window is DS4_METRICS_WIN_SECONDS of per-second buckets feeding
 * the decode/prefill rate gauges; a reader racing a bucket reset can see one
 * partially-cleared second, which is acceptable noise for a rate gauge. */
#define DS4_METRICS_WIN_BUCKETS 64
#define DS4_METRICS_WIN_SECONDS 60
/* Route observation matrix (v0.5.6 Inc 0a): requests by wire surface x
 * serving lane.  Storage only -- the SERVER owns both index sets and their
 * label names (the engine stays protocol-blind); one increment per job at
 * the moment a lane takes it, so a batched attempt that falls back re-enters
 * serial and increments both lanes (lane entries, not final outcomes). */
#define DS4_METRICS_ROUTE_SURFACES 4
#define DS4_METRICS_ROUTE_LANES 3
/* Route decisions (v0.5.6 Inc 2a): one increment per dispatch-time
 * route_decide() outcome, keyed by the FIRST blocking reason (or the chosen
 * batched lane).  Bounded label set owned by the server; storage only here.
 * Decisions differ from route_requests: that matrix counts LANE ENTRIES
 * (a failed batched attempt re-enters serial and counts twice); this vector
 * counts each request's single dispatch decision. */
#define DS4_METRICS_ROUTE_REASONS 14
/* Admission-bound sheds (v0.5.6 Inc 2e): one increment per request refused
 * (or slow-reader evicted) by an explicit server bound.  Bounded label set
 * owned by the server (shed_reason_names); storage only here.
 * Inc 5b adds continuation_hold: serial work refused because the session is
 * reserved for a live tool continuation (grace window or queued hard pin). */
#define DS4_METRICS_SHED_REASONS 6
typedef struct {
    uint64_t stamp;               /* monotonic second this bucket belongs to */
    uint64_t dec_tok, dec_steps, pf_tok;
} ds4_metrics_bucket;
typedef struct {
    /* requests (server-incremented; one increment per generation request) */
    uint64_t requests_started;
    uint64_t requests_completed;            /* a response was written */
    uint64_t requests_failed;               /* a 5xx/prefill-failure was written */
    uint64_t requests_canceled;             /* client left before delivery (Inc 2c) */
    uint64_t requests_refused_deep_serial;  /* deep-serial guard 503s */
    uint64_t requests_serial;               /* served on the legacy serial path */
    uint64_t requests_inflight;             /* gauge */
    /* engine admission + refusals */
    uint64_t graph_fit_refusals;            /* session-graph fit gate said no */
    uint64_t cont_admit_rejects;            /* comp-cache budget rejects */
    uint64_t cont_batch_failures;           /* continuous run ended in error */
    uint64_t admits_cold, admits_warm, admits_fork;
    uint64_t admits_partial_fork, admits_partial_truncate;
    /* tokens */
    uint64_t tokens_prefilled_computed;     /* prompt tokens actually forwarded */
    uint64_t tokens_prefilled_cached;       /* prompt tokens reused from KV */
    uint64_t tokens_decoded;                /* generated tokens (all paths) */
    uint64_t decode_steps;                  /* batched forward steps */
    /* speculation (DSpark / MTP accept path) */
    uint64_t spec_drafts, spec_hits, spec_quench;
    /* gauges */
    uint64_t banks_live;                    /* banks decoding right now */
    uint64_t banks_total;                   /* persistent ctx bank count */
    uint64_t warm_records;                  /* valid warm-start records */
    uint64_t kv_pages_resident;             /* demand-mapped comp/index pages */
    uint64_t boot_stamp;                    /* monotonic second at server boot */
    /* aligned-artifact perf tier (set once at boot): 0=none (raw-layout
     * dispatch), 1=imported from ds4_weight_server, 2=built in-process */
    uint64_t derived_artifact_source;
    uint64_t derived_artifacts;             /* artifact count */
    uint64_t derived_artifact_bytes;
    /* route observation (see DS4_METRICS_ROUTE_* above) */
    uint64_t route_requests[DS4_METRICS_ROUTE_SURFACES][DS4_METRICS_ROUTE_LANES];
    uint64_t route_decisions[DS4_METRICS_ROUTE_REASONS];
    /* admission-bound sheds (v0.5.6 Inc 2e; see DS4_METRICS_SHED_REASONS) */
    uint64_t requests_shed[DS4_METRICS_SHED_REASONS];
    uint64_t out_backlog_bytes;   /* gauge: stream bytes buffered for slow readers */
    /* continuation registry (v0.5.6 Inc 5a; server-owned semantics) */
    uint64_t creg_published;      /* records published at tool-turn terminals */
    uint64_t creg_resolved;       /* T2 admissions served from a LIVE record */
    uint64_t creg_missed;         /* T2 live-state requests refused 409 */
    uint64_t creg_demoted;        /* LIVE -> REPLAY_ONLY transitions */
    uint64_t creg_records_live;   /* gauge: LIVE_FRONTIER records right now */
    /* v0.5.6 Inc 7c: host-side projection cost inside the engine's per-token
     * callback (cont_on_token runs under gen_mu on the worker thread -- detok,
     * semantic accumulator, wire projection into j->out).  Nanoseconds so the
     * per-token quotient stays integer-exact; tokens counts only callbacks
     * that did host work (needs-text rows).  The plan's "measure projection
     * under gen_mu" gate reads these to decide whether offload is warranted. */
    uint64_t cont_ontoken_ns;
    uint64_t cont_ontoken_tokens;
    /* v0.5.6 Inc 7c: /v1/batch fairness observation -- how long the endpoint
     * waited on gen_mu before its turn (the continuous epoch holds gen_mu
     * across its whole rolling loop, so this is the starvation signal the
     * cut-list check reads). */
    uint64_t batch_genmu_wait_ns;
    uint64_t batch_genmu_waits;
    ds4_metrics_bucket win[DS4_METRICS_WIN_BUCKETS];
} ds4_metrics;
ds4_metrics *ds4_metrics_get(void);
static inline void ds4_metric_add(uint64_t *c, uint64_t v) {
    __atomic_fetch_add(c, v, __ATOMIC_RELAXED);
}
static inline void ds4_metric_set(uint64_t *c, uint64_t v) {
    __atomic_store_n(c, v, __ATOMIC_RELAXED);
}
static inline uint64_t ds4_metric_read(const uint64_t *c) {
    return __atomic_load_n(c, __ATOMIC_RELAXED);
}
/* Record dec_tok/dec_steps/pf_tok into the current second's bucket. */
void ds4_metrics_window_add(uint64_t dec_tok, uint64_t dec_steps, uint64_t pf_tok);
/* Rates over the trailing window (clamped to time since boot_stamp):
 * decode tok/s, prefill tok/s, and decoded tokens per step.  Any out
 * pointer may be NULL. */
void ds4_metrics_window_rates(double *dec_tok_s, double *pf_tok_s, double *tok_per_step);

/* Per-sequence serving stats for the per-response `timings` porcelain.
 * Valid ONLY while an on_done callback with real tokens is executing: the
 * continuous loop fills the ctx's last-done slot immediately before each such
 * callback (rejected admits -- on_done(NULL) -- leave it untouched).  All
 * times are now_sec()-domain (CLOCK_MONOTONIC seconds). */
typedef struct {
    double   admit_sec;        /* admission install (prefill queue entry) */
    double   first_token_sec;  /* seed token sampled = prefill complete */
    double   done_sec;
    uint32_t prefill_cached;   /* reused prefix (warm/fork/partial cut) */
    uint32_t prefill_computed; /* suffix tokens actually forwarded */
    uint32_t decode_tokens;    /* emitted tokens (seed + decode/accepts) */
    uint32_t decode_steps;     /* steps this bank participated in */
    uint64_t spec_drafts, spec_hits;   /* this sequence's draft rows / accepts */
} ds4_cont_seq_stats;
/* Copy the finishing sequence's stats; returns 1 when set (see above), 0
 * when no completed-sequence stats are available. */
int ds4_cont_last_done_stats(const ds4_batch_ctx *ctx, ds4_cont_seq_stats *out);

/* Phase 2 S1.1: deterministic MTP gate.  Drives the continuous engine over a fixed
 * set of synthetic prompts (deterministic admission) and asserts the per-seq output
 * tokens are identical with the per-bank MTP draft path off vs on -- the clean
 * non-invasiveness/exactness proof (no server-timing / batch-composition confound).
 * Requires a ctx created with --mtp.  Returns 0 PASS, 1 token MISMATCH, 2 setup error. */
int  ds4_cont_mtp_gate(ds4_batch_ctx *ctx, char *err, size_t errlen);

/* Phase 2 A2a: deterministic warm-start gate.  Drives the continuous engine with
 * fixed REAL-TEXT prompts (confident greedy margins, so cross-packing token
 * comparison is meaningful) and asserts (a) STRUCTURAL: an isolated warm suffix
 * prefill leaves the committed compressed-cache frontier (per-layer counts)
 * exactly equal to a cold full prefill, at two group alignments; (b) a warm
 * admit's token stream matches a cold full prefill of the same effective prompt,
 * including a chained second warm turn and a LONG suffix; (c) a non-matching
 * cached prefix is rejected and degrades to a byte-identical cold run; (d) two
 * banks warm in one run with out-of-order placement directives.  A2b adds the
 * fork-by-copy phases: (e) a fork admit (D2D bank copy + suffix prefill) is
 * STRUCTURALLY frontier-exact vs a cold prefill and token-matches it, including
 * a second fork from the same source (fan-out reuse); (f) the source bank still
 * warm-continues byte-identically after serving two forks; (g) a fork with a
 * mutated cached token is rejected and degrades to cold.  Needs only a batch
 * ctx (no --mtp).  Returns 0 PASS, 1 MISMATCH, 2 setup error. */
int  ds4_cont_warm_gate(ds4_batch_ctx *ctx, char *err, size_t errlen);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
/* Standalone DSpark/dflash drafter GGUF load + strict layout validation (D1 gate).
 * Low-RAM: opens only the drafter file, no base model required. Returns 0 on OK. */
int ds4_dspark_validate(const char *path);
/* DSpark GPU block-forward accept gate (D4.4): replays a DS4DSPK1 hidden trace
 * through the in-engine drafter block forward + Markov refine and reports
 * pos-0 accept / mean commit (target: ~0.875 / ~3.12). Needs base + drafter
 * loaded + a live session. Returns 0 on success. */
int ds4_dspark_block_validate(ds4_engine *e, ds4_session *s, const char *trace_path);
/* DSpark single-forward target-hidden capture gate (D4.5a): live-captures the
 * mean-HC hidden at layers 40/41/42 from each trace record's tokens and compares
 * against the trace's 3-slice hidden. Near-zero diff proves the inline serving
 * capture is faithful. Returns 0 on success. */
int ds4_dspark_capture_validate(ds4_engine *e, ds4_session *s, const char *trace_path);
/* DSpark per-bank injected-KV ring isolation gate (D4.5b): injects different
 * records into different banks and verifies cross-bank ring isolation. 0 = pass. */
int ds4_dspark_slabs_validate(ds4_engine *e, ds4_session *s, const char *trace_path);
/* DSpark inline capture-tap gate (D4.5c): runs the production batched forward with
 * the layer-40/41/42 capture hook on and compares to the trace's 3-slice hidden. */
int ds4_dspark_tap_validate(ds4_engine *e, ds4_session *s, const char *trace_path);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_effort_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode mode);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
int ds4_session_power(ds4_session *s);
int ds4_session_set_power(ds4_session *s, int power_percent);
bool ds4_session_is_distributed(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);
/* Distributed coordinator sessions return 1 when the full layer route is
 * available, 0 when it is still incomplete, and -1 for a local API error. */
int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
/* Per-call speculative-decode accounting for the serial path.  `drafted` is
 * the number of MTP draft tokens proposed (including the first proposed
 * draft), and `accepted` is the number of those drafts that were verified and
 * committed (the unconditionally committed first_token is NOT counted).
 * Mirrors the batch path's ds4_cont_seq_stats.spec_drafts/spec_hits. */
typedef struct {
    int drafted;
    int accepted;
} ds4_spec_stats;

int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        ds4_spec_stats *stats,
                                        char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
/* v0.5.6 Inc 5a (continuation registry, plan §4.6): engine-authoritative
 * session content generation.  Advances whenever previously committed
 * checkpoint content may change: create, invalidate, rewind below the live
 * position, a non-extending sync (rebuild), and payload load.  Pure
 * extension does NOT advance, so (generation, pos) identifies exact
 * committed content and a reader that recorded both revalidates with
 * equality checks alone.  Returns 0 for NULL (never a live generation). */
uint64_t ds4_session_generation(const ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
/* v0.5.2 serial right-sizing: whether the session's lazy graph alloc is
 * still deferred, and whether a session graph at ctx_size would pass the fit
 * gate right now (quiet probe; fail-open like the gate itself). */
int ds4_session_graph_pending(const ds4_session *s);
int ds4_engine_session_graph_fits(ds4_engine *e, int ctx_size);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);
int ds4_session_output_head_bench(ds4_session *s, int iters, FILE *fp, char *err, size_t errlen);

/* Low-level graph slice entry points used by distributed inference.  The
 * transport/session routing logic lives in ds4_distributed.c. */
int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen);
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen);
int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(3)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
/* v3: one u32 of row-format flags follows the per-layer row-count arrays.
 * Packed-primary writers serialize the mirror codes+scales verbatim
 * (~3x smaller than the v2 F32 expansion; restore uploads them without
 * re-encoding).  v2 payloads (F32 rows) remain readable forever. */
#define DS4_SESSION_PAYLOAD_ROWS_FP8_PACKED (UINT32_C(1) << 0)
#define DS4_SESSION_PAYLOAD_ROWS_FP4_PACKED (UINT32_C(1) << 1)
#define DS4_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define DS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
#define DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);

/* Durable pinned banks (v0.3): serialize / restore one cont BANK of a
 * batch ctx through the same wire format as a serial session payload (a
 * bank record is a valid serial checkpoint; its logits block and MTP tail
 * are zeros — restore is warm-admit + suffix prefill, which regenerates
 * both).  Save reads the bank's committed token history (bank_hist);
 * restore repopulates tensors + counters + bank_hist and marks the bank
 * warm-valid, so a following admit validates exactly like a live warm
 * bank.  Banks must be idle (evict/shutdown by construction); both calls
 * run under the engine generation lock like every other cont entry. */
uint64_t ds4_cont_bank_payload_bytes(ds4_batch_ctx *ctx, uint32_t bank);
int ds4_cont_bank_save_payload(ds4_batch_ctx *ctx, uint32_t bank,
                               FILE *fp, char *err, size_t errlen);
int ds4_cont_bank_restore_payload(ds4_batch_ctx *ctx, uint32_t bank,
                                  FILE *fp, uint64_t payload_bytes,
                                  char *err, size_t errlen);
/* (Committed token history reads through the existing
 * ds4_batch_ctx_bank_committed accessor.) */
/* Stage a bank payload into a temp file (bank twin of
 * ds4_session_stage_payload); free with ds4_session_payload_file_free. */
int ds4_cont_bank_stage_payload(ds4_batch_ctx *ctx, uint32_t bank,
                                ds4_session_payload_file *out,
                                char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

#endif
