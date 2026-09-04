#ifndef DS4_MEM_CENSUS_H
#define DS4_MEM_CENSUS_H

#include <stdint.h>

/* =========================================================================
 * memgov D0a-1: allocation-census core (types + checked arithmetic).
 *
 * A deliberately dependency-free leaf header: ds4_cuda.cu carries its own
 * signatures instead of including ds4.h/ds4_gpu.h (the v0.5.6.1 Metal-PR
 * convention), so the census types it shares with the engine and the unit
 * suite live here, the same way cuda/mmq/ds4_mmq.h is shared.  ds4.h
 * includes this header and layers the backend API declarations on top.
 *
 * Census: local/docs/v057/d0a_census_table.md classifies every CUDA
 * allocation site into the fixed consumer classes below.  The CUDA backend
 * owns one cell per class x domain and records {requested, committed} at
 * each allocation/release site.  NO decision path reads these cells --
 * D0a's contract is zero decision change; enforcement arrives with the
 * governor (D0b/D2), which subsumes these classes as lease consumers.
 *
 * Vocabulary (plan sec 4): `requested` = what the consumer asked for,
 * including any bump-allocator alignment it consumed; `committed` = what
 * the allocator physically committed, including chunk/granularity rounding.
 * slack = live committed - live requested: rounding + arena tail the box
 * pays for but no consumer asked for.  Frees are recorded on the same
 * cell; over-free clamps and counts a fault rather than wrapping (checked
 * saturating arithmetic, plan sec 6.2 -- an impossible state must be
 * visible, never a negative ledger). */
typedef enum {                       /* FIXED cardinality -- census sec 6.
                                        ENGINE_OTHER is deliberately the
                                        ZERO value: a zero-initialized
                                        class tag reads as "untagged", and
                                        the close gate asserts that class
                                        settles to zero live bytes. */
    DS4_MEMC_ENGINE_OTHER = 0,       /* untagged funnel fallback           */
    DS4_MEMC_WEIGHT_ARENA,           /* VMM/cudaMalloc weight arena chunks */
    DS4_MEMC_WEIGHT_SPAN,            /* direct per-span device copies      */
    DS4_MEMC_WEIGHT_WHOLE,           /* whole-model device copies          */
    DS4_MEMC_WEIGHT_DERIVED,         /* q8->f16/f32 dequant caches         */
    DS4_MEMC_WEIGHT_IMPORT,          /* VMM/IPC weight-server imports      */
    DS4_MEMC_WEIGHT_ARTIFACT,        /* in-process aligned artifacts       */
    DS4_MEMC_WEIGHT_HOST_PIN,        /* model-map host registrations       */
    DS4_MEMC_STAGE_PIN,              /* pinned staging pools               */
    DS4_MEMC_SCALARS_MIRROR,         /* decode/layer/row scalar mirrors    */
    DS4_MEMC_SESSION_TENSORS,        /* engine graph activations/state     */
    DS4_MEMC_KV_PRIMARY,             /* raw ring + compressed KV           */
    DS4_MEMC_BATCH_BANK,             /* bank comp/index slabs (VMM demand) */
    DS4_MEMC_SCRATCH_STICKY,         /* grow-only sticky scratches         */
    DS4_MEMC_KERNEL_PARTIALS,        /* boot-fixed kernel buffers + ws     */
    DS4_MEMC_GRAPH_EXEC,             /* instantiated graphs (empirical)    */
    DS4_MEMC_DIAG,                   /* env-gated selftest/verify buffers  */
    DS4_MEMC_VISION_ENCODER,         /* Track B inc 9: encoder scratch     */
    DS4_MEMC__COUNT
} ds4_mem_consumer_class;

typedef enum {
    DS4_MEMD_UNIFIED_DEVICE = 0,     /* CUDA allocs; aliases system RAM on
                                        integrated (GB10)                  */
    DS4_MEMD_PINNED_HOST,            /* pinned allocs + registrations      */
    DS4_MEMD__COUNT
} ds4_mem_domain;

typedef struct {
    uint64_t requested;              /* monotonic sum of consumer asks     */
    uint64_t committed;              /* monotonic sum of allocator commits */
    uint64_t freed_requested;        /* monotonic sum of released asks     */
    uint64_t freed_committed;        /* monotonic sum of released commits  */
    uint64_t alloc_calls, free_calls;
} ds4_mem_cell;

/* Saturating add: on overflow, pin at UINT64_MAX and count a fault.  A
 * saturated monotonic counter stays ordered (stale reads err HIGH, the
 * tripwire convention) instead of wrapping into a plausible small lie. */
static inline uint64_t ds4_mem_sat_add(uint64_t a, uint64_t b,
                                       uint64_t *faults) {
    if (b > UINT64_MAX - a) {
        if (faults) (*faults)++;
        return UINT64_MAX;
    }
    return a + b;
}

static inline void ds4_mem_cell_note_alloc(ds4_mem_cell *c,
                                           uint64_t requested,
                                           uint64_t committed,
                                           uint64_t *faults) {
    if (!c) return;
    c->requested = ds4_mem_sat_add(c->requested, requested, faults);
    c->committed = ds4_mem_sat_add(c->committed, committed, faults);
    c->alloc_calls++;
}

/* Free clamps to the outstanding balance: releasing more than is live is
 * an impossible state (double-free or misattributed class) -- record the
 * fault and clamp so `live` bottoms at zero instead of wrapping. */
static inline void ds4_mem_cell_note_free(ds4_mem_cell *c,
                                          uint64_t requested,
                                          uint64_t committed,
                                          uint64_t *faults) {
    if (!c) return;
    uint64_t live_req = c->requested - c->freed_requested;
    uint64_t live_com = c->committed - c->freed_committed;
    if (requested > live_req) { if (faults) (*faults)++; requested = live_req; }
    if (committed > live_com) { if (faults) (*faults)++; committed = live_com; }
    c->freed_requested = ds4_mem_sat_add(c->freed_requested, requested, faults);
    c->freed_committed = ds4_mem_sat_add(c->freed_committed, committed, faults);
    c->free_calls++;
}

static inline uint64_t ds4_mem_cell_live(const ds4_mem_cell *c) {
    return c ? c->committed - c->freed_committed : 0;
}

/* Live slack: committed-but-unasked bytes (chunk tails, granularity
 * rounding, import over-allocation, trim pages that unmapped but failed
 * to release).  Floors at 0: requested can briefly exceed committed only
 * in a fault state, never a real ledger. */
static inline uint64_t ds4_mem_cell_slack(const ds4_mem_cell *c) {
    if (!c) return 0;
    const uint64_t live_com = c->committed - c->freed_committed;
    const uint64_t live_req = c->requested - c->freed_requested;
    return live_com > live_req ? live_com - live_req : 0;
}

/* =========================================================================
 * memgov D1a-1: model source descriptors (pure table logic).
 *
 * A model SOURCE is one host mmap the engine serves weights from (base
 * checkpoint, MTP head, DSpark drafter, future support models).  Every
 * range registry in the CUDA backend already keys on the map base pointer;
 * this table names that identity: an opaque handle (the table index) with
 * role, extent, fd identity, and provenance.  Roles are semantic and
 * extensible (scoping sec 5) -- never a base/mtp/dspark hard-code.
 *
 * The table is pure (bind/find below) so the unit suite drives the exact
 * production resolution logic without a GPU, the ds4_gov_evaluate
 * precedent.  The CUDA backend instantiates one table and mirrors every
 * WEIGHT_*-class census note into a per-source side-ledger keyed by the
 * handle; the 2-D census cells stay the porcelain truth, and in a
 * fault-free ledger the source rows of each weight class sum to its cell
 * EXACTLY (dual writes under one epoch bracket; boot reconciles and any
 * divergence counts a census fault). */
#define DS4_MSRC_MAX 8                   /* bound sources + 1 spare row:
                                            index DS4_MSRC_MAX is the
                                            UNATTRIBUTED bucket (a weight
                                            note outside every bound map,
                                            e.g. kernel unit tests that
                                            map-swap without an engine) */

typedef enum {
    DS4_MSRC_ROLE_PRIMARY = 0,           /* the checkpoint being served    */
    DS4_MSRC_ROLE_AUXILIARY,             /* support model bound to the
                                            primary (MTP head)             */
    DS4_MSRC_ROLE_DRAFTER,               /* speculative drafter (DSpark)   */
    DS4_MSRC_ROLE__COUNT                 /* D5-2: metrics cardinality      */
} ds4_model_source_role;

/* memgov D3-2: ONE typed engine residency policy per source (plan sec 12
 * D3).  The value is RESOLVED ONCE at source bind from the first-class
 * knob (DS4_WEIGHT_RESIDENCY[_BASE|_MTP|_DRAFTER]) or the legacy alias
 * shape, and every consumer reads the handle -- never the environment. */
enum {
    DS4_RESIDENCY_EAGER_DEVICE = 0,      /* eager pass promotes the funded
                                            plan; lazy tier backfills      */
    DS4_RESIDENCY_LAZY_COPY_DEVICE,      /* same terminal residency, but
                                            promotion defers to first
                                            touch (the NO_HBM alias)       */
    DS4_RESIDENCY_HOST_MAPPED,           /* terminal host residency: mapped
                                            units NEVER device-promote     */
    DS4_RESIDENCY__COUNT                 /* D5-2: metrics cardinality      */
};

typedef struct {
    const void *map_base;                /* host mmap base = source identity */
    uint64_t    map_len;
    int         role;                    /* ds4_model_source_role          */
    int         fd;                      /* file identity at open, -1 none */
    uint8_t     residency;               /* DS4_RESIDENCY_* (resolved once
                                            at bind; D3-2)                 */
    char        name[24];                /* semantic id ("base"/"mtp"/
                                            "drafter") -- deliberately the
                                            weight-server manifest model_id
                                            vocabulary                     */
    char        path[192];               /* provenance (truncated copy)    */
} ds4_model_source;

typedef struct {
    ds4_model_source v[DS4_MSRC_MAX];
    int count;
} ds4_model_source_table;

/* Manual copy keeps the leaf header dependency-free (no string.h). */
static inline void ds4_msrc_copy_str(char *dst, uint64_t cap, const char *src) {
    uint64_t i = 0;
    if (cap == 0) return;
    if (src) for (; src[i] && i + 1 < cap; i++) dst[i] = src[i];
    dst[i] = '\0';
}

/* Bind (or re-bind) a source.  Returns the handle, or -1 with a fault on
 * a bad descriptor / table overflow.  Re-binding the same map_base
 * refreshes the descriptor in place and keeps the handle stable -- ledger
 * rows outlive a rebind by design (map identity is the row key). */
static inline int ds4_model_source_bind(ds4_model_source_table *t,
                                        const void *map_base, uint64_t map_len,
                                        int role, int fd, int residency,
                                        const char *name, const char *path,
                                        uint64_t *faults) {
    int i;
    if (!t || !map_base || map_len == 0) {
        if (faults) (*faults)++;
        return -1;
    }
    for (i = 0; i < t->count; i++)
        if (t->v[i].map_base == map_base) break;
    if (i == t->count) {
        if (t->count >= DS4_MSRC_MAX) {
            if (faults) (*faults)++;
            return -1;
        }
        t->count++;
    }
    t->v[i].map_base = map_base;
    t->v[i].map_len = map_len;
    t->v[i].role = role;
    t->v[i].fd = fd;
    t->v[i].residency = (uint8_t)residency;
    ds4_msrc_copy_str(t->v[i].name, sizeof(t->v[i].name), name);
    ds4_msrc_copy_str(t->v[i].path, sizeof(t->v[i].path), path);
    return i;
}

/* Resolve a pointer to its source: map-base equality fast path, then
 * containment in [base, base+len).  Distinct mmaps never overlap, so at
 * most one source can contain p.  Returns the handle or -1. */
static inline int ds4_model_source_find(const ds4_model_source_table *t,
                                        const void *p) {
    int i;
    if (!t || !p) return -1;
    for (i = 0; i < t->count; i++)
        if (t->v[i].map_base == p) return i;
    for (i = 0; i < t->count; i++) {
        const uint64_t b = (uint64_t)(uintptr_t)t->v[i].map_base;
        const uint64_t q = (uint64_t)(uintptr_t)p;
        if (q >= b && q - b < t->v[i].map_len) return i;
    }
    return -1;
}

static inline const char *ds4_model_source_role_name(int role) {
    switch (role) {
    case DS4_MSRC_ROLE_PRIMARY:   return "primary";
    case DS4_MSRC_ROLE_AUXILIARY: return "auxiliary";
    case DS4_MSRC_ROLE_DRAFTER:   return "drafter";
    default:                      return "unknown";
    }
}

/* memgov D3-2: the residency resolver.  PURE -- the caller reads the
 * environment once at engine open and passes what it found; the truth
 * table below is CPU-unit-tested without env games.
 *
 * Precedence: an explicit DS4_WEIGHT_RESIDENCY[_<SRC>] value OWNS the
 * decision and REFUSES to share it -- any legacy residency lever beside
 * it is a strict conflict (one vocabulary per boot, the plan's "no
 * interacting negative flags").  With no explicit value the legacy
 * shapes keep their exact historical meaning as aliases:
 *   (none)                  -> EAGER_DEVICE
 *   NO_HBM_CACHE            -> LAZY_COPY_DEVICE (defers the SAME bytes)
 *   NO_HBM_CACHE + NO_FD    -> HOST_MAPPED
 *   DIRECT_MODEL (any mix)  -> HOST_MAPPED (raw-direct mechanism; the
 *                              eager short-circuit + resolve tier keep
 *                              serving it -- levers beside it stay the
 *                              legal-but-inert legacy shape)
 *   NO_FD alone             -> EAGER_DEVICE (mechanism-only lever: the
 *                              lazy tier stays fenced at the resolve
 *                              site; deprecated, not a conflict)       */
typedef struct {
    const char *explicit_val;            /* knob value or 0 (unset)       */
    uint8_t legacy_no_hbm;               /* DS4_CUDA_NO_HBM_CACHE present */
    uint8_t legacy_no_fd;                /* DS4_CUDA_NO_FD_CACHE present  */
    uint8_t legacy_direct;               /* DS4_CUDA_DIRECT_MODEL present */
} ds4_residency_inputs;

static inline int ds4_msrc_str_eq(const char *a, const char *b) {
    uint64_t i = 0;
    if (!a || !b) return 0;
    for (; a[i] && b[i]; i++) if (a[i] != b[i]) return 0;
    return a[i] == b[i];
}

/* Returns DS4_RESIDENCY_* or -1 on conflict with *why set. */
static inline int ds4_residency_resolve(const ds4_residency_inputs *in,
                                        const char **why) {
    static const char *w_unknown =
        "unknown DS4_WEIGHT_RESIDENCY value (want eager|lazy|mapped)";
    static const char *w_lever =
        "DS4_WEIGHT_RESIDENCY set beside a legacy residency lever "
        "(DS4_CUDA_NO_HBM_CACHE / DS4_CUDA_NO_FD_CACHE / "
        "DS4_CUDA_DIRECT_MODEL): one vocabulary per boot -- drop the lever";
    if (why) *why = 0;
    if (!in) { if (why) *why = w_unknown; return -1; }
    if (in->explicit_val && in->explicit_val[0]) {
        if (in->legacy_no_hbm || in->legacy_no_fd || in->legacy_direct) {
            if (why) *why = w_lever;
            return -1;
        }
        if (ds4_msrc_str_eq(in->explicit_val, "eager"))
            return DS4_RESIDENCY_EAGER_DEVICE;
        if (ds4_msrc_str_eq(in->explicit_val, "lazy"))
            return DS4_RESIDENCY_LAZY_COPY_DEVICE;
        if (ds4_msrc_str_eq(in->explicit_val, "mapped"))
            return DS4_RESIDENCY_HOST_MAPPED;
        if (why) *why = w_unknown;
        return -1;
    }
    if (in->legacy_direct) return DS4_RESIDENCY_HOST_MAPPED;
    if (in->legacy_no_hbm)
        return in->legacy_no_fd ? DS4_RESIDENCY_HOST_MAPPED
                                : DS4_RESIDENCY_LAZY_COPY_DEVICE;
    return DS4_RESIDENCY_EAGER_DEVICE;
}

static inline const char *ds4_residency_name(int r) {
    switch (r) {
    case DS4_RESIDENCY_EAGER_DEVICE:     return "eager";
    case DS4_RESIDENCY_LAZY_COPY_DEVICE: return "lazy";
    case DS4_RESIDENCY_HOST_MAPPED:      return "mapped";
    default:                             return "unknown";
    }
}

/* =========================================================================
 * memgov D5-2: residency-unit materialize outcomes -- THE public
 * vocabulary behind ds4_residency_units{policy,state} (plan sec 14).
 * One enum shared by the engine's tick sites and every porcelain: the
 * engine-private CUDA_MAT_* names, retired in favor of these.  Fixed
 * cardinality like the census classes; WAIVED_OPTIONAL stays reserved
 * (no optional promote units exist in the 0731 catalogs).  The metrics
 * family keys these by the SOURCE's resolved residency policy
 * (DS4_RESIDENCY_*): a mapped-policy UNIT under an eager source ticks
 * {policy=eager, state=host_mapped_by_policy} -- the label carries the
 * source story, the state the unit story.  Row DS4_RESIDENCY__COUNT is
 * the UNATTRIBUTED bucket (tick outside any bound source), so totals
 * are preserved by construction. */
typedef enum {
    DS4_RESUNIT_POPULATED = 0,        /* copied + published                */
    DS4_RESUNIT_ALREADY_READY,        /* exact covering range pre-existed  */
    DS4_RESUNIT_SATISFIED_IMPORT,     /* covered by a wider import range   */
    DS4_RESUNIT_HOST_MAPPED_BY_POLICY,
    DS4_RESUNIT_LAZY_DEFERRED,        /* schedule defers to first touch    */
    DS4_RESUNIT_EXPERT_COLD_BY_POLICY,
    DS4_RESUNIT_WAIVED_OPTIONAL,      /* reserved                          */
    DS4_RESUNIT_FAILED,
    DS4_RESUNIT__COUNT
} ds4_residency_unit_state;

#define DS4_RESUNIT_POLICY_ROWS (DS4_RESIDENCY__COUNT + 1)

/* Failure attribution stages for ds4_residency_failures_total
 * {model_role,stage}: the eager boot pass vs the lazy first-touch tier. */
enum {
    DS4_RESSTAGE_BOOT = 0,
    DS4_RESSTAGE_LAZY,
    DS4_RESSTAGE__COUNT
};

/* =========================================================================
 * memgov D0a-2: typed memory observation.
 *
 * The provider (ds4_gpu_mem_observe) reports a typed status + WHICH source
 * produced the winning free-bytes answer, replacing sentinel-only returns.
 * The legacy shim below maps the typed form to the EXACT ds4_gpu_mem_info
 * contract (rc!=0 with outputs untouched on anything but OK) so no
 * decision can move in D0a; enforcement changes are D0b/D2's.  On
 * integrated CUDA the provider takes max(cudaMemGetInfo free,
 * /proc/meminfo MemAvailable) -- the source field makes that choice,
 * previously silent, observable. */
typedef enum {
    DS4_MEMOBS_OK = 0,
    DS4_MEMOBS_UNSUPPORTED,          /* backend keeps no answer (Metal)   */
    DS4_MEMOBS_QUERY_ERROR           /* backend query failed this call    */
} ds4_mem_obs_status;

typedef enum {
    DS4_MEMOBS_SRC_NONE = 0,
    DS4_MEMOBS_SRC_CUDA_FREE,        /* cudaMemGetInfo free won           */
    DS4_MEMOBS_SRC_MEMINFO_AVAILABLE /* MemAvailable won (integrated)     */
} ds4_mem_obs_source;

typedef struct {
    int status;                      /* ds4_mem_obs_status                */
    int source;                      /* ds4_mem_obs_source of free_bytes  */
    uint64_t free_bytes;
    uint64_t total_bytes;
    /* memgaps MG-2 (2026-08-16): the raw pair behind the max-of-two pick
     * (0 = that source had no answer this call).  Decisions keep
     * consuming free_bytes; these exist so porcelains can render BOTH
     * estimates -- the aged-box divergence between a low meminfo answer
     * and admissions that still succeed is the field signature of the
     * kernel-estimate drift, and the D6 correction's data source. */
    uint64_t cuda_free_bytes;
    uint64_t meminfo_avail_bytes;
} ds4_mem_observation;

/* Typed -> legacy ds4_gpu_mem_info semantics: rc 0 + outputs only on OK;
 * every non-OK state is rc 1 with the outputs left untouched (callers
 * pre-zero or fail open, exactly as before). */
static inline int ds4_mem_obs_to_legacy(const ds4_mem_observation *o,
                                        uint64_t *free_bytes,
                                        uint64_t *total_bytes) {
    if (!o || o->status != DS4_MEMOBS_OK) return 1;
    if (free_bytes) *free_bytes = o->free_bytes;
    if (total_bytes) *total_bytes = o->total_bytes;
    return 0;
}

/* =========================================================================
 * v0.6.2 Inc 0: reconciliation arithmetic (pure).
 *
 * The zero-headroom law made continuous: the box's raw free-memory answer
 * (MemAvailable on integrated CUDA) moved since boot settle by SOME
 * amount; the census explains part of it (live committed growth), the
 * named one-time charges (census-invisible driver/runtime warmup) explain
 * more; whatever is left is the RESIDUAL -- a phantom, a leak, another
 * process, or kernel-estimate drift, but never silent.  Signed
 * throughout: the kernel can also RETURN memory (reclaim, a neighbor
 * exiting), and a negative residual is as reportable as a positive one.
 * Inputs clamp to INT64_MAX and every signed step saturates at
 * +/-INT64_MAX (a saturated census counter errs HIGH, the tripwire
 * convention; INT64_MIN is never produced, so negation stays defined);
 * real boxes sit orders of magnitude below the clamps, so a saturated
 * value is itself an impossible-state tell, never a rounding event. */
typedef struct {
    int64_t observed_delta;   /* raw-source drop since boot (+ = consumed) */
    int64_t census_growth;    /* census live growth since boot (signed)    */
    int64_t explained;        /* census_growth + one-time charges          */
    int64_t residual;         /* observed_delta - explained                */
    int flagged;              /* |residual| > tol                          */
} ds4_mem_reconcile;

static inline int64_t ds4_mem_i64_clamp(uint64_t v) {
    return v > (uint64_t)INT64_MAX ? INT64_MAX : (int64_t)v;
}

/* Saturating signed add over [-INT64_MAX, INT64_MAX] (both operands
 * always in that range here: they are differences of clamped
 * non-negatives or prior saturated results). */
static inline int64_t ds4_mem_i64_sat_add(int64_t a, int64_t b) {
    if (b > 0 && a > INT64_MAX - b) return INT64_MAX;
    if (b < 0 && a < -INT64_MAX - b) return -INT64_MAX;
    return a + b;
}

static inline ds4_mem_reconcile ds4_mem_reconcile_compute(
        uint64_t boot_raw, uint64_t now_raw,
        uint64_t boot_census_live, uint64_t now_census_live,
        uint64_t onetime_bytes, uint64_t tol_bytes) {
    ds4_mem_reconcile r;
    r.observed_delta = ds4_mem_i64_clamp(boot_raw) - ds4_mem_i64_clamp(now_raw);
    r.census_growth  = ds4_mem_i64_clamp(now_census_live) -
                       ds4_mem_i64_clamp(boot_census_live);
    r.explained = ds4_mem_i64_sat_add(r.census_growth,
                                      ds4_mem_i64_clamp(onetime_bytes));
    r.residual  = ds4_mem_i64_sat_add(r.observed_delta, -r.explained);
    const int64_t mag = r.residual < 0 ? -r.residual : r.residual;
    r.flagged = mag > ds4_mem_i64_clamp(tol_bytes);
    return r;
}

/* =========================================================================
 * memgov D0a-4: trim failure-injection (TEST ONLY).
 *
 * ds4_gpu_tensor_trim survives two driver failures with distinct ledger
 * shapes (D-1b): a cuMemUnmap failure keeps the page owned, mapped, and
 * charged; a cuMemRelease failure after a successful unmap is a physical
 * leak the cell reports as slack.  The CUDA backend can force either at
 * the real driver boundary -- armed by the DS4_CUDA_TRIM_INJECT env spec
 * or programmatically (ds4_gpu_trim_inject_set, declared in ds4.h).  The
 * spec grammar lives here as a pure function so the unit suite pins the
 * exact production parse without a GPU. */
typedef enum {
    DS4_TRIM_INJECT_OFF = 0,
    DS4_TRIM_INJECT_UNMAP,           /* force cuMemUnmap failure           */
    DS4_TRIM_INJECT_RELEASE          /* force cuMemRelease failure         */
} ds4_trim_inject_site;

static inline const char *ds4_mem_prefix_match(const char *s, const char *pre) {
    while (*pre) { if (*s != *pre) return 0; s++; pre++; }
    return s;
}

/* Parse "unmap:N" | "release:N" (decimal N in [1, UINT32_MAX]).  Returns 1
 * and writes site/count on a valid spec; 0 with the outputs untouched
 * otherwise.  N=0 is rejected: "off" is spelled by unsetting the env, so
 * an armed-but-inert state cannot exist.  Manual scan keeps the leaf
 * header dependency-free. */
static inline int ds4_trim_inject_parse(const char *spec,
                                        int *site_out, uint32_t *count_out) {
    if (!spec) return 0;
    int site = DS4_TRIM_INJECT_OFF;
    const char *digits = ds4_mem_prefix_match(spec, "unmap:");
    if (digits) {
        site = DS4_TRIM_INJECT_UNMAP;
    } else if ((digits = ds4_mem_prefix_match(spec, "release:")) != 0) {
        site = DS4_TRIM_INJECT_RELEASE;
    } else {
        return 0;
    }
    if (*digits < '0' || *digits > '9') return 0;
    uint64_t n = 0;
    for (; *digits; digits++) {
        if (*digits < '0' || *digits > '9') return 0;
        n = n * 10u + (uint64_t)(*digits - '0');
        if (n > UINT32_MAX) return 0;
    }
    if (n == 0) return 0;
    *site_out = site;
    *count_out = (uint32_t)n;
    return 1;
}

#endif /* DS4_MEM_CENSUS_H */
