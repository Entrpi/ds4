#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stdint.h>

/* Track B: DeepSeek-V4-Flash-Vision-Exp encoder sidecar weights.  Byte offsets
 * into the sidecar GGUF map (upstream antirez/ds4 layout, kept identical so the
 * encoder kernels port 1:1); bound + validated by deepseek4_vision_weights_bind. */
#ifndef DS4_DEEPSEEK4_VISION_TYPES_DEFINED
#define DS4_DEEPSEEK4_VISION_TYPES_DEFINED
#define DS4_DEEPSEEK4_VISION_LAYERS 32u
#define DS4_DEEPSEEK4_LANGUAGE_LAYERS 43u
#define DS4_DEEPSEEK4_MTP_LAYERS 3u
#define DS4_DEEPSEEK4_VISION_TENSORS 316u

typedef struct {
    uint64_t norm1;
    uint64_t qkv_weight;
    uint64_t qkv_bias;
    uint64_t attn_proj_weight;
    uint64_t attn_proj_bias;
    uint64_t norm2;
    uint64_t mlp_w1;
    uint64_t mlp_w2;
} ds4_deepseek4_vision_layer_weights;

typedef struct {
    uint64_t patch_weight;
    uint64_t patch_bias;
    uint64_t post_norm;
    uint64_t aligner_w1;
    uint64_t aligner_w1_bias;
    uint64_t aligner_w2;
    uint64_t aligner_w2_bias;
    uint64_t image_start;
    uint64_t image_pad;
    uint64_t image_newline;
    uint64_t image_end;
    uint64_t visual_router_bias[DS4_DEEPSEEK4_LANGUAGE_LAYERS];
    uint64_t mtp_visual_router_bias[DS4_DEEPSEEK4_MTP_LAYERS];
    uint64_t hash_router_bias[3];
    ds4_deepseek4_vision_layer_weights layer[DS4_DEEPSEEK4_VISION_LAYERS];
} ds4_deepseek4_vision_weights;
#endif

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
typedef struct ds4_gpu_tensor ds4_gpu_tensor;

typedef struct ds4_gpu_top2_result {
    uint32_t id0;
    uint32_t id1;
    float    value0;
    float    value1;
} ds4_gpu_top2_result;

typedef struct ds4_gpu_candidate_cert_result {
    uint32_t candidate_id;
    uint32_t certified;
    uint32_t bound_id;
    float    candidate_logit;
    float    max_bound;
} ds4_gpu_candidate_cert_result;

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
/* R5 Inc1b: demand-mapped reserved tensors.  reserve() takes the full VIRTUAL
 * range (views and pointer arithmetic behave exactly like an eager alloc) but
 * maps no physical pages; ensure() maps the pages covering [offset, offset+
 * bytes) before a write -- pass ANY tensor (reserved base, view into one, or
 * an eager alloc: eager succeeds as a no-op, so call sites need no mode
 * checks).  resident() reports the mapped bytes overlapping the range (eager:
 * the range itself).  vmm_demand_page() returns the mapping granularity, 0
 * when the backend cannot demand-map (Metal; CUDA without VMM) -- reserve()
 * then returns NULL and callers fall back to eager allocation. */
ds4_gpu_tensor *ds4_gpu_tensor_reserve(uint64_t bytes);
int ds4_gpu_tensor_ensure(const ds4_gpu_tensor *tensor, uint64_t offset, uint64_t bytes);
uint64_t ds4_gpu_tensor_resident(const ds4_gpu_tensor *tensor, uint64_t offset, uint64_t bytes);
uint64_t ds4_gpu_tensor_trim(const ds4_gpu_tensor *tensor, uint64_t offset, uint64_t bytes);
/* memgov D4-2: the exact preimage of ds4_gpu_tensor_trim -- mapped bytes of
 * the pages lying ENTIRELY inside [offset, offset+bytes), the same interior-
 * page math trim releases (edge pages shared with neighboring spans are
 * excluded on both sides: never quoted, never released).  Distinct from
 * ds4_gpu_tensor_resident, which counts every OVERLAPPING page.  Pure host
 * walk, no driver calls; 0 for eager tensors and trimless backends. */
uint64_t ds4_gpu_tensor_trim_estimate(const ds4_gpu_tensor *tensor, uint64_t offset, uint64_t bytes);
uint64_t ds4_gpu_vmm_demand_page(void);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
/* Step 6: opaque-pointer accessor for cache-key construction in ds4.c.
 * Returns the backend buffer base address (CUDA: cudaMalloc result;
 * Metal: id<MTLBuffer> contents).  Used only for pointer-identity
 * comparisons in the layer-graph cache key; never dereferenced from C. */
const void *ds4_gpu_tensor_ptr(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count);

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_end_commands(void);
/* 1 iff the backend batches GPU work into explicit command scopes, i.e.
 * begin/end_commands bracket real submission and begin returns 1 only when
 * THIS call opened the scope (Metal).  0 when submission is stream-ordered
 * and begin/end are compatibility stubs whose returns don't mean ownership
 * (CUDA: begin always 1, end is a heavyweight sync).  OPTIONAL bracketing
 * -- "open a scope iff none is open, close it iff I opened it" -- must be
 * gated on this; unconditional top-of-scope brackets need not check. */
int ds4_gpu_commands_scoped(void);
int ds4_gpu_synchronize(void);
int ds4_gpu_stream_synchronize(void);   /* task #22 diag: legacy-stream-only sync */

/* STAGE_PROF_LITE (env DS4_CUDA_STAGE_PROF_LITE): lazy-event GPU-span
 * measurement of whole forward stages, bracketed from ds4.c at the
 * CONT_PROFILE host-bracket points.  Same discipline as MOE_PROF_LITE
 * (per-call sync and CUDA_LAUNCH_BLOCKING both inflate GB10 spans 5-20x;
 * host wall brackets smear under async launch billing): a persistent event
 * ring brackets each stage dispatch on the current stream, pairs are
 * harvested non-blocking when the ring wraps.  begin() returns a slot or
 * UINT32_MAX (disabled / capture active / ring init failed); end() is a
 * no-op for UINT32_MAX.  Metal: stubs, always UINT32_MAX. */
enum ds4_stage_prof_id {
    DS4_SPROF_FWD = 0,   /* whole 43-layer batched forward (per decode step) */
    DS4_SPROF_EMBED,     /* token upload + HC embedding */
    DS4_SPROF_ATTN,      /* per-layer attention half (encloses the four below) */
    DS4_SPROF_ATTNCORE,  /* heads attention read (raw/mixed/indexed batch) */
    DS4_SPROF_IDXSCORE,  /* indexer scoring + top-k */
    DS4_SPROF_CUPD,      /* compressor update (inside EMIT) */
    DS4_SPROF_EMIT,      /* multiseq per-row compressor/indexer emit loop */
    DS4_SPROF_RBCP,      /* rollback-capture state snapshots (inside EMIT) */
    DS4_SPROF_FFN,       /* per-layer FFN half (encloses MOE) */
    DS4_SPROF_MOE,       /* routed-MoE section (host-side; cross-check MOE_PROF_LITE) */
    DS4_SPROF_HEAD,      /* output head + logits readback */
    DS4_SPROF_COUNT
};
uint32_t ds4_gpu_stage_prof_begin(uint32_t stage);
void     ds4_gpu_stage_prof_end(uint32_t slot);

/* =========================================================================
 * Decode-time position scalars (full-layer CUDA-graph capture, Step A).
 * =========================================================================
 *
 * Two parallel substrates carry position-derived scalars to position-
 * dependent kernels:
 *
 *   1. TOKEN-STABLE struct (ds4_decode_scalars, 40 B device-side).  Carries
 *      scalars whose value is constant across all 43 layers within a token:
 *      pos0, raw_row, raw_start, n_raw, emit_phase, flags, token.  Single pinned
 *      host buffer + single device address; one H2D memcpy per token,
 *      currently EAGER on ds4_current_stream() (outside any captured-graph
 *      scope).  Under Step 6's wider per-token graph the memcpy can become
 *      a captured node; the captured-memcpy semantic is probe-validated
 *      address-bound.  The struct's three per-layer fields (n_comp,
 *      comp_row, index_row) are LEGACY today and will be retired in
 *      Step 4c; see plan doc local/docs/ds4_full_layer_graph_capture_plan
 *      .html sec 4.2 (P1a) for why they cannot safely carry per-layer
 *      values through a single pinned buffer.
 *
 *   2. PER-LAYER ARRAY (ds4_layer_scalars[43], 688 B device-side, double-
 *      buffered pinned host).  Carries scalars that DIFFER across layers
 *      within a token: n_comp, comp_row, index_row, per-layer flag bits.
 *      Added in Step 4b after R6 proved a shared single-buffer substrate
 *      races the GPU.  See plan doc sec 15 for the full design.
 *
 * Backends other than CUDA may implement these as no-ops if they don't
 * use the graph-capture path.  On Metal the same scalars are passed via
 * a buffer-pointer kernel argument (see ds4_metal.m).
 *
 * Lifecycle:
 *   ds4_gpu_decode_scalars_init()             once per GPU session
 *   ds4_gpu_decode_scalars_set(pos, ...)      once per decode token
 *   ds4_gpu_decode_scalars_flush()            once per decode token
 *   ds4_gpu_decode_scalars_cleanup()          on teardown
 *
 * The opaque device pointer returned by *_device_ptr() is the value passed
 * to per-kernel shims that depend on the scalars.  It is session-stable
 * and may be cached by the caller after the first init. */
int   ds4_gpu_decode_scalars_init(void);
void  ds4_gpu_decode_scalars_cleanup(void);
const void *ds4_gpu_decode_scalars_device_ptr(void);
void  ds4_gpu_decode_scalars_set(
        uint32_t pos0,
        uint32_t raw_cap,
        uint32_t raw_window,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t flags,
        uint32_t token);
/* Push the most recent ds4_gpu_decode_scalars_set() values to the device-side
 * mirror.  Backends that don't use the graph-capture path implement this as
 * a no-op.  On CUDA this issues a single H2D memcpy on ds4_current_stream(),
 * eager (outside captured-graph scope) today; Step 6 may bring it inside a
 * wider per-token graph at which point the memcpy is captured into the
 * outer graph node list.  Returns 1 on success / no-op, 0 on infrastructure
 * failure. */
int   ds4_gpu_decode_scalars_flush(void);

/* C1 (cont capture): explicit-window variant of ds4_gpu_decode_scalars_set.
 * The batched multiseq step derives its raw read-window on the host via
 * metal_graph_raw_window_for_batch (ring-linearisation semantics that the
 * plain setter's internal formula only matches when raw_window <= raw_cap),
 * so the cont publish path passes the EXACT values the eager kernel args
 * would carry instead of re-deriving them.  emit_phase/comp_row/index_row
 * are left untouched (comp/index rows ride the per-layer substrate). */
void  ds4_gpu_decode_scalars_set_ext(
        uint32_t pos0,
        uint32_t raw_row,
        uint32_t raw_start,
        uint32_t n_raw,
        uint32_t n_comp,
        uint32_t emit_phase,
        uint32_t flags,
        uint32_t token);

/* UNSAFE pending removal in Step 4c (see plan doc sec 4.2 P1a).
 *
 * Per-emit setter for the row scalars used by the R1 row-variant shims.
 * Called from ds4.c at each per-layer emit step (compressor + indexer),
 * followed by ds4_gpu_decode_scalars_flush() so the new values reach the
 * device before the next kernel reads them.  Other scalar fields preserved.
 *
 * STRUCTURAL RACE: the host source (g_decode_host->comp_row/index_row) is
 * overwritten by the next layer's set_emit_rows() while the previous
 * layer's queued cudaMemcpyAsync may still be pending.  Bit-identical
 * parity holds today by accident only (all 43 compressed layers see
 * identical g->layer_n_comp[il] at any ratio-4 emit pos).  Do not call
 * from new code; Step 4c migrates the R1 row-view kernels to read row
 * scalars from the per-layer ds4_layer_scalars substrate and removes
 * these setters. */
void  ds4_gpu_decode_scalars_set_emit_rows(uint32_t comp_row,
                                             uint32_t index_row);

/* UNSAFE pending removal in Step 4c (see plan doc sec 4.2 P1a + R6).
 *
 * Per-layer setter for n_comp.  Same single-buffer race as set_emit_rows;
 * R6 was originally discovered via this setter.  No current callers (the
 * c587d96 fixup-2 attention path stopped using it).  Retained as a header
 * declaration only so existing code doesn't accidentally re-introduce the
 * race during the Step 4b/4c transition. */
void  ds4_gpu_decode_scalars_set_n_comp(uint32_t n_comp);

/* =========================================================================
 * Per-layer scalars substrate (Step 4b: R6 fix).
 * =========================================================================
 *
 * Carries scalars whose value DIFFERS across the 43 layers within a token:
 * n_comp, comp_row, index_row, plus per-layer flag bits.  See plan doc
 * sec 15 for the full design rationale; sec 15.3 for the ordering proof;
 * sec 15.8 for the cache-key invariant.
 *
 * Substrate is the array-of-43 + double-buffered host design empirically
 * validated by tests/cuda_graph_layer_array_probe.cu (PASS on PRO 6000
 * Blackwell sm_120).
 *
 * Lifecycle (will be wired in Step 4c):
 *   ds4_gpu_decode_layer_scalars_init()         once per GPU session
 *   ds4_gpu_decode_layer_scalars_host()         once per decode token: get
 *                                               the active host buffer
 *   <CPU writes all 43 entries>
 *   ds4_gpu_decode_layer_scalars_flush()        once per decode token: queue
 *                                               the H2D memcpy + rotate idx
 *   ds4_gpu_decode_layer_scalars_device_ptr()   pass to per-layer kernel shims
 *                                               (callers add `il * sizeof(...)`)
 *   ds4_gpu_decode_layer_scalars_cleanup()      at GPU teardown
 *
 * Layer-count discipline: the substrate is sized for V4 Flash's 43 layers
 * (DS4_LAYER_SCALARS_COUNT in ds4_cuda.cu).  The same constant lives at
 * DS4_N_LAYER in ds4.c; both must move together if the model topology
 * changes (same convention as DS4_N_HEAD_DIM / DS4_N_ROT).
 *
 * Backends other than CUDA implement these as no-ops; Metal stubs return
 * 1 from init/flush and NULL from device_ptr/host so shim signatures stay
 * uniform across backends.
 *
 * Symbol naming mirrors ds4_gpu_decode_scalars_* so the relationship is
 * obvious at the call site:
 *   decode_scalars       = token-stable (single struct, single buffer)
 *   decode_layer_scalars = per-layer (43-entry array, double-buffered host) */
int   ds4_gpu_decode_layer_scalars_init(void);
void  ds4_gpu_decode_layer_scalars_cleanup(void);
const void *ds4_gpu_decode_layer_scalars_device_ptr(void);
void *ds4_gpu_decode_layer_scalars_host(void);
int   ds4_gpu_decode_layer_scalars_flush(void);

/* Write all four per-layer scalar fields of the currently-active host
 * buffer for layer `il`.  Caller invokes this in a loop over 0..42 at
 * top of token (after ds4_gpu_decode_scalars_set / _flush) and then
 * calls ds4_gpu_decode_layer_scalars_flush() once so the H2D memcpy
 * fires before per-layer kernels read from the device side.
 *
 * Field semantics:
 *   n_comp     -- post-this-token's-emit visible-compressed count.
 *                 Read by attention's ls_override path (Step 4c A1).
 *   n_index_comp -- indexer compressed count, post-emit (PC3).  Equals
 *                 g->layer_n_index_comp[il] + (emit_il ? 1 : 0).
 *                 First consumer is PC5's I1/I2 max-grid + bounds-check
 *                 indexer kernel pilot.
 *   comp_row   -- pre-emit row index for fp8 row-kernel.  Equals
 *                 g->layer_n_comp[il] (pre-increment).
 *   index_row  -- pre-emit row index for indexer_qat row-kernel.  Equals
 *                 g->layer_n_index_comp[il] (pre-increment).
 *
 * Backends other than CUDA stub this as a no-op (Metal kernels read
 * inline args). */
void  ds4_gpu_decode_layer_scalars_set(
        uint32_t il,
        uint32_t n_comp,
        uint32_t n_index_comp,
        uint32_t comp_row,
        uint32_t index_row);

/* C2 Inc2: per-ROW emit-row table for cont capture steps (multi-live).
 * One {comp_row, index_row} entry per (batch row 0..7, layer 0..42): the
 * GLOBAL cache rows this row's emit writes, with the bank base AND the
 * within-step emit ordering already applied by the host at publish time
 * (this retired the C1b emit_row_delta scheme).  Published once per
 * capture step (set per emitting row+layer, then one flush); captured
 * kernels bake the table-entry ADDRESS and read the live value at
 * replay.  Same double-buffered pinned discipline as the per-layer
 * scalars above.  Metal stubs: init/flush return 0/1, set is a no-op.
 * The row bound mirrors the same constant in ds4_cuda.cu (which keeps
 * local decls; see the DS4_LAYER_SCALARS_COUNT precedent). */
#define DS4_ROW_SCALARS_MAX_ROWS 8u
int   ds4_gpu_decode_row_scalars_init(void);
void  ds4_gpu_decode_row_scalars_set(
        uint32_t row,
        uint32_t il,
        uint32_t comp_row,
        uint32_t index_row);
int   ds4_gpu_decode_row_scalars_flush(void);

/* =========================================================================
 * Step 5/6: per-layer cudaGraph capture-and-replay.
 *
 * Wraps each iteration of the per-layer decode loop in
 * cudaStreamBeginCapture / EndCapture / GraphInstantiate, then replays the
 * cached executable on subsequent tokens.  CUDA-only; Metal stubs all of
 * these as no-ops (begin returns -1, end is a no-op, enabled returns 0).
 *
 * On by default since Step 8 (determinism + perf gates passed on sm_120
 * and sm_121).  Set DS4_CUDA_LAYER_GRAPHS=0 to fall back to eager decode.
 *
 * Per-token scalars do NOT enter the key -- they ride on the token-stable
 * and per-layer scalar substrates (see ds4_gpu_decode_scalars_*) which
 * have stable device addresses baked into the captured kernel-node arg
 * lists.  See plan doc sec 4.2 / 15 / 16.
 * ========================================================================= */

struct ds4_layer_graph_key {
    uint32_t il;     /* layer index, 0..DS4_N_LAYER-1 */
    uint32_t n_tok;  /* 1 for normal decode; 2 for MTP decode2-exact */
    uint32_t flags;  /* bit 0: emit_this_step (ratio-4 emit token)
                      * bit 1: indexed_active (n_comp > decode_top_k)
                      * bit 2: compressed_layer (ratio != 0)
                      * bit 3: ratio4 vs ratio1
                      * bits 4..31: reserved */
    uint32_t _pad;   /* align to 8 for the pointer block below */
    /* Tensor base pointers referenced by the captured layer body.
     * Reallocation (KV-cache regrow) invalidates the cached graph
     * (eviction + recapture). */
    void    *cur_hc;
    void    *after_ffn_hc;
    void    *raw_cache;
    void    *comp_cache;
    void    *index_comp_cache;
    void    *q;
    void    *kv;
    void    *heads;
    void    *indexer_q;
    void    *indexer_weights;
    void    *indexer_scores;
    void    *comp_selected;
    void    *comp_kv_cur;
    void    *comp_sc_cur;
    void    *attn_state_kv;
    void    *attn_state_score;
    void    *index_state_kv;
    void    *index_state_score;
    /* Opp C Phase 1A: packed FP8 compressed-KV mirror buffers.  NULL
     * unless DS4_CUDA_FP8_KV is enabled; a stable NULL keeps the key
     * (hence the captured-graph cache) byte-identical when the feature
     * is off. */
    void    *comp_cache_fp8;
    void    *comp_scale;
};

/* Returns 1 unless DS4_CUDA_LAYER_GRAPHS is set to a disable value
 * (0/off/no/false); default ON since Step 8.  Metal stub returns 0.
 * Callers use this to gate the per-token build_key cost and the R4
 * split-flush override. */
int  ds4_cuda_layer_graphs_enabled(void);

/* Opp C Phase 1A: returns 1 unless DS4_CUDA_FP8_KV is set to a disable
 * value (0/off/no/false); default ON since v0.2.4, and refused (0) when
 * the batch comp slabs cannot be VMM demand-mapped.  Gates allocation of
 * the packed FP8 compressed-KV mirror buffers and the FP8 emit /
 * attention-read paths.  Metal stub returns 0. */
int  ds4_cuda_fp8_kv_enabled(void);

/* begin_or_replay return values:
 *   1  -- cache hit; graph replayed; caller skips the layer body encoding.
 *   0  -- capturing; caller encodes the body; close with end_or_commit.
 *  -1  -- disabled / unavailable; caller proceeds eagerly.
 * Metal backend always returns -1. */
int  ds4_cuda_layer_graph_begin_or_replay(uint32_t il,
                                            const struct ds4_layer_graph_key *key);
void ds4_cuda_layer_graph_end_or_commit(uint32_t il);

/* Step 6 diagnostic: cudaPeekAtLastError check.  No-op unless
 * DS4_CUDA_LAYER_GRAPHS_DEBUG=1 AND a layer-graph capture is in flight.
 * Used to bisect capture-mode incompatibilities by call site. */
void ds4_cuda_layer_graph_debug_peek(const char *label);

/* =========================================================================
 * C1: continuous-batch per-layer cudaGraph capture-and-replay.
 * =========================================================================
 *
 * Same warm -> capture -> replay protocol as the serial trio above, with a
 * SEPARATE direct-mapped cache and a key shaped for the multiseq batched
 * decode step (width, bank, emit topology, n_comp band, plus the tensor
 * base pointers whose reallocation must invalidate cached execs).  Opt-in
 * via DS4_CONT_CAPTURE=1 (default OFF); enabled() reports the env gate so
 * ds4.c can skip key construction entirely when off.  Metal stubs: enabled
 * returns 0, begin returns -1, end is a no-op. */
struct ds4_cont_graph_key {
    uint32_t il;          /* layer index */
    uint32_t width;       /* batch rows (1 = cont decode, 5 = D=4 verify) */
    uint32_t row_bank_pattern; /* C2: per-row bank ORDINALS (4b/row, first-
                           * occurrence order).  Inc1: always 0 -- graphs are
                           * bank-agnostic for single-bank steps (state lanes +
                           * ckpts resolve the bank from the device seq_id
                           * array at the baked row index).  Inc2 fills it for
                           * multi-live shapes (segment boundaries implicit). */
    uint32_t flags;       /* bit 0: emit_this_step (any row crosses a ratio
                           *        boundary for this layer)
                           * bit 1: indexed_active (ratio4 && n_comp > top_k)
                           * bit 2: compressed_layer (ratio != 0)
                           * bit 3: ratio4
                           * bit 4: window_sliding (raw_start != 0 regime)
                           * bit 5: rollback_capture (verify step checkpoints)
                           * bits 8..15: emit row bitmask within the batch
                           * bits 16..31: reserved */
    uint32_t n_comp_band; /* indexer grid/stride ceiling (1024/2048/4096/8192);
                           * kernels bound-check against the live substrate
                           * counts, so only band CROSSINGS recapture */
    uint32_t _pad;
    /* Tensor base pointers baked into the captured bodies. */
    void    *cur_hc;          /* batch_cur_hc (ping-pong parity) */
    void    *next_hc;         /* batch_next_hc */
    void    *raw_cache;
    void    *comp_cache;
    void    *index_comp_cache;
    void    *comp_cache_fp8;
    void    *comp_scale;
    void    *index_fp4;
    void    *index_scale;
    void    *attn_state_kv;   /* slab bases (C2: lanes resolve from the device
                               * seq_id array at replay -- bank-agnostic; only
                               * slab regrow rekeys) */
    void    *attn_state_score;
    void    *index_state_kv;
    void    *index_state_score;
    void    *comp_emit_scratch;   /* sticky grow-realloc'd; grow must rekey */
    void    *index_emit_scratch;
    void    *positions;       /* batch_positions / batch_seq_id device arrays */
    void    *seq_id;
    void    *comp_selected;
    void    *indexer_scores;
    void    *batch_heads;
};

int  ds4_cuda_cont_graphs_enabled(void);
/* C3-Inc5: 1 = the batched embed gather (ds4_gpu_embed_tokens_hc_tensor) is a
 * cheap eager launch, so ds4.c uses it at EVERY batch width instead of the
 * host f16-dequant + replicated-HC H2D build.  CUDA returns 1; Metal returns 0
 * (a separate small command buffer costs more than the host write there). */
int  ds4_cuda_embed_gather_all_widths(void);
/* begin_or_replay: 1 = replayed (skip body; caller advances host counters),
 * 0 = capturing (encode body, then end_or_commit), -1 = eager. */
int  ds4_cuda_cont_graph_begin_or_replay(uint32_t il,
                                           const struct ds4_cont_graph_key *key);
void ds4_cuda_cont_graph_end_or_commit(uint32_t il);
/* One-line engagement stats (captures/replays/eager) under
 * DS4_CONT_CAPTURE_STATS=1; printed every `every` replays. */
void ds4_cuda_cont_graph_stats_maybe_print(uint32_t every);

/* C1: public wrapper for the captured-graph invalidation used by the
 * sticky-scratch growers that live in ds4.c (comp/index emit scratch).
 * No-op on Metal. */
void ds4_gpu_invalidate_captured_graphs(const char *why);

/* MT-6 (audit V9/V11): drop the SERIAL per-layer captured-exec pool.
 * Called by ds4_session_free -- every entry bakes the freed graph's
 * tensor pointers, so the pool is stale in whole.  Returns the count
 * dropped; no-op (0) on Metal. */
int ds4_cuda_layer_graphs_drop(void);

/* Captured-decode per-kernel hash dump (DS4_CUDA_LAYER_GRAPHS_HASH_DUMP=1).
 * Permanent, env-gated, no-op-when-off diagnostic for localizing a
 * captured-graph-vs-eager output divergence by hashing each shim's output
 * buffer and printing a per-token table. See the comment block above the
 * implementation in ds4_cuda.cu for the probe-placement recipe.
 * - dump_hash_reset:    clear the auto-slot counter; call once per token.
 * - dump_hash_after:    FNV-1a a tensor into the next auto slot + label it.
 * - dump_hash_at_slot:  same, into a caller-chosen fixed slot (use inside
 *                       captured regions to avoid auto-slot collisions).
 * - dump_hash_flush:    sync, copy hashes to host, print one line per used
 *                       slot ("DS4_HASH pos=N slot=I hexhash label").
 * Metal stubs all four as no-ops. */
void ds4_cuda_dump_hash_reset(void);
void ds4_cuda_dump_hash_after(const ds4_gpu_tensor *tensor,
                              uint64_t n_elem,
                              const char *label);
void ds4_cuda_dump_hash_at_slot(const ds4_gpu_tensor *tensor,
                                uint64_t n_elem,
                                const char *label,
                                uint32_t slot);
void ds4_cuda_dump_hash_flush(uint32_t pos);

/* Cross-TU one-shot probe-slot handoff for the hash dump. ds4.c sets the
 * current layer at the top of each layer body; a probe site in ds4_cuda.cu
 * arms a slot; the mmq TU (cuda/mmq) consumes it after an internal
 * computation and hashes that buffer via the raw helper. The raw helper
 * accepts a raw void* + n_floats for buffers with no ds4_gpu_tensor wrapper.
 * Metal stubs all five as no-ops. */
void     ds4_cuda_dump_set_current_layer(int il);
int      ds4_cuda_dump_get_current_layer(void);
void     ds4_cuda_dump_probe_slot_set(uint32_t slot);
uint32_t ds4_cuda_dump_probe_slot_consume(void);
void     ds4_cuda_dump_hash_raw_at_slot(const void *buf, uint64_t n_floats,
                                        const char *label, uint32_t slot);

/* TEMPORARY DIAGNOSTIC: captured-vs-eager substrate probe.  Launches a
 * tiny single-thread kernel on ds4_current_stream() that reads
 * g_layer_dev[il] (n_index_comp, n_comp) and writes the packed value into
 * g_dump_hashes_dev[slot].  No-op when the hash-dump env gate is off.
 * Used to detect a stale-substrate race in captured-graph replays. */
void     ds4_cuda_probe_layer_substrate_at_slot(uint32_t il,
                                                const char *label,
                                                uint32_t slot);
/* Same shape, but packs PRE-emit (comp_row << 32) | index_row instead of
 * POST-emit (n_index_comp, n_comp).  Complementary diagnostic. */
void     ds4_cuda_probe_layer_rows_at_slot(uint32_t il,
                                           const char *label,
                                           uint32_t slot);

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
int ds4_gpu_import_model_ipc_manifest(const void *model_map, uint64_t model_size, const char *manifest_path, const char *model_id);
/* Self-load aligned artifacts (v0.2.2): with no weight-server manifest, build
 * the aligned-SoA repack artifacts in-process at load (shared layout library
 * cuda/mmq/ds4_repack.cu) so self-load boots ride the same fast dispatches as
 * manifest imports.  Call BEFORE ds4_gpu_set_model_map_range for the base
 * model.  Returns the number of artifacts built (0 = raw tier; boot goes on).
 * Opt-outs: DS4_CUDA_NO_DERIVED_WEIGHTS, DS4_CUDA_BUILD_ARTIFACTS=0. */
int ds4_gpu_build_derived_artifacts(const void *model_map, uint64_t model_size, const char *model_path);
/* Aligned-artifact tier for observability: source 0=none 1=imported 2=built.
 * Any out pointer may be NULL. */
void ds4_gpu_derived_artifact_stats(int *source, uint64_t *count, uint64_t *bytes, double *build_secs);
/* Print the canonical one-line boot banner for the artifact tier. */
void ds4_gpu_report_derived_artifacts(void);
int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
/* Retire the host registration for one mapping BEFORE its owner frees it.
   Required for map-swapping callers (kernel unit tests); a registration that
   outlives its allocation poisons later cudaMemcpy calls whose host buffers
   land on the recycled pages.  No-op when the base was never registered. */
void ds4_gpu_unregister_model_map(const void *base);
/* memgov D1b-1: the EAGER residency pass — materialize the map's funded
 * plan (its bound canonical-unit table) before serve.  Replaces the old
 * per-span walk (ds4_gpu_cache_model_range): the table IS the
 * enumeration, so slice boots promote exactly their plan.  Returns 0
 * only when a funded unit fails to materialize (callers abort engine
 * open, the old walk's failure contract); discrete devices and unbound
 * tables skip the pass (the lazy tier still promotes on demand).
 * populated_bytes (optional) reports bytes actually device-copied. */
int ds4_gpu_materialize_model_plan(const void *model_map, uint64_t model_size, uint64_t *populated_bytes);
/* memgov D2-2: freeze the residency plan at boot settle.  After freeze,
 * arena ranges may enter the publication funnel only through the unit
 * materializer (whose promotions carry governor claims); a bypass counts
 * a governor fault (loud once).  Metal/no-GPU: no plan, no-op. */
void ds4_gpu_model_plan_freeze(void);
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
/* R5 Inc1a: device memory snapshot for budget-computed batch-bank sizing.
   Returns 0 and writes the allocator's current free/total bytes; nonzero when
   the backend cannot answer (Metal) -- callers fall back to try-and-reduce. */
int ds4_gpu_mem_info(uint64_t *free_bytes, uint64_t *total_bytes);
/* deepmem lite-3 (tripwire): bytes of base-map promotable weight spans the
   engine is still certain to device-promote after boot (the lazy fd-cache
   tier), i.e. walk-declared minus covered.  ~0 on eager boots (the standing
   assertion that eager promotion completed); ~8 GiB under
   DS4_CUDA_NO_HBM_CACHE.  Subtracted by the bank fit, ds4_mem_usable(), and
   the serial graph fit so plans and live verdicts charge deferred
   promotions before they land.  Always 0 on Metal / discrete CUDA /
   DS4_CUDA_DIRECT_MODEL. */
uint64_t ds4_gpu_substrate_outstanding(void);
/* inc-14b follow-up: release device reserves the boot prewarm accumulated but
   nothing references anymore (graph-pool slack), keeping warmed content
   resident.  No-op on backends without the concept. */
void ds4_gpu_boot_trim(void);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_print_memory_report(const char *label);
void ds4_gpu_set_attention_output_b_n2_q8_override(int enabled);

/* Bug 2 / Option D: force matmuls to legacy native kernels for the duration
 * of an MTP verifier call.  See local/docs/ds4_mmq_mtp_correctness_plan.html
 * in the auto-round companion repo for the full mechanism.  Call sites wrap
 * each metal_graph_verify_* (and the verifier-context callers of
 * metal_graph_eval_token_raw_swa_top) with set(1)/.../set(0).  Backends
 * other than CUDA implement these as no-ops. */
void ds4_gpu_set_mtp_verifier(int on);
int  ds4_gpu_in_mtp_verifier(void);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc);

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale,
        /* PC5 micro-pilot: max-grid + bounds-check substrate params.
         *   n_comp_max -- session-stable per-layer comp_cap (upper bound
         *                  on n_comp).  Pass 0 to opt out (legacy n_comp
         *                  grid; decode2-exact + Metal stub).
         *   il         -- layer index; pass UINT32_MAX for legacy path.
         * When both are set, the CUDA backend launches the _direct
         * kernel with grid = n_comp_max and the kernel reads the runtime
         * count from ls->n_index_comp (PC3 substrate field).  Set the
         * env var DS4_CUDA_PC5_LEGACY_GRID=1 to force the legacy path
         * for A/B perf measurement. */
        uint32_t                n_comp_max,
        uint32_t                il,
        /* P2 Inc3: optional packed FP4 indexer mirror (g->layer_index_comp_
         * cache_fp4[il] / layer_index_comp_scale[il]); the CUDA _direct kernel
         * reads it bit-identically and arms the read tripwire.  NULL = F32
         * (Metal accepts + ignores). */
        ds4_gpu_tensor       *index_fp4,
        ds4_gpu_tensor       *index_scale);

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        /* P2 Inc3b: optional packed FP4 indexer mirror (NULL = F32; Metal
         * accepts + ignores).  Prefill uses the WMMA tile readers. */
        ds4_gpu_tensor       *index_fp4,
        ds4_gpu_tensor       *index_scale);

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        /* Phase 2 Step 4b: per-seq index_comp bank stride + optional
         * per-row positions[]/seq_id[] (NULL = single seq, bit-exact). */
        uint32_t                comp_cap,
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        /* FE1: bank id when the caller proves the batch is ONE bank's
         * consecutive-position run (admission chunks); the CUDA backend then
         * serves the scores on the serial WMMA path through a bank-offset
         * view.  UINT32_MAX = no claim (per-row kernel, bit-exact). */
        uint32_t                single_bank,
        /* P2 Inc3b: optional packed FP4 indexer mirror (NULL = F32; Metal
         * accepts + ignores).  multiseq + WMMA readers decode it. */
        ds4_gpu_tensor       *index_fp4,
        ds4_gpu_tensor       *index_scale,
        /* v0.3 V5D: indexer-Q QAT block scales ([n_tokens, 64, 4] F32 --
         * the scale-only emit of ds4_gpu_dsv4_indexer_qat_tensor).  Non-NULL
         * routes the CUDA multiseq scores onto the WMMA V5D tile kernel
         * (3.97x); NULL = legacy scalar per-row kernel.  Metal accepts +
         * ignores. */
        ds4_gpu_tensor       *q_block_scale);

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k,
        /* PC5: max-grid + substrate params for captured-decode safety.
         * When il_for_decode1 < DS4_LAYER_SCALARS_COUNT, the CUDA backend
         * picks kernel specialization by n_comp_max (capture-stable) and
         * the kernel reads the live count from g_layer_dev[il].n_index_comp
         * at execute time -- closes the "live score producer, stale top-k
         * consumer" capture bug.  Decode-body caller passes (g->layer_comp_cap[il],
         * il); every other caller (prefill, decode2-exact, output-head
         * vocab top-k, Metal stub) passes (0, UINT32_MAX) for legacy. */
        uint32_t                n_comp_max,
        uint32_t                il_for_decode1);

/* v0.5 inc-13a: combined score+select for eager prefill chunks at all
 * depths (exact CUDA mxf4 coarse + exact top-512 over f32 rows; the
 * inc-5/inc-7 pool+rescore chain is retired).  When it engages it writes
 * F32 score rows into the scores scratch plus the top-512 selected ids
 * (stream-tier byte order, 0xFFFFFFFF sentinels) and sets *engaged = 1;
 * the caller must then SKIP the classic scores+topk pair.  *engaged = 0
 * means fall through (scores scratch untouched).  Returns 0 only on a
 * launch error.  Never engages inside stream capture, without the FP4
 * indexer mirror, at n_comp < 1024 or n_tokens < 32, or on non-sm_121a
 * builds; escape hatch DS4_CUDA_NO_INDEXER_MXF4.  single_bank/comp_cap
 * mirror the FE1 bank-view convention of the scores entry (UINT32_MAX/0 =
 * no bank indirection).  v0.5 inc-13a.2: q_codes/q_scale4 carry the
 * producer-emitted QAT mirror of q (64 B nibble codes + 4 F32 pow2 block
 * scales per (token, head) row, the ds4_gpu_dsv4_indexer_qat_tensor
 * mirror layout); when both are present the chain stages Q straight from
 * the mirror and the internal re-quant launch (plus its full F32 re-read
 * of q) is skipped -- bit-identical coarse inputs by the QAT emit
 * construction.  NULL/NULL = internal re-quant (legacy); escape hatch
 * DS4_CUDA_NO_INDEXER_QMIRROR forces the legacy path for twin gates.
 * Metal: stub, never engages. */
int ds4_gpu_indexer_score_select_prefill_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                top_k,
        float                   scale,
        uint32_t                single_bank,
        uint32_t                comp_cap,
        ds4_gpu_tensor       *index_fp4,
        ds4_gpu_tensor       *index_scale,
        const ds4_gpu_tensor *q_codes,
        const ds4_gpu_tensor *q_scale4,
        int                    *engaged);

/* GPU argmax over n_vocab F32 logits. Writes the winning index as int32 at
 * out_idx[0]. Tie-break: lower index wins (matches host sample_argmax). */
int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab);

/* Batched argmax: one winning index per row over a [n_rows x n_vocab] logits
 * tensor.  Writes n_rows int32 ids to out_idx[0..n_rows).  Same tie-break as
 * ds4_gpu_argmax_tensor (larger value, lower index).  Runs on the current
 * stream so it joins the draft command buffer (one end_commands sync covers
 * all rows). */
int ds4_gpu_argmax_rows_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab,
        uint32_t                n_rows);

/* DSpark D4.5c on-device Markov refine step: per bank b (0..nb), argmax over
 * (base_logits[(b*B+blk_pos)*vocab + v] + bias[b*vocab + v]) -> cand_out[b].
 * bias = markov_w2 @ markov_w1[prev]; base_logits = the [nb*B, vocab] block output. */
int ds4_gpu_dspark_markov_step_tensor(
        ds4_gpu_tensor       *cand_out,
        const ds4_gpu_tensor *base_logits,
        uint32_t                blk_pos,
        uint32_t                B,
        const ds4_gpu_tensor *bias,
        uint32_t                vocab,
        uint32_t                nb);

/* Pack selected DSpark capture rows into a contiguous [n_rows x row_floats]
 * scratch tensor.  src_rows is an int32 device tensor of source row indices. */
int ds4_gpu_dspark_gather_concat_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        const ds4_gpu_tensor *src_rows,
        uint32_t                n_rows,
        uint32_t                row_floats);

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_top2_tensor(
        ds4_gpu_tensor       *top2,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x);

int ds4_gpu_matmul_q8_0_top2_and_logits_n2_tensor(
        ds4_gpu_tensor       *row0_top2,
        ds4_gpu_tensor       *row1_logits,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x2);

int ds4_gpu_matmul_q8_0_candidates_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *candidate_ids,
        uint32_t                candidate_count,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x);

int ds4_gpu_q8_0_row_group_norms_tensor(
        ds4_gpu_tensor       *row_group_norms,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        uint32_t                group_count);

ds4_gpu_tensor *ds4_gpu_imported_q8_0_row_group_norms_tensor(
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        uint32_t                group_count);

int ds4_gpu_matmul_q8_0_candidate_certify_tensor(
        ds4_gpu_tensor       *result,
        const ds4_gpu_tensor *row_group_norms,
        const ds4_gpu_tensor *candidate_ids,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint32_t                group_count);

int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* P3-Inc1: f16 matmul with the input rms_norm folded into the activation
 * convert (f16 activations bit-identical to the unfused rms_norm_plain +
 * f32_to_f16 chain).  Returns 0 with `out` untouched on any precondition
 * miss (fold disabled, cuBLAS unavailable, n_tok at or below the native-f16
 * tier); the caller must then run the unfused chain. */
int ds4_gpu_matmul_f16_rms_fold_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   norm_eps);

/* P3-Inc2: f16 matmul on pre-converted f16 activations (see
 * ds4_gpu_hc_expand_rmsf16_split_tensor, which emits them).  Returns 0 with
 * `out` untouched on any precondition miss. */
int ds4_gpu_matmul_f16_preconv_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *xh,
        uint64_t                n_tok);

/* v0.5 inc-11 F1: shared f16 activation mirror.  ..._preconv_ready says
 * whether the cublas f16 tier (the only tier that converts) would serve
 * n_tok -- fill a shared mirror only then.  ..._f32_to_f16_tensor fills it
 * with the SAME kernel the f16 helper uses per call (bit-identical values).
 * Both are CUDA-only; Metal returns 0 and callers keep the classic helper. */
int ds4_gpu_matmul_f16_preconv_ready(uint64_t n_tok);
int ds4_gpu_f32_to_f16_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        uint64_t                count);

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Batched repeat_hc: for each of n_rows embd-vectors in `rows` ([n_rows x
 * n_embd]), broadcast it into n_hc consecutive copies, writing
 * [n_rows x n_hc x n_embd] to `out` (row-major). */
int ds4_gpu_repeat_hc_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *rows,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                n_rows);

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

/* v0.5 inc-12c: dual-emit variant -- out gets the identical f32 result and
 * out_f16 the __float2half of the same value (bit-identical to running the
 * classic entry followed by a f32->f16 convert launch).  Returns 0 with both
 * outputs untouched on decline (Metal declines always); the caller falls
 * back to the classic entry + convert. */
int ds4_gpu_rms_norm_weight_rows_f16_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *out_f16,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

/* emit_q81 (M2-Inc2a, CUDA decode rows==1 only): nonzero additionally emits
 * q8_1 codes of the q row in-kernel and hands them to the next mmvq consumer
 * (q_b), eliding its quantize prelude; ignored elsewhere/on Metal. */
int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps,
        uint32_t                emit_q81);

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

/* Opp C Phase 1A: batched form.  Same mirror semantics as the row variant
 * below -- pass codes_mirror / scale_mirror NULL on Metal and at sites
 * that do not feed the FP8 mirror (e.g. raw KV, working batch_kv,
 * indexer cache).  When non-NULL the kernel writes packed codes + scales
 * + FP32 rotary tail for rows [0..n_tok).  Callers that want to mirror
 * into the MIDDLE of the per-layer mirror buffer pass tensor views at
 * the matching row offsets. */
int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot,
        ds4_gpu_tensor *codes_mirror,
        ds4_gpu_tensor *scale_mirror);

/* R1 row-variant (Step 4c R1' migration to layer-scalars substrate):
 * writes one row of `base` at the index taken from the per-layer device
 * array g_layer_dev[il].comp_row.  Replaces the (transient view, n_tok=1)
 * form used in the decode-time compressor emit path so the captured
 * kernel-node arg list bakes a stable base pointer + a per-layer baked
 * ls pointer, not a per-token row pointer.  The shim computes the
 * per-layer ls = &g_layer_dev[il] internally.  See plan doc R1 + sec
 * 15.4. */
/* Opp C Phase 1A: when DS4_CUDA_FP8_KV is enabled, `codes_mirror` and
 * `scale_mirror` are the per-layer packed FP8 mirror tensors
 * (g->layer_comp_cache_fp8[il] / g->layer_comp_scale[il]); the kernel
 * writes the 1-byte E4M3 codes + per-64-lane scales + FP32 rotary tail to
 * those buffers in addition to its existing in-place FP32 quantisation of
 * `base`.  Pass NULL for both when the feature is disabled; the buffers
 * stay NULL through the layer-graph key for stable capture/replay. */
/* C1 (cont capture): optional `src` override for the fp8/fp4-primary
 * continuous path, where the F32 emit lands in a shared scratch row 0
 * (the bank's F32 cache page is write-dead and never mapped).  When src
 * is non-NULL the kernel does its in-place F32 round-trip on src row 0
 * and writes the packed mirror rows at g_layer_dev[il].comp_row as
 * before -- base may then be NULL.  NULL src = row addressed inside
 * `base` (serial decode path, unchanged). */
int ds4_gpu_dsv4_fp8_kv_quantize_row_tensor(
        ds4_gpu_tensor *base,
        uint32_t          head_dim,
        uint32_t          n_rot,
        /* C2 Inc2: row source select.  row_table=0 -> the per-layer
         * substrate row &g_layer_dev[il].comp_row (serial captured decode;
         * row_idx ignored).  row_table=1 -> the per-ROW emit-row table
         * entry (row_idx, il), whose published value carries bank base +
         * within-step ordering (cont capture steps). */
        uint32_t          row_idx,
        uint32_t          il,
        int               row_table,
        ds4_gpu_tensor *codes_mirror,
        ds4_gpu_tensor *scale_mirror,
        ds4_gpu_tensor *src);

/* P2 Inc2b: expand packed FP8 rows back to FP32 -- the exact inverse of the
 * quantize encode (each lane reconstructs bit-identically to what the
 * attention-side fp8_kv_read returns).  `codes`/`scale` are views positioned
 * at the first row to expand; `dst` receives rows [0..n_rows) densely.  Used
 * by the fp8-primary paths that still need an F32 image of stored rows:
 * checkpoint save, the partial-fork boundary-row stash, and the MTP
 * committed-context host read.  CUDA only -- Metal never stores FP8 and its
 * stub fails loudly. */
int ds4_gpu_dsv4_fp8_kv_dequant_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *codes,
        const ds4_gpu_tensor *scale,
        uint32_t                n_rows,
        uint32_t                head_dim);

/* P2 Inc3: when DS4_CUDA_FP4_INDEX is enabled, `codes_mirror`/`scale_mirror`
 * are the per-layer packed FP4 indexer mirror tensors
 * (g->layer_index_comp_cache_fp4[il] / g->layer_index_comp_scale[il]); the QAT
 * kernel writes the 4-bit e2m1 codes (64 B/row) + per-32-lane block scales
 * (16 B/row) in addition to its in-place F32 quantisation.  Both NULL when the
 * feature is off (Metal always passes NULL). */
int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim,
        ds4_gpu_tensor *codes_mirror,
        ds4_gpu_tensor *scale_mirror);

/* R1 row-variant (Step 4c R1' + C2 Inc2): writes one row of `base` at the
 * index taken from the per-ROW emit-row table entry (row_idx, il).
 * C1 (cont capture): `src` override mirrors fp8_kv_quantize_row_tensor --
 * non-NULL src = F32 work on src row 0, packed mirror rows at the table
 * row, base may be NULL. */
int ds4_gpu_dsv4_indexer_qat_row_tensor(
        ds4_gpu_tensor *base,
        uint32_t          head_dim,
        uint32_t          row_idx,
        uint32_t          il,
        int               row_table,
        ds4_gpu_tensor *codes_mirror,
        ds4_gpu_tensor *scale_mirror,
        ds4_gpu_tensor *src);

/* P2 Inc3: expand packed FP4 indexer rows back to F32 -- exact inverse of the
 * encode (bit-identical to indexer_fp4_read).  `codes`/`scale` are views at
 * the first row to expand; `dst` receives rows [0..n_rows) densely (head_dim
 * 128).  Used by the fp4-primary host paths (checkpoint save, partial-fork
 * boundary stash, MTP committed read).  CUDA only -- Metal stub fails loud. */
int ds4_gpu_dsv4_indexer_fp4_dequant_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *codes,
        const ds4_gpu_tensor *scale,
        uint32_t                n_rows,
        uint32_t                head_dim);

/* P2 Inc3b: pack already-committed F32 index rows into the packed FP4 mirror
 * (no Hadamard -- the input is the post-QAT committed value).  Rewrites `x`
 * with the reconstructed value too, so F32 and codes stay consistent.  Used by
 * session load to refresh a stale mirror.  CUDA only -- Metal stub fails loud. */
int ds4_gpu_dsv4_indexer_fp4_encode_tensor(
        ds4_gpu_tensor       *x,
        ds4_gpu_tensor       *codes,
        ds4_gpu_tensor       *scale,
        uint32_t                n_rows,
        uint32_t                head_dim);

/* P2 Inc3: 1 when packed FP4 indexer storage is engaged (default ON since
 * v0.2.4; DS4_CUDA_FP4_INDEX=0 disables, VMM guard can refuse). */
int ds4_cuda_fp4_index_enabled(void);

int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        /* Phase 2 Step 2: optional per-row absolute positions (int32 device
         * tensor, n_tok entries).  NULL -> scalar fast-path (pos0 + t). */
        const ds4_gpu_tensor *positions,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow);

/* Full-layer-graph-capture-compatible variant of the above (Step 3 pilot).
 * Reads pos0 from the device-side decode_scalars struct rather than baking
 * it into the kernel-node argument list.  `scalars` is the opaque pointer
 * returned by ds4_gpu_decode_scalars_device_ptr().  `pos_offset` is added
 * to s->pos0 at execution time (signed; pass 0 for plain decode, pass
 * 1-(int)ratio for the compressor-emit caller).  `pos_stride` matches the
 * inline-pos0 shim's hardcoded 1; pass higher values for batched prefill.
 *
 * Backends that don't implement layer-graph capture provide a stub that
 * still does the right thing numerically (Metal stub computes pos0 on the
 * CPU from the same struct fields).  Returns 1 on success, 0 on failure. */
int ds4_gpu_rope_tail_scalars_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        const void       *scalars,
        int32_t           pos_offset,
        uint32_t          pos_stride,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot,
        /* PC4 (K0): optional device-scalars override.  Decode-time caller
         * passes ds4_gpu_decode_scalars_device_ptr() so the raw-store
         * kernel reads raw_row from g_decode_dev at execution time --
         * capture-safe.  Decode2-exact path passes NULL (kernel uses
         * inline row).  Metal backend ignores the argument. */
        const void       *scalars);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ds4_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim,
        /* PC4 (K0): same semantics as the _fp8_store_raw variant above. */
        const void             *scalars);

int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim,
        /* Phase 2 Step 3: optional per-row positions[]/seq_id[] (int32 device
         * tensors, n_tokens entries).  NULL = single-sequence scalar path. */
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id);

/* =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries.  Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.
 */

/* PC2: row-field selector for ds4_gpu_compressor_update_tensor().  The
 * shim has two distinct callers in decode1 (primary compressor + indexer
 * compressor) which need to read different fields from the same per-layer
 * substrate entry.  Encoded as a tiny enum-like pair rather than packing
 * into the high bit of `il` (which complicates Step 5's cache-key
 * machinery).  Decode2-exact callers pass DS4_COMPRESSOR_ROW_COMP with
 * il = UINT32_MAX -- row_field is then ignored. */
#define DS4_COMPRESSOR_ROW_COMP   0
#define DS4_COMPRESSOR_ROW_INDEX  1
/* C1 (cont capture): fp8/fp4-primary emit into a shared scratch row 0.
 * Substrate mode for the store pos (kernel reads s->pos0) but the emit-tail
 * kernels write FIXED row 0 of the passed comp_cache (the scratch) instead
 * of dereferencing g_layer_dev[il].comp_row -- the packed-mirror row pack
 * that follows (fp8/fp4 row-variant with `src`) is what lands at the
 * substrate row.  Requires il < layer count (fails loud otherwise). */
#define DS4_COMPRESSOR_ROW_SCRATCH0 2
/* C2 Inc2 (cont capture): emit-tail row read from the per-ROW emit-row
 * table entry (row_idx, il) -- &g_row_dev[row_idx*COUNT+il].comp_row /
 * .index_row.  The published value already carries the bank base and the
 * within-step emit ordering, so ONE captured graph serves single-bank AND
 * multi-live shapes.  Requires il < layer count + a published table
 * (fails loud otherwise). */
#define DS4_COMPRESSOR_ROW_TABLE_COMP  3
#define DS4_COMPRESSOR_ROW_TABLE_INDEX 4

/* M2-Inc5 (CUDA): one decode compressor event -- BOTH pair f16 matmuls
 * (kv + gate, in_dim x width), their split-K combines, and the compressor
 * store -- as one cooperative kernel, bit-identical to the unfused
 * pair-matmul + store chain.  Returns 0 without touching any output when
 * the shape/mode is unsupported, an f16 dispatch env knob is set, the
 * cooperative launch is unavailable, or DS4_CUDA_NO_COMP_PAIR_FUSED=1; the
 * caller then runs the unfused chain.  On success the caller must follow
 * with ds4_gpu_compressor_update_tail_tensor (NOT the full update -- the
 * store has already run). */
int ds4_gpu_compressor_pair_store_fused_tensor(
        ds4_gpu_tensor       *kv_cur,
        ds4_gpu_tensor       *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                w_kv_offset,
        uint64_t                w_sc_offset,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint64_t                in_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        uint32_t                il);

/* C3-Inc2 (CUDA): batched-decode rows variant of the fused compressor pair --
 * BOTH bulk f16 matmuls over n_tok <= 8 x rows and their split-K combines as
 * one cooperative kernel, bit-identical to the two-bulk-matmul chain (at
 * n_tok <= the native-f16 cap each row runs the exact n=1 split-K+combine).
 * do_store != 0 additionally fuses the compressor store -- ONLY valid on the
 * single-row (n_tok == 1) multiseq emit shape, because at n > 1 the per-row
 * emit loop interleaves stores with pool reads (same-bank chains reuse ring
 * slots) and hoisted stores would reorder them.  positions/seq_id are the
 * per-step device substrates (per-row pos + bank lane on the FULL state
 * slabs, resolved at execution time -- bank-agnostic under capture; the
 * caller validates the lane extent).  Returns 0 without touching any output
 * on any precondition miss; the caller then runs the unfused chain.  On a
 * do_store success the caller must follow with
 * ds4_gpu_compressor_update_tail_tensor (NOT the full update). */
int ds4_gpu_compressor_pair_store_rows_fused_tensor(
        ds4_gpu_tensor       *kv_cur,
        ds4_gpu_tensor       *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                w_kv_offset,
        uint64_t                w_sc_offset,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint64_t                in_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        uint32_t                n_tok,
        int                     do_store);

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        /* Step 4c C2 + PC2: per-layer substrate selector.  Decode1 path
         * passes il (0..42) + DS4_COMPRESSOR_ROW_COMP (primary) or
         * DS4_COMPRESSOR_ROW_INDEX (indexer).  Decode2-exact + Metal
         * callers pass il = UINT32_MAX to signal "no substrate"; kernels
         * fall back to inline comp_row and row_field is ignored.  See
         * plan doc sec 16 commit C2 / sec 15.8. */
        uint32_t                il,
        int                     row_field,
        /* C2 (bank-agnostic + multi-live cont capture): per-row device
         * substrate.  positions non-NULL: store/rope position read at
         * execution time from positions[row_idx] (per-step device array).
         * seq_id non-NULL: state_kv/state_score are FULL multi-bank slabs,
         * kernels offset to lane seq_id[row_idx] (stride coff*ratio*width;
         * the CALLER validates the lane extent against the slab).
         * row_field TABLE_COMP/TABLE_INDEX read the emit row from the
         * per-ROW table entry (row_idx, il) -- published value carries the
         * bank base + within-step emit ordering.  NULL/NULL/0 for every
         * serial / eager caller; Metal backend rejects seq_id != NULL. */
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        uint32_t                row_idx,
        /* 1 => comp_cache is an F16 buffer (the banked attn emit passes
         * layer_attn_comp_cache[il] direct): run pool/rms/rope in an F32
         * scratch then copy_f32_to_f16 into the row.  0 => F32 cache. */
        int                     output_is_f16);

/* M2-Inc5: emit tail of ds4_gpu_compressor_update_tensor only (pool + norm +
 * rope + ratio4 shift; no store) -- the follow-up call after a successful
 * ds4_gpu_compressor_pair_store_fused_tensor.  Same signature as the full
 * update. */
int ds4_gpu_compressor_update_tail_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        uint32_t                il,
        int                     row_field,
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        uint32_t                row_idx,
        int                     output_is_f16);

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        /* Step 4c C1: optional device-scalars override.  Decode-time
         * caller passes ds4_gpu_decode_scalars_device_ptr(); prefill +
         * batch callers pass NULL (kernel uses inline pos0).  Metal
         * backend ignores the argument. */
        const void             *scalars,
        /* C2 Inc1: per-row device substrate (see the update shim above);
         * NULL/NULL/0 for every non-capture caller. */
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        uint32_t                row_idx);

/* C2 Inc1: rollback-checkpoint lane copy for cont capture steps.  dst/src
 * are FULL multi-bank state slabs; the copied lane is resolved on device
 * from seq_id[row_idx] (so captured copy nodes are bank-agnostic).
 * lane_bytes = one bank's state stride; seq_check = the host's lane belief
 * (extent validation only).  Metal stub returns 0. */
int ds4_gpu_state_lane_copy_tensor(
        ds4_gpu_tensor       *dst_kv,
        ds4_gpu_tensor       *dst_sc,
        const ds4_gpu_tensor *src_kv,
        const ds4_gpu_tensor *src_sc,
        const ds4_gpu_tensor *seq_id,
        uint32_t                row_idx,
        uint64_t                lane_bytes,
        uint32_t                seq_check);

int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        /* Opp C Phase 1A: see the ds4_gpu_dsv4_fp8_kv_quantize_tensor comment. */
        ds4_gpu_tensor       *comp_cache_fp8,
        ds4_gpu_tensor       *comp_scale);

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        ds4_gpu_tensor       *comp_cache_fp8,
        ds4_gpu_tensor       *comp_scale);

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0);

/* Decode-time attention shim (n_tok=1).  `scalars` is the opaque device-side
 * decode_scalars pointer from ds4_gpu_decode_scalars_device_ptr(); when
 * non-NULL the kernel reads n_raw, raw_start, n_comp from the struct at
 * execution time instead of from the inline args (Step-4 / R5 invariant).
 * Pass NULL for callers that don't participate in graph capture. */
int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim,
        const void             *scalars,
        /* Step 4c A1: per-layer index for the ds4_layer_scalars substrate.
         * Decode1 passes il (0..42) to lift n_comp off the inline arg;
         * decode2-exact + batch callers pass UINT32_MAX (no substrate). */
        uint32_t                il_for_decode1,
        /* Opp C Phase 1A.3: optional packed FP8 mirror of the compressed
         * rows (g->layer_comp_cache_fp8[il] / g->layer_comp_scale[il]).
         * NULL/NULL when DS4_CUDA_FP8_KV is off; the dense kernel then
         * keeps reading the FP32 cache bit-identically.  Only consulted
         * by the gridX=1 dense decode branch -- the score-buffer / window
         * fallbacks ignore it. */
        const ds4_gpu_tensor *comp_fp8,
        const ds4_gpu_tensor *comp_scale);

int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim,
        /* Phase 2 Step 3: optional per-row positions[]/seq_id[] (NULL = single seq). */
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        /* M4b Inc3: optional per-row raw-span override for the batched MTP draft
         * (NULL = bit-exact for the normal batched decode). */
        const ds4_gpu_tensor *draft_n_raw,
        /* FE2: caller opt-in for the per-seq heads8-online fast path (the
         * admission-chunk forward); 0 = pre-FE2 dispatch, bit-exact. */
        uint32_t                allow_mseq_heads8,
        /* C1 (cont capture): optional decode-scalars substrate + per-layer
         * index.  When scalars != NULL the kernel reads pos0/n_raw/raw_start
         * live from the device struct at execution time (capture-safe); when
         * il < layer count it reads n_comp from g_layer_dev[il] (raw layers
         * publish 0).  (NULL, UINT32_MAX) = inline args, bit-exact legacy. */
        const void             *scalars,
        uint32_t                il_for_decode1,
        /* Token-tile eligibility hint: see
         * ds4_gpu_attention_indexed_mixed_batch_heads_tensor. */
        uint32_t                tt_run_pos0);

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        /* Phase 2 Step 4a: per-seq compressed-cache bank stride (rows); the
         * kernel reads row t's compressed rows from bank seq_id[t]*comp_cap.
         * Ignored when seq_id==NULL (single bank, bit-exact). */
        uint32_t                comp_cap,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        /* Opp C Phase 1A.3: optional packed FP8 mirror; see
         * ds4_gpu_attention_decode_heads_tensor for semantics. */
        const ds4_gpu_tensor *comp_fp8,
        const ds4_gpu_tensor *comp_scale,
        /* Phase 2 Step 3: optional per-row positions[]/seq_id[] (NULL = single seq). */
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        /* FB1: caller opt-in for the per-seq heads8-online fast path (the
         * admission-chunk forward; wide single-bank batches).  0 keeps every
         * per-seq caller on the generic kernel (pre-FB1 dispatch, bit-exact). */
        uint32_t                allow_mseq_heads8,
        /* C1 (cont capture): see ds4_gpu_attention_decode_raw_batch_heads_
         * tensor.  Substrate present forces the generic kernel (the heads8
         * online tier bakes inline window scalars). */
        const void             *scalars,
        uint32_t                il_for_decode1,
        /* Token-tile eligibility hint: see
         * ds4_gpu_attention_indexed_mixed_batch_heads_tensor. */
        uint32_t                tt_run_pos0);

/* Decode-time indexed-attention shim.  See ds4_gpu_attention_decode_heads_
 * tensor for the `scalars` semantics.  Pass NULL for the batched/prefill
 * callers; pass ds4_gpu_decode_scalars_device_ptr() for the in-decode-
 * body caller. */
int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        /* Phase 2 Step 4b: per-seq compressed-cache bank stride (rows). */
        uint32_t                comp_cap,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const void             *scalars,
        /* Step 4c A1: per-layer index for ds4_layer_scalars substrate.
         * UINT32_MAX = no substrate. */
        uint32_t                il_for_decode1,
        /* Opp C Phase 1A.4: optional packed FP8 mirror; see
         * ds4_gpu_attention_decode_heads_tensor for semantics.  Read by
         * the generic attention_indexed_mixed_kernel branch -- the one
         * that fires for all decode (n_tokens==1) and also for the
         * batch path (n_tokens>1) when DS4_CUDA_NO_INDEXED_HEADS8
         * disables the heads8 fast path.  The heads8 variants
         * themselves still read FP32. */
        const ds4_gpu_tensor *comp_fp8,
        const ds4_gpu_tensor *comp_scale,
        /* Phase 2 Step 3: optional per-row positions[]/seq_id[] (NULL = single seq). */
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        /* FB1: caller opt-in for the per-seq heads8-online fast path; 0 =
         * pre-FB1 dispatch (per-seq always generic), bit-exact. */
        uint32_t                allow_mseq_heads8,
        /* Token-tile eligibility hint (2026-07-09): when the batch is one
         * sequence's consecutive-position run, the position of row 0
         * (host-verified against the ms_* mirrors); UINT32_MAX otherwise.
         * Consumed only by the CUDA token-tile prefill branch; every other
         * backend/branch ignores it. */
        uint32_t                tt_run_pos0);

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        /* P2 Inc2b: optional packed FP8 mirror of the compressed rows
         * (CUDA only; NULL/NULL when DS4_CUDA_FP8_KV is off, ignored on
         * Metal).  Same row-0 base as comp_kv. */
        const ds4_gpu_tensor *comp_fp8,
        const ds4_gpu_tensor *comp_scale);

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        /* P2 Inc2b: optional packed FP8 mirror (CUDA only, see above). */
        const ds4_gpu_tensor *comp_fp8,
        const ds4_gpu_tensor *comp_scale);

int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

/* v0.5 inc-10 F2: batch entry above with the attention-output inverse RoPE
 * tail folded into the f16 group pack of the cublas out_a branch (CUDA
 * only).  Returns 0 with every output untouched when that branch would not
 * be taken -- the caller must then run the classic
 * ds4_gpu_rope_tail_tensor(inverse) + ds4_gpu_attention_output_q8_batch_
 * tensor pair.  On success `heads` is left UN-rotated in f32; callers must
 * not read it expecting post-rope values.
 *
 * v0.5 flat-pool p1: on the wide batch path this entry now serves out_a
 * with ONE fused own kernel (f32 heads read direct, rope via a (c,s)
 * table companion, aligned-Q8_0 in-register dequant, interleaved f32 low
 * write) -- the f16 pack + cublas pair and the q8->f16 out_a cache are
 * retired on that path.  VALUE-parity vs the pair (HMMA k-order), staged
 * f16 inputs bit-identical by construction.  Kill switch
 * DS4_CUDA_NO_OUTA_OWN restores the pack+cublas pair (and the inc-12f
 * f16-cache boot prebuild) exactly. */
int ds4_gpu_attention_output_q8_batch_inverse_rope_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens,
        uint32_t                head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        const ds4_gpu_tensor *positions,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow);

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads);

int ds4_gpu_attention_output_low_q8_batch_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/Q4_K routed experts.
 */

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n);

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

/* M2-Inc3 (CUDA): the whole decode router stage -- f16 logits matmul
 * (in_dim x n_expert), split-K combine, and top-6 select -- as one
 * cooperative kernel, bit-identical to the unfused
 * matmul_f16 + router_select chain.  Returns 0 without touching any output
 * when the shape/mode is unsupported, a router/f16 bisect env knob is set,
 * the cooperative launch is unavailable, or DS4_CUDA_NO_ROUTER_FUSED=1;
 * the caller then runs the unfused chain. */
int ds4_gpu_router_fused_tensor(
        ds4_gpu_tensor       *logits,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *x,
        uint64_t                in_dim,
        uint64_t                n_tok);

/* C3-Inc1: batched fused router (n_tok <= 8 rows, one coop launch).  Bit-
 * exact twin of the batch unfused chain (per-row split-K matmul + combine +
 * router_select_warp_topk); returns 0 to fall back on any precondition miss.
 * tokens = device per-row token-id tensor (hash routing). */
int ds4_gpu_router_fused_batch_tensor(
        ds4_gpu_tensor       *logits,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *tokens,
        uint64_t                in_dim,
        uint64_t                n_tok);

int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits);

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x);

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16);

/* v0.5 inc-8 F5: defer-sum sibling of the batch entry.  When it returns 1
 * with *out_sum_deferred = 1, `down` holds per-expert unsummed rows and
 * `out` is NOT written — consume via the moe-variant fused expand or
 * ds4_gpu_moe_sum_tensor.  Metal never defers (forwards + reports 0). */
int ds4_gpu_routed_moe_batch_defer_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16,
        int                    *out_sum_deferred);

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* DSpark D4.5c inline capture: mean over n_hc HC lanes of `hc` ([n_tokens x
 * n_hc x n_embd]) -> `concat` ([n_tokens x (n_slots*n_embd)]) at column
 * slot*n_embd.  Enqueue right after a layer's encode to snapshot its mean hidden
 * into the fusion concat without a readback. */
int ds4_gpu_dspark_capture_mean_tensor(
        ds4_gpu_tensor       *concat,
        const ds4_gpu_tensor *hc,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                n_tokens,
        uint32_t                slot,
        uint32_t                n_slots);

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps);

/* M2-Inc1: fused decode HC stage.  Drop-in for the three-call chain
 *   rms_norm_plain(flat_scratch, residual_hc) ->
 *   matmul_f16(mix, fn_weight, flat_scratch) ->
 *   hc_split_weighted_sum_norm(out, norm_out, split, mix, residual_hc).
 * CUDA runs one cooperative kernel when preconditions hold (n_hc==4, F16 fn
 * weights, decode single row); otherwise -- and always on Metal -- it
 * executes the unfused chain.  flat_scratch is only written on the fallback
 * path.  Kill switch: DS4_CUDA_NO_HC_STAGE_FUSED.
 * emit_q8 (M2-Inc1b, CUDA fused path only): bit 0 emits q8_0 codes of
 * norm_out in-kernel and hands them to the next consumer entry, eliding its
 * quantize prelude; pass 0 unless a q8_0 GEMV consumes norm_out next.  Fold
 * kill switch: DS4_CUDA_NO_HC_Q8_FOLD. */
int ds4_gpu_hc_stage_fused_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        ds4_gpu_tensor       *mix,
        ds4_gpu_tensor       *flat_scratch,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                fn_weight_offset,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps,
        uint32_t                emit_q8);

/* M2-Inc2b: fused head_rms_norm + q rope_tail with device-scalars pos
 * (capture-safe).  Returns 0 on any precondition miss and ALWAYS on Metal;
 * callers fall back to the separate head_rms_norm + rope_tail_scalars
 * chain, which is bit-exact to this kernel. */
/* v0.5 inc-8: fused per-head RMS-norm + q-rope tail for the eager batch
 * path (one read/write of the head row instead of two; math bit-exact vs
 * the head_rms_norm + rope_tail pair — the tail values get the rms scale
 * applied inline before rotation).  positions: optional per-row absolute
 * positions (n_tok int32), NULL = scalar pos0 + t.  Returns 0 when not
 * taken (shape mismatch or Metal) — callers fall back to the pair. */
int ds4_gpu_head_rms_norm_rope_tail_tensor(
        ds4_gpu_tensor       *x,
        uint32_t                n_tok,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        const ds4_gpu_tensor *positions,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps);

int ds4_gpu_head_rms_norm_rope_tail_scalars_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, const void *scalars, int32_t pos_offset, uint32_t pos_stride,
        uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow,
        float eps);

/* M2-Inc2c: fused decode kv rope_tail + FP8 KV quantize + raw-cache store
 * (single row; pos and raw_row from the device scalars substrate).  Returns
 * 0 on any precondition miss and ALWAYS on Metal; callers fall back to the
 * rope_tail_scalars + fp8_kv_quantize + store_raw_kv chain (bit-exact). */
int ds4_gpu_kv_rope_fp8_store_scalars_tensor(
        ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache, uint32_t raw_cap,
        uint32_t head_dim, uint32_t n_rot, const void *scalars, int32_t pos_offset,
        uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow);

int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* P3-Inc2: fused batched hc_expand + next-HC-stage rms_norm f16 emission
 * (block_add == NULL -> plain expand, non-NULL -> add variant).  xh_out
 * receives the next mix GEMM's f16 activations.  1-ulp FMA-contraction
 * class vs the unfused expand + rms_norm_plain + f32_to_f16 chain (NOT
 * bit-exact; bounded-ulp selftest + quality battery gate it).  Returns 0
 * with all outputs untouched on any precondition miss. */
int ds4_gpu_hc_expand_rmsf16_split_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *xh_out,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        float                   norm_eps);

/* v0.5 inc-8 F5: moe-variant of the fused expand — consumes the PER-EXPERT
 * UNSUMMED routed down output (from the defer-sum MoE entry) and folds the
 * guarded expert sum + shared add + HC expand + RMS + f16 emit into one
 * kernel.  Returns 0 when not taken; callers then run
 * ds4_gpu_moe_sum_tensor + the unfused expand. */
int ds4_gpu_hc_expand_rmsf16_split_moe_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *xh_out,
        const ds4_gpu_tensor *moe_down_unsummed,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                n_expert_used,
        float                   norm_eps);

/* v0.5 inc-8 F5: guarded standalone expert sum (bit-identical to the MoE
 * path's internal sum) — the fallback consumer for a deferred sum. */
int ds4_gpu_moe_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *down,
        uint32_t                out_dim,
        uint32_t                n_expert,
        uint32_t                n_tokens);

int ds4_gpu_hc_expand_add_split_n2_rows_tensor(
        ds4_gpu_tensor       *out0_hc,
        ds4_gpu_tensor       *out1_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_matmul_q8_0_hc_expand_n2_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_matmul_q8_0_hc_expand_n2_split_residual_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual0_hc,
        const ds4_gpu_tensor *residual1_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

#endif
