#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stdint.h>

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

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_end_commands(void);
int ds4_gpu_synchronize(void);

/* =========================================================================
 * Decode-time position scalars (full-layer CUDA-graph capture, Step A).
 * =========================================================================
 *
 * Two parallel substrates carry position-derived scalars to position-
 * dependent kernels:
 *
 *   1. TOKEN-STABLE struct (ds4_decode_scalars, 40 B device-side).  Carries
 *      scalars whose value is constant across all 43 layers within a token:
 *      pos0, raw_row, raw_start, n_raw, emit_phase, flags.  Single pinned
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
        uint32_t flags);
/* Push the most recent ds4_gpu_decode_scalars_set() values to the device-side
 * mirror.  Backends that don't use the graph-capture path implement this as
 * a no-op.  On CUDA this issues a single H2D memcpy on ds4_current_stream(),
 * eager (outside captured-graph scope) today; Step 6 may bring it inside a
 * wider per-token graph at which point the memcpy is captured into the
 * outer graph node list.  Returns 1 on success / no-op, 0 on infrastructure
 * failure. */
int   ds4_gpu_decode_scalars_flush(void);

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

/* =========================================================================
 * Step 5/6: per-layer cudaGraph capture-and-replay.
 *
 * Wraps each iteration of the per-layer decode loop in
 * cudaStreamBeginCapture / EndCapture / GraphInstantiate, then replays the
 * cached executable on subsequent tokens.  CUDA-only; Metal stubs all of
 * these as no-ops (begin returns -1, end is a no-op, enabled returns 0).
 *
 * Opt-in via the DS4_CUDA_LAYER_GRAPHS env var; default OFF for Step 5/6/7,
 * default ON at Step 8 after determinism + perf gates pass.
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
};

/* Returns 1 iff DS4_CUDA_LAYER_GRAPHS is set to an enable value.  Metal
 * stub returns 0.  Callers use this to gate the per-token build_key cost
 * and the R4 split-flush override. */
int  ds4_cuda_layer_graphs_enabled(void);

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

/* Step 7 per-kernel hash dump (DS4_CUDA_LAYER_GRAPHS_HASH_DUMP=1).
 * Used to bisect the captured-vs-eager output divergence by hashing
 * each shim's primary output buffer and printing the per-token table.
 * - dump_hash_reset: clear the slot table; called once per token.
 * - dump_hash_after: launch FNV-1a kernel on `tensor->ptr` for the next
 *                    slot and record `label` for it.
 * - dump_hash_flush: sync, copy hashes to host, print one line per slot
 *                    used this token: "DS4_HASH pos=N slot=I hexhash label".
 * Metal stubs all three as no-ops. */
void ds4_cuda_dump_hash_reset(void);
void ds4_cuda_dump_hash_after(const ds4_gpu_tensor *tensor,
                                uint64_t n_elem,
                                const char *label);
void ds4_cuda_dump_hash_at_slot(const ds4_gpu_tensor *tensor,
                                  uint64_t n_elem,
                                  const char *label,
                                  uint32_t slot);
void ds4_cuda_dump_hash_flush(uint32_t pos);

/* Step 7 narrowing: routed-MoE branch tag + tag-write helper.  Used to
 * record which of the three routed-MoE dispatch paths (1=MMVQ-decode,
 * 2=MMQ-batch, 3=legacy) actually ran for the most recent call, and
 * to record categorical values into a hash dump slot. */
int  ds4_cuda_dump_get_last_moe_branch(void);
void ds4_cuda_dump_set_last_moe_branch(int b);
void ds4_cuda_dump_tag_at_slot(uint32_t tag, const char *label, uint32_t slot);

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);
int ds4_gpu_import_model_ipc_manifest(const void *model_map, uint64_t model_size, const char *manifest_path, const char *model_id);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
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
        uint32_t                il);

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
        float                   scale);

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
        float                   scale);

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

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
        float                   eps);

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);

/* R1 row-variant (Step 4c R1' migration to layer-scalars substrate):
 * writes one row of `base` at the index taken from the per-layer device
 * array g_layer_dev[il].comp_row.  Replaces the (transient view, n_tok=1)
 * form used in the decode-time compressor emit path so the captured
 * kernel-node arg list bakes a stable base pointer + a per-layer baked
 * ls pointer, not a per-token row pointer.  The shim computes the
 * per-layer ls = &g_layer_dev[il] internally.  See plan doc R1 + sec
 * 15.4. */
int ds4_gpu_dsv4_fp8_kv_quantize_row_tensor(
        ds4_gpu_tensor *base,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          il);

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim);

/* R1 row-variant (Step 4c R1'): writes one row of `base` at the index
 * taken from g_layer_dev[il].index_row. */
int ds4_gpu_dsv4_indexer_qat_row_tensor(
        ds4_gpu_tensor *base,
        uint32_t          head_dim,
        uint32_t          il);

int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
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
        uint32_t                head_dim);

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
        int                     row_field);

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
        const void             *scalars);

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
        float                   rms_eps);

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
        float                   rms_eps);

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
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim,
        const void             *scalars,
        /* Step 4c A1: per-layer index for the ds4_layer_scalars substrate.
         * Decode1 passes il (0..42) to lift n_comp off the inline arg;
         * decode2-exact + batch callers pass UINT32_MAX (no substrate). */
        uint32_t                il_for_decode1);

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
        uint32_t                head_dim);

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

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
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        const void             *scalars,
        /* Step 4c A1: per-layer index for ds4_layer_scalars substrate.
         * UINT32_MAX = no substrate. */
        uint32_t                il_for_decode1);

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

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
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        bool                   *mid_is_f16);

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
