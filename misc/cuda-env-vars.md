# CUDA backend env-var reference

The CUDA backend (`ds4_cuda.cu`) mirrors `ds4_metal.m`'s role on NVIDIA. It
dispatches Q8_0 dense matmuls through one of three kernel families, has its
own n_tok=1 mmvq decode path, optionally captures the decode-block kernel
sequence into a `cudaGraphExec_t`, and can either allocate weight memory
in-process (default) or import it from the `ds4_weight_server` sidecar.

Every CUDA-specific env var is below, with the intent behind each default.

## Q8_0 dispatcher

cuBLAS is initialised unconditionally at backend startup regardless of the
selected strategy: on sm_121 we observed that this triggers CUDA driver state
making mmq ~4&times; faster than a binary that skips `cublasCreate`, so the
cublas path stays resident even when not selected.

| Strategy | When picked                                       | What it runs                                       |
|----------|---------------------------------------------------|----------------------------------------------------|
| `mmq`    | default; every CUDA arch we've validated          | vendored llama.cpp `mul_mat_q` (`cuda/mmq/`)       |
| `cublas` | explicit override or fallback if mmq init fails   | `cuda_q8_f16_ptr` Q8&rarr;FP16 cache + `cublasGemmEx` |
| `warp8`  | explicit override or last-resort fallback         | native `matmul_q8_0_preq_*_kernel` family          |

Logged once on first dispatch, e.g.:

    ds4: CUDA Q8_0 dispatch: mmq (sm_120, 1792 GB/s memory bandwidth) [default]

The bandwidth figure is informational; we don't tier on it.

## Env-var inventory

- `DS4_MODEL_ANON_HUGE=N` (Linux, default off; lives in ds4.c model_open, not
  the CUDA backend). Copy GPU-backend model files out of the file-backed mmap
  into anonymous `MADV_HUGEPAGE` memory at load. `N<=1` copies every GPU
  model; `N>1` copies only files of at least `N` GiB (e.g. `32` = base model
  only). WHY: the mmap'd GGUF leaves weights on 4 KB page-cache folios whose
  physical contiguity depends on how they entered the cache; on GB10 the
  GPU-side cost is decided at `cudaHostRegister` pin time — fragmented folios
  cost ~5x on routed-MoE spans, and even sequentially-rewarmed cache pays
  2-3.6x vs anon-THP (measured 2026-07-02: base decode 86-96 &rarr; 68.5
  ms/tok, w1 MoE span 0.276 ms = isolated-kernel floor). CAUTION: anon memory
  is unevictable — only enable when total copied bytes leave ~10 GB of true
  free RAM after banks/runtime, else the kernel splits the huge pages under
  reclaim pressure and performance lands BELOW plain mmap (measured: the
  95 GB base+MTP+drafter set on a 121 GB GB10). The copy streams with
  `posix_fadvise(DONTNEED)` so it does not compete with itself for RAM; boot
  cost is one sequential read of the file (~25 s for 81 GB on GB10 NVMe).
  A pressure guard skips the copy (keeping the file mapping, with a log line)
  when it would leave less than `DS4_MODEL_ANON_HUGE_MARGIN_GB` (default 20)
  of MemAvailable — the margin must also cover KV banks and runtime, and
  ctx-scaled bank growth at 32k+ can still exceed it.

- `DS4_MODEL_ANON_HUGE_MARGIN_GB=N` (default 20). Minimum GiB of MemAvailable
  that must remain after an anon huge-page model copy; below it the copy is
  skipped. See `DS4_MODEL_ANON_HUGE`.

- `DS4_WEIGHT_RESIDENCY=eager|lazy|mapped` (memgov D3-2; per-source
  overrides `DS4_WEIGHT_RESIDENCY_BASE` / `_MTP` / `_DRAFTER` win over the
  global). ONE typed weight-residency policy per model source, resolved
  once at engine open: `eager` (default) device-promotes the funded
  non-expert plan at boot with lazy backfill on misses; `lazy` defers the
  same promotions to first touch; `mapped` is TERMINAL host residency —
  mapped units never device-promote, and on integrated (GB10) the raw
  ATS-coherent mmap pointer serves them directly (capture-safe, no
  per-range registrations). Setting it beside a legacy residency lever
  (`DS4_CUDA_NO_HBM_CACHE` / `DS4_CUDA_NO_FD_CACHE` /
  `DS4_CUDA_DIRECT_MODEL`) is a strict boot error; with it unset the
  legacy shapes act as exact aliases (none = eager, `NO_HBM_CACHE` =
  lazy, `NO_HBM_CACHE`+`NO_FD_CACHE` = mapped, `DIRECT_MODEL` = mapped).
  The boot log prints the resolution:
  `ds4: weight residency: base=... mtp=... drafter=... (explicit|legacy-alias)`.

- HISTORY: `-DDS4_CUDA_SPARK_HBM_CACHE=1` (compile-time fork) is RETIRED
  (deepmem lite-2; plumbing deleted in D3-3). Startup promotion is
  compiled unconditionally, runtime-gated, and driven by the canonical
  unit materializer (the old `accelerator_cache_model_tensor_spans` walk
  is deleted). The substrate numbers that motivated device residency
  (measured 2026-07-02, per-stream): GPU streaming reads of
  `cudaHostRegister`'d host memory cap at ~161 GB/s while `cudaMalloc`
  device memory reads ~236 GB/s on the same LPDDR5x; base decode 63.3
  ms/tok with the walk vs 73-74 without. The whole-system re-measurement
  under serving load is the D3-1 rent gate (speed-bench/rent_gate.sh),
  which selects the release default. Promotion budget cap stays
  `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB` (default 24).

- Range-lookup fix note (no env var; 2026-07-02): `cuda_model_range_ptr` now
  resolves device-resident HBM cache copies in a first pass before the
  whole-model registered mapping. Previously a single ordered scan returned
  the mapped (slow) pointer for every offset interior to a cached span —
  the registered whole-model range is pushed first — so only tensors that
  happened to start a merged span (in practice just the output head) ever hit
  the cache. This is why the HBM cache historically looked like "~10%".

- `DS4_CUDA_STAGE_PROF_LITE=1` (default off). Lazy-event GPU-span measurement
  of whole decode-forward stages (fwd/embed/attn/attncore/idxscore/cupd/emit/
  rbcp/ffn/moe/head), the MOE_PROF_LITE discipline generalized: event pairs on
  the current stream, non-blocking harvest on ring wrap, two stderr lines
  every 8192 harvests (`ms/call` per stage and `ms/fwd` = per-decode-step
  attribution). Valid in async production; nested stages overlap by design
  (attn ⊃ attncore/idxscore/emit ⊃ cupd/rbcp; ffn ⊃ moe) — subtract when
  attributing. Σ(fwd+embed+head) ≈ wall ms/tok when healthy; a gap means an
  unbracketed stage or launch bubbles.

- Budgeting note (no env var; behavior change 2026-07-02): on integrated GPUs
  the KV-bank budget (`ds4_gpu_mem_info`) now subtracts GPU-pinned
  *file-backed* weight registrations from MemAvailable. The kernel keeps
  counting pinned page-cache pages as reclaimable, so the old budget spent
  the model's own residency on banks — the measured NVMe-thrash mechanism at
  32 banks. Default bank counts on GB10 drop accordingly (the honest number);
  `DS4_SERVER_COALESCE_MAX` still caps explicitly.

- `DS4_BATCH_FIT_HEADROOM_MB=N`. Bytes the bank-count fit and the VMM
  comp-page budget leave free for runtime growth (tmp pools, capture graphs,
  logits staging). Derived since v0.6.2 as the operator floor
  (`DS4_MEM_FLOOR_GB`, default 3 since v0.7.0) plus the boot-burst constant
  (`DS4_BATCH_FIT_BURST_MB`, default 2048); since v0.7.0 boots with `-c` at or
  below `DS4_BATCH_FIT_SERIAL_CTX_MAX` (default 131072) add a serial hold-out
  (`DS4_BATCH_FIT_SERIAL_MB`, default 2048, 0 = off) so the serial lane's
  session graph stays fundable beside a full bank pool at shallow shapes; at
  deeper boots the term is off (both lanes cannot be funded at depth, banks
  win). Setting `DS4_BATCH_FIT_HEADROOM_MB` pins one value outright;
  `DS4_BATCH_FIT_HEADROOM_DERIVED=0` restores the old static 6144. The boot
  line `batch fit headroom: ...` discloses the derivation.
- `DS4_SERVER_SERIAL_RIGHTSIZE_QUOTE=0`. Restores the physical-only serial
  right-size search. Since v0.7.0 the search also asks the memory governor,
  so a serial request whose full-context graph fits physically but is
  refused on the governor's margin gets the largest graph the governor
  admits instead of a 503 (the log line `serial right-size: governor-aware
  shrink ctx A -> B`).

- Serving-path note (2026-07-03): `DS4_SERVER_CONTINUOUS=0` routes requests
  down the original graph-captured serial session path — on GB10 short-ctx
  this decodes ~11% faster at N=1 (56.5 vs 63.3 ms/tok) because the eager
  continuous-batch width-1 forward carries ~7 ms/tok of un-captured
  per-step overhead. It serializes the GPU per request, so it is a
  dedicated single-user setting only; the continuous default is correct
  for any concurrent serving. (This also explains the historical
  `DS4_CUDA_Q8_F16_PRELOAD` "win": the F16 copies starved the bank budget,
  continuous admission failed, and requests silently fell back to the
  serial path — the preload itself contributes nothing and is not a
  recommended knob.)

- `DS4_SERVER_DEFAULT_TEMP=<float>` (2026-07-13): temperature applied to
  requests that do not send one (default 1.0). Agent frameworks routinely
  omit temperature, and tool-calling requests are batchable greedy-only
  (`job_is_batchable`), so the 1.0 default silently routes them down the
  serial path — no continuous batching, no DSpark. `DS4_SERVER_DEFAULT_TEMP=0`
  makes omitted-temperature traffic greedy and therefore fast-path eligible.
  Requests that DO send a temperature are untouched. Pair with the
  `deepseek-chat` model alias (disables think mode) for agent clients:
  thinking also forces tools serial.

- `DS4_CUDA_PREFILL_PATH=mmq|cublas|warp8|auto` (default `auto` &rarr; mmq).
  Explicit override. `auto` and unset both resolve to mmq.

- `DS4_CUDA_USE_MMQ=0` (legacy alias). Equivalent to
  `DS4_CUDA_PREFILL_PATH=cublas`. The newer variable takes precedence.

- `DS4_CUDA_MMQ_MOE_MIN_TOKENS=N` (default 2). Minimum `n_tokens` at which
  the routed-MoE mmq path activates. At n=1 mmq's matrix-matrix-shaped path
  has higher per-launch cost than the vector path; that case is handled by
  the mmvq decode branch.

- `DS4_CUDA_MOE_SMALL16=N` (default 16; set 0 to disable). Direct routed-MoE
  vector tier for MTP/DSpark verifier widths 9..16 with top-k=6. It bypasses
  the MMQ tile path for these small batches and reuses the CUDA MMVQ/Q8_1
  vector machinery with internal column chunking above width 8.
  `N` may be 9..16 to cap coverage; `DS4_CUDA_MOE_NO_SMALL16=1` is an alias
  for disable. Since 2026-07-03 the tier applies to ALL models by default
  (`DS4_CUDA_MOE_SMALL16_ALL=0` restores the old MTP/verifier + small-Q4_K
  scope): base continuous-batch decode at widths 9..16 previously fell into
  the sorted/MMQ machinery and cliffed hard at n_live=9 (GB10 N=12 agg
  26.2 -> 41.3 tok/s with the tier; widths <=8 unaffected; quality-gated
  within score band on gsm8k/HumanEval/MBPP at conc 12). The larger Q4_K
  DSpark drafter still keeps its MMQ numerics unless SMALL16_ALL is set
  explicitly; `DS4_CUDA_MOE_NO_SMALL16_MTP_DRAFT=1` restricts the legacy
  scope to verifier forwards only. Widths 17..23 remain a known dead zone
  (fall back to MMQ, N=20 agg ~26 vs N=16 ~51 on GB10) — prefer scheduling
  decode widths <=16 or >=24. `DS4_CUDA_MOE_SMALL16_DIRECT=1` selects the experimental
  Q8_K direct fallback instead of preserving the normal MMQ backup path.
  `DS4_CUDA_MOE_Q81_FUSED=1` selects experimental canonical-Q8_1 fused
  gate+up+mid and down+sum helpers; early GB10 probes preserved acceptance but
  regressed throughput at wider routed batches, so this is diagnostic-only.
  `DS4_CUDA_MOE_Q8K_GATE_IN_VEC=1` and `DS4_CUDA_MOE_Q8K_DOWN_IN_VEC=1`
  are stage-isolation diagnostics: they keep the MMVQ branch active but swap
  only gate/up or only down to the Q8_K direct kernels.
  `DS4_CUDA_MOE_PAIR_RAW_VEC=1` selects an exact gate/up MMVQ diagnostic that
  quantizes X to Q8_1 once, runs the trusted MMVQ gate and up matvecs from that
  shared buffer, and leaves the existing clamp-aware SwiGLU kernel in place.
  `DS4_CUDA_MOE_PROFILE=1` prints per-call MoE stage timings for the fallback
  direct path and, in eager mode, for this MMVQ small16 path. For MMVQ timing
  runs, set `DS4_CUDA_LAYER_GRAPHS=0 DS4_CUDA_MOE_GRAPHS=0` so event recording
  is not inside a captured CUDA graph.

- `DS4_CUDA_MOE_NO_IQ2_ALIGNED=1`. Kill switch for the aligned-SoA IQ2_XXS
  decode path (megakernel program M1-Inc1). The path activates automatically
  at decode widths 1..16 (`136cec7` extended it past `n_tokens=1`) when the
  weight server serves repacked artifacts — DEFAULT ON since the 2026-07-04
  quality gate (gsm8k 97.4 / HumanEval 92.1 / MBPP 89.0 vs 97.6/88.4/90.0;
  disable server-side with `--no-repack-iq2-aligned`). The historical
  opt-in flag `--repack-iq2-aligned` re-serves the routed-expert gate/up stacks in
  a byte-neutral 64B-aligned layout (`[dq][pad][qs]`, replacing their raw
  ranges) and lifts the vec-tier gate/up rate from ~142 GB/s (66-byte block
  stride, 2x16-bit code loads) to ~215+ GB/s. With repacked artifacts
  present, raw-layout consumers (prefill MMQ tiles, widths 2..16) read the
  gate/up ranges directly from the client mmap through HMM instead of the
  device copy — correct, prefill-amortized, but slower per byte; do not
  combine a repack server with `DS4_CUDA_NO_DERIVED_WEIGHTS=1` (that hides
  the artifacts AND the raw exclusion remains, forcing every consumer onto
  the mmap substrate). Parity gate: `cuda/mmq/test/test_iq2_aligned_entry.cu`.

- `DS4_CUDA_MOE_NO_IQ2_DEREPACK=1`. Kill switch for the M1-Inc2b raw-layout
  device scratch. With repacked artifacts present, the prefill MMQ tile path
  no longer reads the excluded gate/up ranges from the client mmap; it
  inverts the repack device->device (`ds4_mmq_iq2_xxs_aligned_derepack`)
  into two persistent ~528 MiB scratch buffers, refilled per layer
  (~4.8 ms/tensor at ~233 GB/s), byte-exact. A/B on GB10, 15.2k-token
  prefill: 50-55 s vs 391-394 s on the mmap fallback (7.9x). One-shot boot
  log: `iq2 derepack scratch active`. Disabling restores the mmap-raw
  behavior above; decode (`n_tokens=1`) is unaffected either way.

- `DS4_CUDA_MOE_NO_Q2K_ALIGNED=1`. Kill switch for the row-pair-SoA Q2_K
  moe-down decode path (megakernel program M2). The path activates
  automatically at the vec-tier down leg when the weight server serves the
  repacked artifact (`--repack-q2k-aligned`, opt-in, REPLACES the raw
  `.ffn_down_exps` ranges at byte parity). Outputs are bit-identical to the
  raw `mul_mat_vec_q_moe<Q2_K,2>` path (proto_m2_q2k.cu: 240/240 parity +
  graph capture/replay); the twin reads at ~214 GB/s vs ~154 raw on the proto
  rig. One-shot boot log: `M2 Q2K aligned moe-down active`. CAUTION: with a
  repacked manifest the raw range is not served, so this switch drops the vec
  tier to the client-mmap raw path (~100x) — the real off switch is
  `--no-repack-q2k-aligned` on the weight server. The WS flag is DEFAULT ON
  since the 2026-07-05 quality gate (gsm8k 97.2 / HumanEval 91.5 / MBPP 88.5,
  within boot noise of the iq2/q8-flip 97.4/92.1/89.0; the twin is
  bit-identical). Needs clients >= `e221241`.

- `DS4_CUDA_MOE_NO_Q2K_DEREPACK=1`. Kill switch for the moe-down raw-layout
  device scratch (same contract as the IQ2 one above): with the q2k artifact
  present, the prefill MMQ down path inverts the repack device->device
  (`ds4_mmq_q2_K_aligned_derepack`) into one persistent ~672 MiB scratch,
  refilled per layer, byte-exact. One-shot boot log:
  `q2k derepack scratch active`. Disabling restores the mmap-raw prefill
  behavior; the vec-tier decode path is unaffected either way.

- `DS4_CUDA_NO_Q8_ALIGNED=1`. Kill switch for ALL aligned-SoA Q8_0 decode
  paths: the dense mmvq site (M1-Inc3) and the three custom warp8 kernels
  (M1-Inc4: hc_expand, q8 pair, grouped output_a). They activate
  automatically at decode when the weight server serves the q8 artifacts —
  DEFAULT ON since the 2026-07-04 quality gate; disable server-side with
  `--no-repack-q8-aligned` (345 artifacts, ~6.2 GiB, ADDITIVE — raw stays
  served, so every fallback is still a device read; one-shot boot logs
  `dense decode using aligned Q8_0 artifacts` plus three `M1-Inc4` lines).
  Inc3 alone was ~0.4 ms/tok (the dense mmvq site covers only a small span);
  Inc4 extends the layout to the warp8 kernels (21.2 ms/step of the serial
  decode) — proto_q8_warp8.cu shows +7-13% per kernel with bit-identical
  accumulation, and the combined e2e decode-only win is ~1.4 ms/tok on GB10
  (.33: 53.1 -> 51.7). With Inc4 the artifacts are worth serving for serial
  N=1 production, hence the default flip. CAVEAT: high-N continuous serving
  (decode widths >16, the MMQ tile path) still reads raw iq2 from mmap or
  pays derepack refills on a repacked manifest — keep a raw manifest
  (`--no-repack-iq2-aligned`) for that regime until the gap is closed.

- `DS4_CUDA_NO_HC_STAGE_FUSED=1`. Kill switch for the M2-Inc1 fused decode
  HC stage. Default ON when preconditions hold (Flash n_hc=4, F16 hc fn
  projections, decode single row, cooperative launch available): one
  48-block cooperative kernel with three `grid.sync()`s replaces the
  four-launch latency-floor chain `rms_norm_plain` -> f16 splitk matmul
  (16384x24) -> splitk combine -> `hc_split_weighted_sum_norm`, twice per
  layer (~40 us -> ~13.5 us per chain, proto_m2_hc.cu; ~-2.3 ms/tok
  projected on GB10 serial decode). Runs the dot products on the RAW hc
  input and applies the rms scale at combine time (mathematically exact,
  not bit-identical to the unfused chain — double-ref parity in-family
  with baseline). One-shot boot log: `M2-Inc1 fused HC stage active`.
  Disabled implicitly under `DS4_METAL_DECODE_STAGE_PROFILE` so stage
  spans keep their boundaries. Cooperative launches are captured into the
  per-layer cudaGraphs and replay bit-identically (proto-proven on GB10 /
  CUDA 13).

- `DS4_CUDA_NO_HC_Q8_FOLD=1`. Kill switch for ALL producer-emitted q8
  activation folds. M2-Inc1b: the cooperative HC-stage kernel's final phase
  emits the q8_0 codes of `attn_norm`/`ffn_norm` (bit-exact vs
  `quantize_q8_0_f32_kernel` — same butterfly reductions and rounding,
  proto_m2_hc.cu V4), and the next q8_0 pair consumer (attn q_a+kv, shexp
  gate+up) takes them instead of launching its quantize prelude — 2 fewer
  launches/layer. M2-Inc2a extends the registry with a q8_1 flavor
  (canonical block_q8_1, bit-exact vs the vendored `quantize_q8_1`): the HC
  stage emits ffn_norm's q8_1 blocks for the routed-MoE mmvq consumer and
  the qkv-rms kernel emits qr_norm's for the q_b consumer (hooks in
  ds4_mmq.cu skip `quantize_row_q8_1_cuda` on a registry hit) — 2 more
  launches/layer. Registry is encode-scoped, pointer+length keyed,
  pop-on-lookup per format, reset at every fused-HC-entry call. One-shot
  boot logs: `M2-Inc1b HC-stage q8 activation fold active (pair decode)` and
  `M2-Inc2a q8_1 activation fold active (mmvq decode)`. The q8_0 fold is
  implicitly off whenever the fused HC stage is off; the q8_1 folds are
  additionally off under `DS4_CUDA_NO_QKV_POST_FUSED`.

- `DS4_CUDA_NO_QKV_POST_FUSED=1`. Kill switch for the M2-Inc2 fused
  QKV-post kernels on the serial decode path: (2b) `head_rms_norm` + q
  `rope_tail` as one per-head kernel (device-scalars pos — the capture-safe
  twin of the long-dead host-pos fused kernel), and (2c) kv `rope_tail` +
  FP8 KV quantize + raw-cache store as ONE 8-block kernel (rope only
  touches the 64-float rotary tail, fp8 only the 448 nope elems — disjoint;
  the production fp8 kernel's 7 sequential 64-elem groups become 7 parallel
  blocks, bit-exact scales). Both fusions bit-exact vs their unfused chains
  (proto_m2_qkv.cu, plain+yarn rope, multiple positions, f16-round-trip
  store included). Also implicitly off when `DS4_METAL_GRAPH_DUMP_PREFIX`
  is set (the fused paths lose the Qnorm/KVrope intermediate dump points)
  and in reference-kv mode.

- `DS4_CUDA_NO_ROUTER_FUSED=1`. Kill switch for the M2-Inc3 fused decode
  router stage: the f16 logits matmul (4096x256), split-K combine, and top-6
  select run as ONE cooperative kernel (`router_fused_coop_kernel`, up to 128
  blocks; select-side bias/hash model-map reads prefetched behind the
  matmul), replacing three launches per MoE layer. Bit-exact vs the unfused
  chain (proto_m2_router.cu: 240/240 across bias/no-bias/hash, exact-tie and
  all-equal-logits cases, capture==eager). One-shot boot log: `M2-Inc3 fused
  router active (coop N-blk)`. Also implicitly off under any legacy
  router/f16 dispatch knob (`DS4_CUDA_SERIAL_F16_MATMUL`,
  `DS4_CUDA_SERIAL_ROUTER`, `DS4_CUDA_ORDERED_F16_MATMUL`,
  `DS4_CUDA_NO_WARP_ROUTER_SELECT`, `DS4_CUDA_NO_PARALLEL_ROUTER_SELECT`) so
  each of those keeps selecting the kernel it names.

- `DS4_CUDA_NO_COMP_PAIR_FUSED=1`. Kill switch for the M2-Inc5 fused decode
  compressor event: both pair f16 matmuls (kv + gate, 4096 x width), their
  split-K combines, and the compressor store run as ONE cooperative kernel
  (`comp_pair_store_fused_kernel<KS>`, up to 128 blocks), replacing five
  launches per compressor event at the three decode widths (1024 = primary
  ratio-4, 512 = primary ratio-2, 256 = indexer ratio-4).  On the fused path
  the emit tail runs via `ds4_gpu_compressor_update_tail_tensor` (the full
  update would double-store).  Bit-exact vs the unfused chain
  (proto_m2_comp.cu: 324/324 across widths x ape f16/f32 x pos sweep,
  capture==eager).  One-shot boot log: `M2-Inc5 fused compressor pair+store
  active (coop N-blk)`.  Also implicitly off under `DS4_CUDA_SERIAL_F16_MATMUL`
  and `DS4_CUDA_ORDERED_F16_MATMUL` (each keeps selecting the kernel it
  names).

- `DS4_CUDA_MMQ_X_MAX=N`. Clip `get_mmq_x_max_host` to N (rounded down to a
  multiple of 8) when sweeping tile widths. Diagnostic only; the vanilla
  128 wins on sm_120.

- `DS4_CUDA_NO_MMVQ_DECODE`. Opt-out of the vendored `mul_mat_vec_q` decode
  path. mmvq is structurally optimal for n_tok=1 routed-MoE and dense
  attention projection (one block per output row, no column-tile waste).
  Wires into `routed_moe_launch` and `cuda_matmul_q8_0_tensor_labeled`.

- `DS4_CUDA_MMVQ_DECODE_MAX_TOKENS=N` (default 8). Cap on n_tokens routed
  through the mmvq decode branch in `routed_moe_launch`. Range 0&ndash;8;
  0 disables. Values 2&ndash;8 extend mmvq coverage to short-prefill
  batches, subject to the `DS4_CUDA_MOE_VEC_MAX_ASSIGN` assignment envelope.

- `DS4_CUDA_MOE_GRAPHS=0` (default on). Opt-out of CUDA Graph
  capture+replay around the mmvq routed-MoE decode block and the n_tok=1
  dense Q8_0 vec path. Each captured launch is bracketed by
  `cudaEventRecord` / `cudaStreamWaitEvent` so g_moe_stream and stream=0
  stay correctly ordered across the boundary.

- `DS4_CUDA_LAYER_GRAPHS=0` (default on). Opt-out of per-layer
  decode-body CUDA Graph capture+replay. On by default since the Step 7
  determinism + perf gates passed: each transformer layer's decode body
  is captured into its own `cudaGraphExec_t`, keyed on layer index /
  token-flags / double-buffer parity, and replayed on subsequent
  matching tokens. Per-token state rides device-resident scalar
  substrates so it never enters the graph key. Verified bit-identical
  to eager decode through n=256 on sm_120 (PRO 6000) and sm_121 (GB10);
  decode-only, prefill is untouched. Set to 0 (also `off`/`no`/`false`)
  to fall back to the eager per-layer decode path. Also forced off when
  `DS4_CUDA_NO_MMVQ_DECODE` is set (the legacy non-MMVQ decode path is not
  capture-safe).

- `DS4_CUDA_LAYER_GRAPHS_HASH_DUMP=1` (default off). Arms the
  captured-decode per-kernel hash-dump diagnostic. When set, the
  `ds4_cuda_dump_hash_*` entry points FNV-1a a probed device buffer into a
  slot table and print one `DS4_HASH pos=N slot=I hexhash label` line per
  used slot at each token flush; when unset every entry point is a no-op,
  so a normal build is unaffected. Used to localize a
  captured-graph-vs-eager output divergence: probe the same prompt with
  and without `DS4_CUDA_LAYER_GRAPHS=0` and diff the `DS4_HASH` lines — the
  first `(pos,slot)` that differs is the divergent kernel. The probe call
  sites are added temporarily by the investigator (see the comment block
  above the implementation in `ds4_cuda.cu`); only the substrate is
  permanent. See also `tests/cuda_layer_graph_determinism_probe.sh`.

- `DS4_CUDA_MTP_VERIFIER_USE_MMQ` (default 0). Bisection switch. Normally
  `ds4.c` brackets every MTP verifier call with
  `ds4_gpu_set_mtp_verifier(1/0)` and the CUDA backend routes Q8_0
  matmuls onto `warp8` for the duration. mmq's stream-k + MMA FP32
  reduction order drifts ~1 ULP/layer from warp8; the drafter is trained
  against legacy decoding so an mmq verifier flips tight-margin tokens
  (0/314 acceptance on GB10 with mmq verifier active). Set to 1 to
  reproduce the broken behavior for bisection.

## DSpark / DFlash diagnostics

- `DS4_DSPARK_MAX_KV=N` (default 65536; 0 disables the gate). Production
  kv-depth auto-gate. Speculative decoding's advantage decays with context
  depth: acceptance drops as the sequence deepens while the multi-row verify
  forward's attention cost grows with kv. A bank whose
  kv frontier reaches `N` stops packing draft rows (verify = 1 row, plain
  decode width), is excluded from the block draft, and logs a one-shot
  `cont-dspark kv-gate` line; when every live bank is gated the step degrades
  to plain batched decode (no rollback capture, no draft, no ring injection),
  the same lossless path as `DS4_DSPARK_MAX_NLIVE`. The gate is one-way per
  request (positions only grow) and cannot affect output: the target verify
  forward remains the sole source of committed tokens. The default comes from
  the 2026-07-11 default-decision probes: spec still wins at 49k depth on both
  prose (1.10-1.49x) and code (1.19x), while the loss regime starts at ~64k+
  (prose 0.90x, 0.75x at 64-112k in the release frontier sweep; code reaches
  breakeven 1.02x at 65k). Structured content (code, math) keeps higher
  acceptance and crosses over latest, so raise `N` for code-heavy serving, or
  set 0 to always speculate. `DSPARK_PROFILE` reports `kv_gate_steps` (fully
  gated steps) and `kv_gate_saved` (draft rows suppressed by the per-bank gate
  on mixed steps).

- `DS4_DSPARK_ADAPT_GATE=1` (default off, experimental). Replaces the static
  `DS4_DSPARK_MAX_KV` cutoff with a runtime measure-and-switch controller.
  The static threshold is calibrated on prose, the weakest-acceptance content;
  structured/agentic content keeps acceptance deeper, so a fixed cutoff forgoes
  real wins there. Past `DS4_DSPARK_ADAPT_START` (default = `DS4_DSPARK_MAX_KV`)
  the solo-stream decode loop times a `DS4_DSPARK_ADAPT_RUN`-token window
  (default 512) in the settled mode, probes the other mode for
  `DS4_DSPARK_ADAPT_WIN` tokens (default 64), and keeps whichever decodes
  faster with `DS4_DSPARK_ADAPT_MARGIN_PCT` (default 3) hysteresis, re-probing
  every cycle so content shifts mid-generation re-open the decision. Below the
  start depth it is dormant (pure spec, zero overhead). Engages only at
  `n_live==1` (the production `DS4_DSPARK_MAX_NLIVE=1` regime). Unlike the
  static gate, spec can re-enter, so ring injection and the capture tap stay on
  during spec-off windows (re-entry across ring holes collapses acceptance);
  the hard gate and its injection skip are inert while this mode is on.
  Lossless either way — the verify forward remains the sole token source.
  Logs `adapt-gate ENGAGED` once per request and each mode switch (first 8);
  `DSPARK_PROFILE` reports `ad_probes`/`ad_switches`/`ad_plain_steps`.

- `DS4_DSPARK_QUENCH` (default ON; `=0` disables). Terminal per-request yield
  quench: each verify step accumulates cumulative regret
  `debt += guard − tokens_committed` per bank (no clamp — surplus from good
  steps banks credit); once `spec_steps ≥ minev`, `yield_EWMA < guard`, and
  `debt > budget`, speculation turns off for the rest of that request via the
  kv-gate's lossless per-bank nd=0 path (reset at admit). Bounds low-accept
  requests at ~0.96× plain (always-spec floor was 0.72× on deep prose) while
  leaving winners untouched. Calibrated defaults guard 2.22 / alpha 0.125 /
  minev 4 / budget 4.0 / credit cap ∞, overridable via
  `DS4_DSPARK_SHADOW_{GUARD,ALPHA,MINEV,BUDGET,CREDIT_CAP}` (shared with the
  trace shadow). Supersedes `DS4_DSPARK_ADAPT_GATE` when both are set. Logs
  `yield-quench bank=... -> spec off for this seq (terminal)`; `DSPARK_PROFILE`
  reports `quench_steps`/`quench_saved`.

- `DS4_DSPARK_QUENCH_FORCE_STEP=N` (testing). With quench on, force-quench
  every bank at its Nth spec step regardless of yield (0 = at admit — the
  whole request runs the terminal-plain path; used for the plain-identity
  gate). Disables the policy trigger.

- `DS4_DSPARK_TRACE=1` (default off). Per-request per-step telemetry:
  `DSPARK_TRACE` (yield/comparisons/packed-drafts/n_live/spec/step-ms records)
  and `DSPARK_SHADOW` (the quench arithmetic replayed over the request).
  Feeds `tools/dspark_trace_replay.py` (`validate` proves trace totals equal
  `CONT_MTP_ACCEPT` exactly; `replay --grid` calibrates quench parameters
  offline; `inspect`; `selftest`).

- `DS4_DSPARK_PROFILE=1` (default off). Print aggregate DSpark continuous
  decode timings split into verifier forward, accept loop, inject pack,
  inject projection, inject store, rollback, deferred commit, block draft,
  and Markov refine. Use with `DS4_CONT_PROFILE=1` when comparing against
  whole-engine forward/sample buckets.

- `DS4_DSPARK_COMPACT_INJECT=1` (default off). Diagnostic path that injects
  only accepted verifier rows into the DSpark KV rings instead of every
  verifier row. It packs captured target hidden rows with a CUDA gather
  kernel before the existing DSpark projection/inject stages. Intended to
  quantify whether rejected-row injection is material.

- `DS4_DSPARK_VERIFY_DEPTH=N` (range `1..4`). Diagnostic speed/acceptance
  dial for DSpark block verify width. Unset uses the production policy:
  verify all four drafts at `n_live<=2`, verify three drafts at `n_live==3`,
  and let the existing `DS4_DSPARK_MAX_NLIVE` gate disable DSpark above that.
  When set, the value is exact and disables the `n_live==3` auto-depth rule.
  The block drafter still generates four candidate drafts; this dial controls
  how many are consumed by the verifier on the next step.

- `DS4_DSPARK_ADAPT_DEPTH=1` (default off). Diagnostic per-bank verify-depth
  controller. Each live bank shrinks its next verifier width after a miss and
  grows it after accepting the full currently verified prefix. Correctness is
  unchanged because accepted tokens still come only from the target verifier;
  this only trades verifier rows against draft yield.

## In-process VMM weight arena

The arena allocates each weight tensor in its own CUDA Driver VMM
region (`cuMemCreate` &rarr; `cuMemAddressReserve` &rarr; `cuMemMap`
&rarr; `cuMemSetAccess`), giving every tensor its own
2&nbsp;MiB-aligned virtual address.  This matches what the
out-of-process `ds4_weight_server` provides imported workers.  On
discrete GPUs this is worth roughly 2&times; prefill; on integrated
GPUs it's neutral-to-positive.

### Why per-tensor chunks specifically

The chunk-size bisect we ran during development clarified the
mechanism.  VMM with one large chunk (e.g.
`DS4_CUDA_VMM_ARENA_CHUNK_MB=1792`) performs identically to the
cudaMalloc-backed arena (~1080 t/s prefill on PRO 6000), even though
the underlying memory is still 2&nbsp;MiB-paged.  The actual
differentiator is **per-tensor 2&nbsp;MiB-aligned base addresses**:
when each weight tensor sits at its own fresh
`cuMemAddressReserve`-handed VA, matmul kernels' tile-load coalescing
and L2 spatial-locality patterns improve enough to roughly double
prefill.  Pack the same VMM-paged memory into one big chunk and the
bases land at sub-granularity offsets &mdash; the perf advantage
disappears.

This also unifies cleanly with the drift below: same root cause, two
effects you cannot separate.

### Known trade-off: FP32 reduction-order drift vs official vectors

Per-tensor VMM-allocated weight ranges produce a small but real
**reduction-order drift** in the matmul kernels relative to the
cudaMalloc-backed arena.  The same cache/tile-arrival-order behavior
that gives the 2&times; perf win also changes the order in which tile
partial sums reach the FP32 accumulator; FP32 is non-associative, so
the order matters.  This is structural to the kernels' parallel
reduction strategy, not a misuse of the API.

Investigation established:

1. The uploaded weight bytes are byte-identical between the two
   allocators (verified by post-upload checksum of all 138 weight
   ranges).
2. Kernels do not read past tensor bounds (verified by poisoning the
   chunk tail with 0xAB instead of zero &mdash; output unchanged).
3. The drift is shared by both the vendored mmq family and the legacy
   `warp8` native kernels and is therefore upstream of the Q8_0
   dispatcher.  Same drift on PRO 6000 sm_120 and GB10 sm_121.
4. Logit-level magnitude is small (~0.08 logprob units at step 0)
   &mdash; bounded, deterministic, of the same shape as the documented
   mmq-vs-warp8 ULP-per-layer drift behind `DS4_CUDA_MTP_VERIFIER_USE_MMQ`
   (Option D).  Most tokens are unaffected; only tight-margin choices
   flip.

**Observable cost:** in `./ds4_test --logprob-vectors`, one of four
test vectors (`short_code_completion`, step 1: the `c` language tag
after triple-backticks) flips to a textually-equivalent but
byte-different alternative under the VMM-arena default.  The other
seven failures in that test family are pre-existing on the CUDA
backend and reproduce identically with `DS4_CUDA_VMM_ARENA=0`.

**Workaround for users who need official-vector byte equivalence:**
set `DS4_CUDA_VMM_ARENA=0` to use the cudaMalloc-backed arena.  Prefill
ceiling drops by ~50% on discrete GPUs in exchange for the parity.

### Env vars

- `DS4_CUDA_VMM_ARENA=0`. Disable; fall back to the cudaMalloc-backed
  arena.  Also the workaround for the reduction-order drift above.

- `DS4_CUDA_VMM_ARENA_CHUNK_MB=N`. Minimum chunk size per `cuMemCreate`.
  Default 0 (chunk = request size, rounded up to the driver-reported
  granularity; matches the weight server's per-range allocation).
  Values 1024+ collapse the per-tensor placement and forfeit the perf
  benefit; useful only for bisection.

- `DS4_CUDA_WEIGHT_IPC_MANIFEST=/path/to/manifest.json`. Worker-side
  import path for weights owned by `ds4_weight_server`. When set, the
  in-process VMM arena is hard-gated off because the sidecar already
  provides identical VMM ranges and running both would double-allocate
  the model. See `misc/proof-harness/README.md` for the sidecar
  lifecycle.
