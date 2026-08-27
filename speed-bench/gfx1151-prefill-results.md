# gfx1151 DeepSeek V4 Flash prefill results

## Scope

- Host: `fw2`, AMD Strix Halo `gfx1151`
- Runtime: ROCm 7.14
- Engine base: DS4 `main` at `c1d4597a80e300b803dc642519718f2c999589da`
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- Prompt: `speed-bench/promessi_sposi.txt`
- Goal: 300 tokens/s warm prefill

## Current result

The validated experimental stack reaches 278.42 tokens/s in a warm 4K run
after compacting the IQ2 MoE expert-tile launch, versus 277.29 with only the
tightened launch bound, 264.62 and 269.02 before that change, 263.45-263.65
before the small-M F16 specialization,
256.12-256.31 before the tiny-M F16 specialization, 248.56 before vectorized
indexed-attention KV staging, 227.32 before the attention-output-B
specialization, 187.26 for clean DS4, and 190.66 for the accepted Q2_K
down-only change. The best repeat is 43.7% faster than clean DS4. The remaining
gap to 300 tokens/s is 7.8% in throughput, or about 7.2% less interval time
from the current stack.

Required experimental switches:

```sh
DS4_ROCM_MMQ_IQ2=1
DS4_CUDA_MMQ_X_MAX=64
DS4_ROCM_ATTN_WMMA32_INDEXED=1
DS4_ROCM_ATTN_WMMA32_RING=1
DS4_ROCM_ATTN_OUTPUT_B_WMMA=1
DS4_ROCM_ATTN_F32_VEC2=1
DS4_ROCM_F16_TINYM_WMMA=1
DS4_ROCM_F16_SMALLM_WMMA=1
DS4_ROCM_MMQ_TIGHT_NCOLS=1
DS4_ROCM_MMQ_COMPACT_TILES=1
```

The implementation is split into reviewable commits:

| Commit | Change |
|---|---|
| `c9c231a` | Reuse the Q2_K down-kernel LDS output tile. |
| `d2bd27a` | Add the standalone gfx1151 MMQ kernel lab. |
| `f7b900c` | Route the production IQ2 gate/up pair through the tuned x64/y64 MMQ path and restore the F16 down handoff. |
| `5474132` | Use 128-row/eight-wave Q8 WMMA tiles for smaller outputs and a 256-row/16-wave tile for the 32K-row projection. |
| `66294c6` | Add 32-head rocWMMA indexed and raw-ring attention paths. |
| `aac604b` | Link the MMQ objects into the ROCm regression target. |
| `fe74d50` | Replace hot SoA IQ2 sign-mask recomputation with the existing cache-hot 1 KiB lookup table. |
| `55f4072` | Replace the 4096xN-by-8192 attention-output-B hipBLAS GEMM with a shape-exact 64x64 wave32 rocWMMA kernel and add its standalone harness. |
| `8ae33fd` | Vectorize indexed-attention F32-to-F16 KV tile staging with aligned `float2` loads and packed `half2` LDS stores. |
| `24a339c` | Replace the repeated `24x2048x16384` native-F16 hipBLAS GEMM with a shape-exact padded-row rocWMMA kernel and add its harness. |
| `73c7091` | Replace the `64/256/512/1024 x 2048 x 4096` F16 hipBLAS GEMMs with shape-selected rocWMMA tiles and extend the exact-shape harness. |
| `0643952` | Bound IQ2 gate/up expert buckets by `n_tokens`, eliminating the known top-k factor from the rectangular launch. |
| `7345686` | Build a device-side list of live IQ2 `(expert, column-tile)` pairs and reuse it for gate and up. |

## Correctness and build checks

The Q8 tiling candidate is bit-identical to the saved attention/MMQ stack at
all 129,280 output logits for the 512, 1024, 2048, and 4096 token frontiers:
top-1 matches, max absolute error is 0, and RMSE is 0 at every frontier. The
attention/MMQ stack itself preserves top-1 at all four frontiers versus the
saved down-only reference; its worst observed full-logit difference is max-abs
5.14 and RMSE 0.871.

The IQ2 sign-mask change is also bit-identical to the saved 226.81 stack at all
four frontiers: same top-1, max absolute error 0, and RMSE 0. Its production
SoA gate/up microkernel falls from the saved 35.128 ms to 32.451 ms at 2K
(-7.6%) and reaches 65.532 ms at 4K. The end-to-end warm gain is smaller,
226.81 to 227.32 tokens/s.

The attention-output-B WMMA candidate preserves top-1 versus the 227.32 stack
at the 512, 1024, 2048, and 4096 frontiers. Its worst full-logit difference is
max-abs 4.276 and RMSE 0.712, inside the previously accepted attention/MMQ
envelope of 5.14/0.871. The standalone harness compares all 8,388,608 outputs:
max-abs is 4.66e-8, RMSE is 4.15e-9, and no element differs by more than 0.05.

The indexed-attention F32 vector-staging path is byte-identical to the validated
248.56 stack across every full-vocabulary logit at 512, 1024, 2048, and 4096
tokens: same top-1, max-abs 0, RMSE 0, and zero differing elements. It introduces
no new allocation or cache lifetime; conversion remains bounded by each owned
LDS tile.

The tiny-M F16 harness matches all 49,152 outputs (max-abs 0.00108, RMSE
0.00038, zero over 0.05) and takes 1.259 ms versus hipBLAS 3.230 ms. Full engine
logits are byte-identical through 2048 tokens; at 4096 top-1 matches with
max-abs 1.249 and RMSE 0.256, inside the accepted envelope.

The small-M F16 shape sweep matches every output against hipBLAS with worst
max-abs 9.35e-5 and zero elements over 0.05. The selected tiles take 0.149,
0.390, 0.745, and 1.393 ms for M=64, 256, 512, and 1024 respectively, versus
0.837, 1.029, 1.935, and 3.721 ms for hipBLAS. Full engine logits are
byte-identical to the tiny-M lead through 2048 tokens; at 4096 top-1 matches
with max-abs 1.168 and RMSE 0.213, inside the accepted envelope. Artifacts are
under `~/ds4/correctness/f16-smallm-wmma-20260827/`.

The tightened IQ2 expert-column bound is byte-identical to E042 at the 512,
1024, 2048, and 4096 token frontiers. It changes only the conservative launch
extent: a true top-k route cannot place the same token into one expert twice,
so no expert bucket can exceed `n_tokens`. The exact validation artifacts are
under `~/ds4/correctness/e045-tight-ncols-exact-20260827/`.

The compact IQ2 launch is also byte-identical to E045 at all four frontiers.
It preserves the x64/y64 MMQ tile body and changes only how live ragged expert
tiles are scheduled. The exact validation artifacts are under
`~/ds4/correctness/e046-compact-tiles-20260827/`.

`git diff --check` passes. After `aac604b`, the ROCm regression build compiles
and links `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, `ds4-agent`, and the test
binaries with the MMQ objects. Q4_K and MXFP4 dot tests pass 4/4, and the answer
extractor and agent tests pass. The monolithic `ds4_test` is not recorded as a
suite pass: it also expects prompt/golden fixtures absent from this checkout and
runs Metal-only exactness checks under ROCm.

Correctness artifacts on `fw2`:

```text
~/ds4/correctness/mmq-y64-attn-wmma32-warm-20260827/
~/ds4/correctness/q8-wmma16w-large-20260827/
~/ds4/correctness/mmq-sign-table-20260827/
~/ds4/correctness/attn-b-f16-wmma-20260827/
```

## Profile evidence

- The original x64/y64 IQ2 gate/up pair falls from 51.529 ms to 34.435 ms in
  the standalone production-shape lab, a 33.2% kernel improvement.
- The Q8 shape-aware tiling cuts the profiled custom-Q8 total from 2.148 s to
  1.660 s (-22.7%). The 32K-row projection falls from 14.605 to 11.160 ms/call.
- The 32-head indexed attention tile reduces the main 512-key kernel from about
  62.1 to 50.1-51.4 ms/layer. The raw-ring WMMA path removes a further roughly
  0.42 s from the warm interval.
- The current IQ2 MMQ kernel uses 184 VGPRs with no scratch spill. Counter and
  ISA inspection identify IQ2 unpack/live-state pressure as the next structural
  target: the kernel contains 384 `v_lshlrev_b16`, 256 `v_sub_nc_i16`, and 256
  `v_and_b16` instructions, with about a 69.5% L2 hit rate.
- At 2K, the saved IQ2 trace sizes each expert from all 12,288 gathered rows.
  Bounding an expert by the 2,048 input tokens removes the known top-k factor
  of six from this empty-tile launch and raises warm 4K prefill from 269.02 to
  277.29 tok/s.
- Compacting the remaining ragged expert tiles raises the separate 2K/4K
  candidate from E045's 198.17/277.29 to 207.22/278.42 tok/s. The larger
  short-context gain and small warm gain show that empty-CTA overhead is now
  largely removed; the warm path is dominated by useful IQ2 tile computation.
- Fusing raw-layout IQ2 gate/up into one x64/y32 CTA reaches 221.21/277.73
  tok/s at 2K/4K, but tracing rejects the apparent cold gain: the fused pair
  takes 31.677/31.657 ms versus 30.942/30.950 ms for E046's two launches
  combined (+2.4%/+2.3%). A first-layer differential probe verifies all
  6,291,456 outputs per leg bit-for-bit, and the clean 512/1024/2048/4096
  engine frontiers are byte-identical to E046. The rejected implementation and
  runtime switch are removed; only this measurement is retained.
- The small-M F16 specialization replaces 184 warm hipBLAS calls with 184
  rocWMMA launches. Its profiled pool is 180.0 ms and total warm GPU work falls
  from 7.517 s to 7.304 s; the profiled frontier reaches 267.70 tokens/s.
- The warm 4K interval starts at dispatch 4243. Its largest remaining kernel
  groups after the attention-output-B replacement are IQ2 MMQ gate/up
  (1.573 s), indexed attention (1.044 s), Q2_K hot down (0.994 s), and a
  230-call 64x32 hipBLAS group (0.778 s).
- The 64x64 F16 WMMA attention-output-B kernel cuts the warm 42-call pool from
  1.087 s (25.89 ms/call) to 0.499 s (11.88 ms/call), a 54.1% reduction. It
  uses 56 VGPRs, 128 SGPRs, 4 KiB LDS, and no scratch. The standalone shape
  sweep measures 13.784 ms versus 22.080 ms hipBLAS; 128x32, 128x16, 64x32,
  64x128, 128x64, and 32x128 all lose to 64x64.

## Rejected variants

- MMQ y32, y80, and y96 lose at the warm frontier; x64/y64 remains best.
- Eight-wave split-column MMQ spills at 256 VGPRs and regresses.
- x128/y64 MMQ takes 53.335 ms versus 34.435 ms for x64/y64.
- `launch_bounds` occupancy 3 lowers VGPRs from 184 to 177 without spilling but
  regresses the kernel to 37.3-37.5 ms.
- A 128x32 Q8 tile loses to 128x64, and an eight-wave Q2_K down tile lowers the
  warm result to 219.75 tokens/s.
- A four-output-tile Q2_K down kernel halves grid X but raises VGPRs from 48 to
  80. It takes 2.154 s across the profiled 2K/4K run versus 2.014 s for the
  two-output-tile kernel (+6.9%), and lowers the unprofiled warm result to
  225.95 tokens/s. The first 227.28 tokens/s run did not exercise this variant:
  its selector was in an inactive all-Q2 launcher, and the trace confirmed n2.
- rocBLAS logging maps the dominant attention-output-B GEMM to
  F16xF16-to-F32 `m=4096, n=2048, k=8192`, about 25.9 ms/call for 42 warm
  calls. Replacing it with DS4's existing direct-Q8 WMMA path lowers the warm
  result to 219.90 tokens/s, so that path is not a viable substitute. An
  earlier 227.22/227.63 result was invalid because the all-hipBLAS early return
  still selected the original B GEMM; its trace caught the inactive selector.
- A 64-head/1024-thread indexed-attention tile reaches 247.16 tokens/s with the
  E029 lead enabled, below the validated 248.56 result. Halving KV staging does
  not offset the larger block's occupancy cost; keep the 32-head tile.
- The x64/y64 IQ2 output mapping cannot be retuned to 8 or 16 waves by changing
  the wave-count parameter: both variants fail the structural assertion
  `nwarps * tile_C::I == mmq_y`. A mapping rewrite is required, and the prior
  split-column design already spilled at 256 VGPRs.
- The exact attention-output-A shape is strided-batched F16 GEMM
  `T,N m=1024,n=2048,k=4096,batch=8`. A standalone 64x64 rocWMMA kernel is
  correct across all 16,777,216 outputs (max-abs 2.29e-5, RMSE 2.56e-6) and
  takes 12.055 ms, but the live hipBLAS pool averages only 6.836 ms/call. It is
  therefore not integrated; `gfx1151_f16_batched_gemm_lab.cu` preserves the
  exact-shape harness.
- Packed-half2 Q2_K dequantization reaches 203.03/247.62 tokens/s at 2K/4K,
  below the validated lead, so it is rejected and the source is restored.
- Four-lane Q2_K scale broadcast reaches only 201.39/245.35 tokens/s. The two
  shuffles cost more than the redundant cache-hot metadata loads they replace.
- Caching a full 32-key-by-512 KV tile in LDS improves the short frontier to
  209.57 tokens/s but lowers warm 4K to 245.21 because it doubles online
  softmax/rescale blocks.

## Experimental 255 tokens/s lead

An opt-in indexed-attention F16 mirror plus packed half2 LDS staging reaches
208.01/255.46 tokens/s at 2K/4K; the lifetime-safe validation run reaches
255.19 tokens/s at 4K. HIP-event instrumentation measures about 46.24 ms for
the half-KV attention kernel versus 49.73 ms for the saved F32-load kernel,
with about 0.053 ms spent producing both mirrors.

This candidate is rejected and fully reverted. Reusing DS4's global temporary
buffer caused an asynchronous lifetime fault. Dedicated scratch completes all
four frontiers and preserves top-1, but the mirror still exceeds the accepted
1024-token full-logit envelope (max-abs 17.67, RMSE 2.55). Scalar half staging
is worse (24.93/3.98), isolating the difference to mirror production rather
than vector indexing. The follow-up using scalar `__float2half` conversion
generated repeated gfxhub TCP permission page faults at 16:02 and 16:07 and
ultimately required rebooting fw2. Preserve the 255 tok/s measurement only as
evidence for native F16 KV production or lifetime-bounded tile conversion; do
not reproduce the full-cache mirror.

## Next campaign

The refreshed lead profile prioritizes IQ2 gate/up (1.573 s), indexed attention
(1.044 s), and Q2_K hot down (0.994 s). Q2_K work must reuse scale metadata or
change data staging rather than widen output ownership or merely substitute
packed-half arithmetic. The indexed-attention experiment proves that F16 KV
plus vector staging can contribute another ~2.8% end-to-end, but the unsafe
full-cache mirror is closed. Future attention work must produce F16 KV natively
or convert only bounded tiles. E045 shows that IQ2 also wastes substantial
dispatch capacity on empty expert tiles. The next candidate is a device-built
compact `(expert, column-tile)` worklist that preserves the existing MMQ math
while removing the remaining rectangular per-expert overlaunch. E046 validates
that design but yields only 1.13 tok/s at the warm frontier, so the next IQ2
campaign must reduce useful-tile cost: share activation staging across gate/up,
shorten IQ2 unpack live ranges, or replace the raw-layout decoder with a
gfx1151-specific paired kernel. Every surviving kernel
must pass standalone output comparison and the saved four-frontier logit gate
before admission.

E047 implements shared activation staging in a paired x64/y32 raw-layout
kernel, but its pair is about 2.3% slower than E046's separate kernels. Sharing
Q8 staging alone does not repay the y32 scheduling and dual-weight decode cost.
The next IQ2 design should reduce duplicated decode instructions or accumulator
live state while retaining the validated compact x64/y64 launch topology.

E037 safely captures the tile-staging part of that opportunity and raises the
validated warm lead to 256.12-256.31 tokens/s. The remaining gap to 300 is about
14.6% of interval time. Attention work should now target WMMA or online-softmax
scheduling; the next major independent pools remain IQ2 gate/up and Q2_K hot
down.
