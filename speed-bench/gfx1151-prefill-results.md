# gfx1151 DeepSeek V4 Flash prefill results

## Scope

- Host: `fw2`, AMD Strix Halo `gfx1151`
- Runtime: ROCm 7.14
- Engine base: DS4 `main` at `c1d4597a80e300b803dc642519718f2c999589da`
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- Prompt: `speed-bench/promessi_sposi.txt`
- Goal: 300 tokens/s warm prefill

## Current result

The validated experimental stack reaches 248.56 tokens/s at the warm 4K
frontier (247.49 tokens/s in the initial performance run), versus 227.32 before
the attention-output-B specialization, 187.26 for clean DS4, and 190.66 for the
accepted Q2_K down-only change. This is a 32.7% improvement over clean DS4. The
remaining gap to 300 tokens/s requires about 17.1% less interval time from the
current stack.

Required experimental switches:

```sh
DS4_ROCM_MMQ_IQ2=1
DS4_CUDA_MMQ_X_MAX=64
DS4_ROCM_ATTN_WMMA32_INDEXED=1
DS4_ROCM_ATTN_WMMA32_RING=1
DS4_ROCM_ATTN_OUTPUT_B_WMMA=1
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
- The warm 4K interval starts at dispatch 4243. Its largest remaining kernel
  groups are IQ2 MMQ gate/up (1.719 s), a 42-call 16K-by-64 hipBLAS projection
  (1.087 s), indexed attention (1.054 s), Q2_K hot down (1.002 s), and the
  remaining hipBLAS projection shapes (0.733 s).
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

## Next campaign

Regenerate the warm ranking around the 248.56 lead before selecting the next
kernel pool. Q2_K work should change dequantization or data movement without
adding live output accumulators; the n4 result rules out wider output reuse at
the current tile. IQ2 remains a structural target through shorter unpack live
ranges or a newer llama.cpp-style organization. Every surviving kernel must
pass standalone output comparison and the saved four-frontier logit gate before
admission.
