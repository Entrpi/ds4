# gfx1151 DeepSeek V4 Flash prefill results

## Scope

- Host: `fw2`, AMD Strix Halo `gfx1151`
- Runtime: ROCm 7.14
- Engine base: DS4 `main` at `c1d4597a80e300b803dc642519718f2c999589da`
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- Prompt: `speed-bench/promessi_sposi.txt`
- Goal: 300 tokens/s warm prefill

## Current result

The validated experimental stack reaches 227.32 tokens/s at the warm 4K
frontier (226.81 tokens/s before the IQ2 sign-mask change), versus 187.26
tokens/s for clean DS4 and 190.66 tokens/s for the accepted Q2_K down-only
change. This is a 21.4% improvement over clean DS4. The remaining gap to 300
tokens/s is 24.4% of the target throughput, requiring about 24% less interval
time from the current stack.

Required experimental switches:

```sh
DS4_ROCM_MMQ_IQ2=1
DS4_CUDA_MMQ_X_MAX=64
DS4_ROCM_ATTN_WMMA32_INDEXED=1
DS4_ROCM_ATTN_WMMA32_RING=1
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

## Rejected variants

- MMQ y32, y80, and y96 lose at the warm frontier; x64/y64 remains best.
- Eight-wave split-column MMQ spills at 256 VGPRs and regresses.
- x128/y64 MMQ takes 53.335 ms versus 34.435 ms for x64/y64.
- `launch_bounds` occupancy 3 lowers VGPRs from 184 to 177 without spilling but
  regresses the kernel to 37.3-37.5 ms.
- A 128x32 Q8 tile loses to 128x64, and an eight-wave Q2_K down tile lowers the
  warm result to 219.75 tokens/s.

## Next campaign

The next work should isolate IQ2 decode/unpack and the remaining Q2_K/hipBLAS
groups in shape-specific microbenchmarks, then sweep structural variants against
the x64/y64 reference. The leading hypotheses are to shorten IQ2 unpack live
ranges, stage packed values differently for gfx1151 wave32 WMMA, and selectively
replace the remaining projection paths using the current llama.cpp gfx1151 MMQ
organization as a reference. Every surviving kernel must pass standalone output
comparison and the saved four-frontier logit gate before admission.
