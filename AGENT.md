# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short Metal smoke tests for build verification.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `ds4_cuda.cu`: CUDA backend.  Single TU; mirrors ds4_metal.m's role on
  NVIDIA / DGX Spark.  Dispatches quantized matmuls either through its own
  bespoke kernels or - when `DS4_CUDA_USE_MMQ=1` is set - through the
  vendored llama.cpp `mul_mat_q` kernels in `cuda/mmq/`.
- `cuda/mmq/`: vendored llama.cpp ggml-cuda kernels (`mmq.cuh`, `mma.cuh`,
  `vecdotq.cuh`, `quantize.{cu,cuh}`, `mmid.{cu,cuh}`, `common.cuh`,
  `ggml-common.h`, `vendors/cuda.h`) pinned to upstream commit `5c0e9468`,
  plus the ds4-side adapter (`ds4_ggml_stubs.{h,cu}`, `ds4_mmq.{h,cu}`,
  redirect `ggml.h` / `ggml-impl.h` / `ggml-cuda.h`).  See
  `cuda/mmq/VENDOR.md` for the symbol-resolution table and the upstream
  re-sync procedure.  Phases 0-7 of the lift are documented in
  `local/docs/ds4_mmq_lift_plan.html` (in the auto-round companion repo).
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

## CUDA environment variables

The CUDA backend has a few opt-in switches that affect the inner matmul
dispatch.  Defaults preserve the historical behavior; set the variables
to opt in to the newer paths.

- `DS4_CUDA_USE_MMQ` (default on): the vendored llama.cpp `mul_mat_q`
  kernels in `cuda/mmq/` are the default CUDA path for quantized
  matmuls - Q8_0 dense (attention projections, shared expert, lm_head)
  and the routed-MoE block when the GGUF uses IQ2_XXS for gate/up and
  Q2_K for down (the V4 Flash configuration).  Other quant
  combinations fall through to the existing kernels.  Set
  `DS4_CUDA_USE_MMQ=0` (or `off` / `no` / `false`) to disable and
  revert to the native Q8 warp kernels (the
  `matmul_q8_0_preq_*_kernel` family).  The legacy Q8&rarr;FP16
  expansion cache plus `cublasGemmEx` pipeline that previously sat
  between mmq and the native kernels was deleted in Step 1 of the
  optimization plan - mmq is fast enough that the cache was no longer
  earning its complexity.  Validated on RTX PRO 6000 Blackwell
  (sm_120, CUDA 13.0) against V4 Flash: prefill 357-1041 tok/s vs
  357-373 baseline (sustained ~2.80x), gen within run-to-run
  variance.
- `DS4_CUDA_MMQ_MOE_MIN_TOKENS=N`: minimum `n_tokens` at which the
  routed-MoE mmq path activates.  Default 2.  The legacy decode kernel
  (`moe_gate_up_mid_decode_lut_qwarp32_kernel`) wins at `n_tokens=1`
  because mmq's matrix-matrix-shaped path has higher per-launch fixed
  cost than the legacy fused decode kernel.  Override to 1 to force
  mmq even at decode (slower today; may flip if mmvq kernels are
  lifted later).

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.
