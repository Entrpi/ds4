# DSpark speculative decoding on Apple Silicon (Metal)

This branch makes the **DSpark block drafter** run end-to-end on the Metal
backend for DeepSeek‑V4‑Flash. It combines two pieces:

1. The three previously‑stubbed DSpark Metal ops (PR #2):
   `capture_mean`, `gather_concat`, `markov_step`.
2. Four correctness fixes on the continuous‑batching + drafter path that were
   blocking the drafter (and, for the first, blocking non‑drafter continuous
   batching too).

With all of it applied, the continuous path serves without falling back to the
serial path, and the DSpark drafter produces real, accepted drafts.

## The four fixes

Each is small, self‑contained, and (where noted) byte‑identical to the prior
behavior on the paths it does not intend to change.

- **Compressed‑KV banked emit wrote F32 through an F32‑sized view of the F16
  cache.** `ds4_gpu_compressor_update_tensor` runs pool/rms/rope in F32 and is
  shared by two callers with different cache dtypes: the serial decode caller
  passes the F32 stage (correct), while the banked `ms_row_emit` caller passes
  the F16 `layer_attn_comp_cache[il]` directly. Writing F32 into an
  `sizeof(float)`‑strided view of a half‑width buffer corrupts one element per
  comp row (a NaN at a fixed dim), which then propagates to 100% NaN at the
  first compressed‑KV attention layer once `n_comp` pushes the visible key set
  past the boundary. Fix: a caller‑scoped `output_is_f16` flag — the banked
  caller runs the emit in an F32 scratch and `copy_f32_to_f16`s into the
  distinct cache row; every F32 caller is unchanged.

- **`ds4_gpu_matmul_f32_tensor` was GEMV‑only** (`n_tok != 1` returned failure).
  The main model's HC‑mixer weight is F16 and takes the batched F16 path, but
  the DSpark drafter's HC‑mixer weight is F32 and the block runs it at
  `n_tok = block size`, hitting the early return. Fix: for `n_tok != 1`, run
  `n_tok` independent GEMVs over per‑row views. Byte‑identical for `n_tok == 1`.

- **`ds4_gpu_tensor_copy` needs an open batch command buffer.** It blits into
  the current `g_batch_cb` and returns failure when none is open (unlike
  `ds4_gpu_tensor_copy_f32_to_f16`, which owns its own). The DSpark inject block
  ends the batch cb, so the checkpoint‑restore copies in
  `mtp_cont_rollback_restore_all` had no cb and failed. Fix: open a cb around
  the rollback only when none is open (`begin_commands` returns 0 when one is
  already open, so the MTP path is unchanged).

- **The HC split/sinkhorn's "numerically stable" sigmoid overflows to NaN.**
  `0.5*tanh(0.5*z)+0.5` was introduced to bound the sigmoid, but under Metal's
  fast‑math `tanh` (which is `exp`‑based) it evaluates `exp(large) → inf`, then
  `inf/inf → NaN`, for large‑positive `z`. A drafter block row whose HC‑mixer
  logit is large enough (the real seed token) trips it. Fix: clamp the sigmoid
  input to `[-30, 30]` before the transcendental — sigmoid is fully saturated
  by ±30, so in‑range values are unchanged and only the overflow is removed.

## Observed behavior

Speculative decoding acceptance is content‑dependent (as it always is). Rough
single‑run numbers on an M3 Ultra, Q4‑imatrix target + the public Q2K DSpark
drafter, `temperature = 0`, with the per‑sequence yield guard disabled so the
drafter runs on every step (`DS4_DSPARK_QUENCH=0`):

| workload            | acceptance | tokens/step |
|---------------------|-----------:|------------:|
| repetitive text     |      ~68%  |       ~2.4x |
| stepwise arithmetic |      ~22%  |       ~1.3x |
| code                |      ~20%  |       ~1.2x |
| free prose          |      ~13%  |       ~1.2x |

These are illustrative, not a rigorous benchmark (single runs, one machine, one
quant). The yield guard is on by default and shuts the drafter off per sequence
when it is not paying for itself. A higher‑precision drafter (the public GGUF is
2‑bit; the reference checkpoint's experts are FP4) would raise the harder‑content
numbers, but is not required for a working, speedup‑positive drafter.

## Running it

```
./ds4-server -m <v4-flash-target>.gguf --dspark <dspark-drafter>.gguf --ctx 100000 ...
```

The drafter arms on the continuous path. `DS4_DSPARK_QUENCH=0` disables the
per‑sequence yield guard (useful for measuring raw acceptance).

## Known issue: M5 Max serving decode collapse (unresolved)

On an Apple M5 Max (128 GB, macOS 26.5/25F71), `ds4-server` from this fork at
v0.5.6.1 decodes at ~1 tok/s on the continuous Metal path, while upstream
`antirez/ds4` (`b030961`) on the same machine, same GGUF (the 0731 ship quant),
same session reaches 38–40 tok/s. Prefill is unaffected. It reproduces with
`--no-spec`, so the drafter is not the cause — with the drafter armed, the yield
guard quenches immediately (accept 0.0%), which is a symptom of the slow verify
steps. `DS4_METAL_NO_RESIDENCY=1` and context sizes 32k–49k make no difference.
The same build serves normally on CUDA/GB10. See the PR description for full
measurements. Until this is root-caused, upstream is the faster Metal server on
at least this hardware.

## Credits

This work stands on:

- **Entrpi/ds4** — the fork this branch targets (CUDA/GB10 DSpark drafter,
  block scorer, continuous batching).
- **antirez/ds4** — the upstream DeepSeek‑V4‑Flash inference engine and its
  Metal backend.
- **DeepSeek** — the DeepSeek‑V4‑Flash model and the DSpark speculative‑decoding
  method and reference drafter checkpoint (MIT).
- **Unsloth** — DeepSeek‑V4‑Flash GGUF quantization work.
- **llama.cpp** contributors — the Metal GEMV/GEMM and softmax kernel patterns
  these ops and fixes build on.

## Note on authorship

The four fixes, the root‑cause analysis, and this document were produced with AI
assistance (Claude) and reviewed before submission. Measurements are from a
single machine and should be reproduced independently before being relied on.
