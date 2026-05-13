# CUDA MTP on DGX Spark / GB10

This note covers the CUDA speculative decoding path for DeepSeek V4 Flash MTP
on NVIDIA GB10 systems such as DGX Spark. The feature is experimental, exact by
default, and currently useful as a small greedy-decoding speedup rather than a
large throughput multiplier.

## Hardware Target

The current CUDA MTP work was tuned against:

- NVIDIA GB10, `sm_121`.
- 128 GB memory class.
- Linux CUDA build with cuBLAS.
- Main q2-imatrix DeepSeek V4 Flash GGUF.
- Optional Q4_K MTP GGUF.

The optimization assumptions are specific to this shape: a very large target
model, an MTP draft block that is much smaller but still non-trivial, and exact
verification over a two-token suffix.

## Build

On Linux, the default build uses CUDA:

```sh
make clean
make -j8
```

The build uses `CUDA_ARCH=native` by default. Override it if the local CUDA
toolchain cannot infer the GPU architecture:

```sh
make clean
make -j8 CUDA_ARCH=sm_121
```

Run the CUDA regression target after changing CUDA kernels:

```sh
make cuda-regression
```

## Model Files

Download a supported main model and the MTP model:

```sh
./download_model.sh q2-imatrix
./download_model.sh mtp
```

The MTP model is not loaded automatically. Pass it explicitly with `--mtp`.

## Basic Run

No-MTP CUDA baseline:

```sh
./ds4 --cuda \
  -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --temp 0 --nothink -n 256 \
  -p "List 500 prime numbers, comma-separated, just numbers."
```

Exact CUDA MTP, draft depth 2:

```sh
DS4_CUDA_MTP_TOP2=1 \
DS4_CUDA_MTP_VERIFY_TOP2=1 \
DS4_CUDA_MTP_VERIFY_OPT_OUTPUT=1 \
./ds4 --cuda \
  -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 \
  --temp 0 --nothink -n 256 \
  -p "List 500 prime numbers, comma-separated, just numbers."
```

Use greedy decoding (`--temp 0`) when evaluating MTP throughput. Speculation is
most meaningful when target verification can accept or reject deterministic
drafts.

## Optimization Flags

The currently useful CUDA MTP flags are:

- `DS4_CUDA_MTP_TOP2=1`: use the CUDA top-2 output path for MTP draft decisions.
- `DS4_CUDA_MTP_VERIFY_TOP2=1`: use top-2/top-1 verifier shortcuts where full
  logits are not needed.
- `DS4_CUDA_MTP_VERIFY_OPT_OUTPUT=1`: schedule verifier output work to avoid an
  avoidable readback/command break on full accepts.

The structural CUDA optimizations are default-on after this work:

- Batched Q8 pair projections for two-token verifier passes.
- Fused six-expert MoE down+sum for the two-token exact verifier.
- Paired decode Q/KV projections.
- Startup pre-cache of both the base model and MTP model tensor spans.

Rollback switches:

- `DS4_CUDA_NO_BATCH_Q8_PAIR=1`
- `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6_N2=1`
- `DS4_CUDA_NO_DECODE_Q8_PAIR=1`

These are intended for A/B performance checks and debugging, not normal use.

## Benchmark Method

Use identical prompt, model, token count, and decoding settings across all
runs. Capture stderr separately because throughput and MTP timings are written
there.

```sh
BASE=./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
MTP=./gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
PROMPT="List 500 prime numbers, comma-separated, just numbers."

./ds4 --cuda -m "$BASE" --temp 0 --nothink -n 256 -p "$PROMPT" \
  > /tmp/ds4_nomtp.out 2> /tmp/ds4_nomtp.log

DS4_CUDA_MTP_TOP2=1 \
DS4_CUDA_MTP_VERIFY_TOP2=1 \
DS4_CUDA_MTP_VERIFY_OPT_OUTPUT=1 \
DS4_MTP_TIMING=1 \
./ds4 --cuda -m "$BASE" --mtp "$MTP" --mtp-draft 2 \
  --temp 0 --nothink -n 256 -p "$PROMPT" \
  > /tmp/ds4_mtp.out 2> /tmp/ds4_mtp.log
```

Compare output bytes before trusting a speed result:

```sh
cmp -s /tmp/ds4_nomtp.out /tmp/ds4_mtp.out && echo MATCH
```

For a rollback comparison:

```sh
DS4_CUDA_MTP_TOP2=1 \
DS4_CUDA_MTP_VERIFY_TOP2=1 \
DS4_CUDA_MTP_VERIFY_OPT_OUTPUT=1 \
DS4_CUDA_NO_BATCH_Q8_PAIR=1 \
DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6_N2=1 \
DS4_CUDA_NO_DECODE_Q8_PAIR=1 \
DS4_MTP_TIMING=1 \
./ds4 --cuda -m "$BASE" --mtp "$MTP" --mtp-draft 2 \
  --temp 0 --nothink -n 256 -p "$PROMPT" \
  > /tmp/ds4_mtp_rollback.out 2> /tmp/ds4_mtp_rollback.log
```

## Current GB10 Result

On the 256-token prime prompt above, a clean CUDA build on GB10 measured:

| Run | Prefill | Generation |
| --- | ---: | ---: |
| No MTP | 30.12 t/s | 15.46 t/s |
| Optimized exact MTP | 30.10 t/s | 16.21 t/s |
| Rollback structural opts | 27.00 t/s | 15.14 t/s |

The no-MTP, optimized MTP, and rollback MTP outputs were byte-identical for
that run.

MTP verifier timing from the same run:

| Run | Micro verify avg | Margin-skip verify avg |
| --- | ---: | ---: |
| Optimized exact MTP | 106.08 ms | 64.53 ms |
| Rollback structural opts | 118.81 ms | 65.75 ms |

The verified speedup is modest but real on this benchmark: optimized exact MTP
beat the no-MTP baseline by about 4.9%, and beat the structural rollback by
about 7.1%.

## Why The Gain Is Modest

MTP is not free. Each speculative cycle does three expensive things:

1. Runs the MTP draft block to propose candidate tokens.
2. Runs the target model over the drafted suffix to verify exact correctness.
3. Produces full target logits for the token that becomes the new sampling
   frontier.

The verifier dominates the remaining cost. Even with draft depth 2, the target
model still has to execute all layers over a two-token suffix. The CUDA work in
this branch reduces avoidable overhead inside that verifier, but it does not
remove the target-model pass itself.

The most important current structural wins are therefore narrow:

- Avoid duplicate Q8 activation quantization for paired projections.
- Fuse verifier MoE down projection and expert summation for the two-token case.
- Avoid unnecessary full-vocab output work when top-1/top-2 is enough.
- Keep startup model and MTP tensor spans warm so first-use copies do not pollute
  decode timing.

Future work should continue to target the target-layer verifier pass, especially
the large attention/output and MoE stages, while preserving exact output parity.
