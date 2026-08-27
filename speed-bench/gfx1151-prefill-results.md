# gfx1151 DeepSeek V4 Flash prefill

## Scope

- Host: `fw2`, AMD Strix Halo `gfx1151`
- Runtime: ROCm 7.14
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- Prompt: `speed-bench/promessi_sposi.txt`
- Workload: pure prefill, 2048-token increments, ending at 4096 context

## Result

| Build | 2K prefill | Warm 4K prefill |
|---|---:|---:|
| Clean DS4 | - | 187.26 tok/s |
| Tuned gfx1151 path | 231.91 tok/s | 292.86 tok/s |

The warm 4K result is 56.4% faster than the clean baseline. The final run had
all tuning selectors unset; gfx1151 now selects this path by default.

## Retained changes

- Use the tuned x64/y64 IQ2 MMQ gate/up path, bound each expert by the input
  token count, and launch only live expert tiles.
- Use shape-specific wave32 rocWMMA kernels for the hot Q8 and F16 projections.
- Use 32-head, 80-key rocWMMA attention tiles with vectorized F32-to-F16
  staging, plus the specialized attention-output B projection.
- Vectorize Q8 activation staging and stage each Q2_K weight block once per
  256-value K slab.

The environment variables remain as per-path debug overrides: setting one to
`0` disables that path. They are not required for normal gfx1151 execution.

## Validation

- The 512, 1024, 2048, and 4096 frontier logits are byte-identical to the
  previously validated 80-key attention reference.
- That reference preserves top-1 at all four frontiers against the earlier
  accepted stack. Its worst full-logit difference is max-abs 4.446 and RMSE
  0.770, inside the accepted max-abs 5.14 and RMSE 0.871 envelope.
- ROCm 7.14 builds and links `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and
  `ds4-agent`; all five pass runtime help smoke tests.
- Answer-extractor self-tests pass. Q4_K and MXFP4 dot tests pass 4/4 each.

Final fw2 artifacts:
`~/ds4/correctness/final-gfx1151-default-e19ca32-20260827/`.
