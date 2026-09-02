# DeepSeek V4 Flash Test Vectors

These vectors were captured from the official DeepSeek V4 Flash API using
`deepseek-v4-flash`, greedy decoding, thinking disabled, and
`top_logprobs=20`. The hosted API does not expose full logits, so these files
store the best logprob slice the API provides.

Files:

- `prompts/*.txt`: exact user prompts.
- `official/*.official.json`: official API continuations and top-logprobs.
- `official.vec`: compact C-test fixture generated from the official JSON.
- `local-golden.vec`: local top-k/logit fixture captured from a known-sane DS4
  Flash run. It is used to catch substantial backend drift that can keep the
  same greedy token while damaging the logits distribution.

Regenerate official vectors:

```sh
DEEPSEEK_API_KEY=... ./tests/test-vectors/fetch_official_vectors.py
```

Running the fetcher without `--only` also regenerates `official.vec`.

The C runner consumes `official.vec` directly:

```sh
./ds4_test --logprob-vectors
```

It also consumes the local golden fixture:

```sh
./ds4_test --local-golden-vectors
```

The runner opens the normal non-quality path with accelerator-specific fast
routes disabled and pins `DS4_METAL_PREFILL_CHUNK=2048` for this strict
official-vector check.

`official.vec` is intentionally trivial to parse from C: each case points to a
prompt file and each expected token is hex-encoded by bytes. The official JSON
files remain in the tree so the compact fixture can be audited against the raw
API response.

## Checkpoint directories

Flash checkpoints are not interchangeable for these tests. The flat files in
this directory are the fixtures the runner defaults to (the checkpoint the
default GGUF is; `local-golden-0731.vec` is the 0731 local golden — a
byte-identical copy of the flat `local-golden.vec` under an explicit name,
kept so the checkpoint is named in the path; commit it alongside the
checkpoint swap). Other checkpoints get their own directory,
upstream-style:

- `flash-vision-exp/`: imported verbatim from upstream (antirez/ds4
  `fc8bf3c`, 2026-08-31) for `DeepSeek-V4-Flash-Vision-Exp`. These are
  OpenRouter greedy **continuations** (`deepseek/deepseek-v4-flash-vision-exp`,
  provider Novita, `max_tokens` 32, reasoning off). That provider returns no
  logprobs, so every `official/*.official.json` carries zero logprob steps and
  no `official.vec` can be generated from them. Use `continuations/*.txt` with
  `manifest.tsv` as a greedy-agreement smoke against a Vision-Exp boot (the
  five prompts are the same as the flat set's), and record a
  `local-golden-vision-exp.vec` from a known-sane Vision-Exp run for drift
  regression (`DS4_TEST_LOCAL_GOLDEN_FILE=... ./ds4_test --local-golden-vectors`).
  Running a 0731 fixture against a Vision-Exp GGUF, or the reverse, is an
  invalid test, not a model-quality result.

To inspect a local top-logprob dump manually:

```sh
./ds4 --metal --nothink -sys "" --temp 0 -n 4 --ctx 16384 \
  --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
  --dump-logprobs /tmp/long_code_audit.ds4.json \
  --logprobs-top-k 20
```
