# Changelog

All notable fork-side changes to this project are documented here.
Fork: [Entrpi/ds4](https://github.com/Entrpi/ds4) of
[antirez/ds4](https://github.com/antirez/ds4); upstream fork point `e16ead1`
(2026-05-29). Upstream's own changes are not repeated here.

## Unreleased

- **`--tool-call-reminder` now covers the Anthropic surface, live
  tool-result continuations, and output-only Responses chains** —
  closing the coverage gaps in the v0.6.5 feature, which despite its
  help text only fired on the general render path for
  `tool`/`function`-role results:
  - Anthropic `/v1/messages` tool results are embedded into user-role
    content at parse time, so they never reached the reminder
    injection on ANY path. Past the depth gate, a tool-result-bearing
    user message now carries one reminder before its last result
    terminator (trailing text blocks, e.g. client system reminders,
    stay after it). Note this changes rendered bytes for deep
    (>~96KB) Anthropic histories vs v0.6.5, so the first replay of a
    pre-upgrade deep conversation re-prefills from the first reminder
    point.
  - Live tool-result continuations (Anthropic and Responses fast
    lanes) rendered their KV tail independently of the full renderer
    and appended tool results bare, so the deep tool loops the knob
    was built for never showed the model the reminder — and the live
    frontier's actual KV bytes diverged from the visible transcript
    the bookkeeping records. The live tail is now sliced byte-exactly
    out of a marked full render (`render_live_tool_tail_exact`), so
    the tail carries exactly the reminders — at exactly the bytes — a
    later full re-render produces, and warm prefix reuse and
    continuation revalidation see one consistent byte stream.
  - Responses continuations that send ONLY tool outputs without
    history (the stateless-chain shape) carry it too. A chained turn's
    `prompt_text` is a render of the outputs alone, not the transcript,
    so the depth verdict cannot come from the request: every
    continuation record (serial-session and batch-bank owners alike)
    now carries a rendered-byte frontier, seeded from `prompt_text` +
    the visible assistant suffix on a turn whose prompt is the
    transcript, and advanced by the consumed live tail + visible
    suffix on an output-only turn. The output-only tail evaluates the
    full renderer's predicate against that base, so its verdict is a
    pure function a later full re-render reproduces byte for byte
    (warm prefix reuse and continuation revalidation stay consistent).
    Where no base exists (no LIVE record, a record that published no
    frontier, the Anthropic unanchored fallback) the tail renders
    bare: under-injection stays the safe direction.
  - New server-group units: the Anthropic user-embedded shape
    (injection point, id guard, terminator literals in plain text),
    both live-tail lanes probed at the exact depth-gate boundary for
    byte-identity with the full render, a two-hop output-only chain
    (seed, advance, second-hop boundary) proven byte-identical to the
    full re-render, and the registry frontier lookup + arithmetic
    edge cases.
- **DeepSeek-V4-Flash-Vision-Exp loads as a text-only checkpoint
  (opt-in)** — the 2026-08-31 continued-training checkpoint's language
  GGUF (`antirez/deepseek-v4-gguf`, the same IQ2_XXS/Q2_K asymmetric
  recipe and the same imatrix as the 0731 ship quant) is accepted by the
  loader: it reads `deepseek4.checkpoint_variant`,
  `deepseek4.vision.sidecar_required` and `general.source.revision`,
  TAKES the checkpoint's `rms_eps = 1e-20` into the runtime shape (the
  previous loader only compared the value under an absolute 1e-6
  tolerance, so it silently accepted 1e-20 and kept serving at 1e-6;
  the epsilon is now taken from every checkpoint after a range check,
  and a deviation from the shape default is announced), and prints the
  variant on the boot line. Image input is NOT implemented: the encoder
  sidecar is not run (loading it is the `--vision` entry below), and
  image/file/audio content blocks on all three API surfaces are now
  refused with a named 400 instead of being silently flattened to their
  text parts. The 0731 checkpoint remains
  the auto-picked default. Boot-verified on GB10 (2026-09-02): the
  variant banner, the named 400s on all three surfaces, both wrong-pairing
  refusals, the auto-attached-drafter degrade, cache isolation under a
  shared `--kv-dir`, and the 0731 golden bit-exact on the same binary
  (`top20_max_abs=0`, cross-box).
  - Cache identity: `ds4_engine_model_id` is 2 for Vision-Exp (0731
    stays 0), so disk KV records, agent checkpoints and the distributed
    hello never match across the generations; a shared `--kv-dir` also
    stops being destructive — records of a non-zero model id are named
    by SHA1(id, text), so the two generations no longer erase each
    other's files on the same prompts. The API model id stays
    `deepseek-v4-flash`; `/v1/models` shows the checkpoint in the name.
  - Support-model gate: a Vision-Exp base refuses the 0731 DSpark
    drafter and the MTP file, and a 0731 base refuses the Vision-Exp
    drafter — by checkpoint variant, and a Vision-Exp support model
    must carry a pinned source revision that matches the base's. The
    gate runs before the tensor bind, so the refusal is named and the
    file is closed. A support model that launch defaults volunteered
    (sibling lookup) degrades to serving without it on refusal; an
    explicit `--dspark`/`--mtp` still fails the boot.
  - Launch defaults now resolve the base through `realpath`, so the
    documented `./ds4flash.gguf -> ~/gguf/<base>` symlink install
    detects the generation from the real file name and finds the
    drafter beside the real file. Note for existing 0731 symlink
    installs: the base now reads as 0731 (MTP retired, the 0731 drafter
    beside it auto-arms) where the link name previously read as legacy.
    Beside a Vision-Exp base only `DSpark-drafter-Q2K-Q8-vision-exp.gguf`
    is attached (the fork's own extraction: `gguf-tools/dspark_extract.py`
    with the 0731 Q2_K recipe; the extractor stamps the variant from the
    source `config.json` and requires `--source-revision` for it); MTP is
    retired for Vision-Exp as it is for 0731. New server units
    `test_launch_generation_names` (incl. the symlink case) and the
    media-block refusal.
  - `download_model.sh` gains `vision-q2` / `vision-q2-q4` /
    `vision-mxfp4` (the encoder is downloaded alongside),
    `vision-encoder`, and `drafter-0731` / `drafter-vision` (the fork's
    drafter repo) targets.
  - Fixtures: upstream's `tests/test-vectors/flash-vision-exp/`
    (OpenRouter/Novita greedy continuations; the provider returns no
    logprobs) imported verbatim as the API-side reference.
  - Quality parity of the Vision-Exp text weights against the fork's
    0731 baselines is pending (frozen-suite re-base); the default does
    not move before it lands, and the DSpark yield-quench guard keeps
    its 0731 calibration (2.16) until re-measured on Vision-Exp.
- **`--vision FILE`: the Vision-Exp encoder sidecar loads, binds and
  validates (Track B increments 0 and 1; image input still pending)** —
  the first two increments of image input for
  DeepSeek-V4-Flash-Vision-Exp, upstream-interop by design (the same
  sidecar GGUF, synthetic ids, preprocess and token layout as upstream
  antirez/ds4 `110afdd`, so upstream's binary serves as the oracle for
  every later increment):
  - Image substrate imported verbatim from upstream: the vendored
    single-header JPEG/PNG decoders (`third_party/iris`, MIT),
    `ds4_image.[ch]` (SHA-256 fingerprints, EXIF orientation, decode,
    bicubic resize, the Vision-Exp preprocess: patch 14, downsample 3,
    ≤ 384 tokens on a single variable-resolution grid, the N-order token
    layout with its position-dependent alignment pad) and its unit test,
    run by `make test`. The GLM 5.3 preprocess is dropped (the fork is
    DeepSeek-only). The substrate links with libc + `-lm` alone.
  - `--vision FILE` on the server, CLI and agent opens the encoder sidecar
    GGUF and binds it: architecture `deepseek4-vision`, variant
    `vision-exp`, `general.source.revision` equal to the base's, exactly
    316 tensors, the 11 geometry expectations and every tensor's
    type/rank/dims (BF16 ViT/aligner/sentinels, F32 router biases).
    Refusals are by name and fail the open: a non-Vision-Exp base, a
    distributed role, any bind mismatch. Nothing is auto-attached (launch
    defaults never look for a sidecar). The boot line reports
    `vision sidecar bound: … (316 tensors, sha256 …)`; the file's SHA-256
    is the sidecar half of the image cache identity. Images are still
    refused with the named 400s on all three surfaces: no API surface
    feeds the encoder yet.
  - `ds4 --validate-vision FILE` opens only the sidecar (no base) and
    prints name, variant, revision, the 316-tensor bind verdict and the
    SHA-256; the shipped encoder validates in about 3 s.
- **The Vision-Exp image encoder runs on CUDA, bit-exact against
  upstream (Track B increment 2)** — upstream antirez/ds4's DeepSeek
  encoder (`110afdd`: the 32-block ViT, the aligner, the BF16 round-trip
  points and the cuBLAS call, byte for byte) re-expressed on the fork's
  idioms: every launch on the current stream, weights through the
  registered sidecar map, the fork's global cuBLAS handle bound to the
  stream around the GEMM and refused under graph capture, a caller-owned
  BF16 staging buffer instead of the shared scratch. Compiled into every
  CUDA build by default (`-DDS4_NO_VISION_ENCODER` opts out; Metal and CPU
  refuse by name). Engine API: `ds4_engine_vision_encode_file/memory`
  return a `[tokens x 4096]` F32 embedding in upstream's natural layout
  with the image SHA-256 (`fingerprint`, upstream-identical) and the cache
  identity (`identity` = SHA-256 of image ‖ sidecar ‖ preprocess version).
  The sidecar gets its own residency row (`DS4_WEIGHT_RESIDENCY_VISION`,
  boot line `vision=`) and catalog line. No API surface feeds it images
  yet: the named 400s stay. Receipt (GB10, 2026-09-03): the fork's
  embeddings are bit-identical (`max_abs=0`) to upstream's own binary on
  all eight fixture images (16x16 to 2400x1800, 8:1 aspect, PNG and JPEG),
  the fork reproducing itself exactly across runs while upstream's encoder
  differed between two of its own runs on one shape; the text path is
  unchanged with the encoder compiled in (unit battery + 0731 golden on the
  same binary). Oracle harness `tests/dump_vision_embedding` +
  `tests/compare_vision_dumps.py` are tracked (they build unchanged against
  upstream's tree).
- **`DS4_CONT_MTP_GATE=1` certifies a DSpark drafter** — the in-engine
  losslessness gate (the s11 frontier-invariance / token-identity proof)
  required a `--mtp` boot; 0731 and Vision-Exp retired the MTP head, so
  their ship pairings (base + DSpark drafter) exited rc=2 and could not
  be certified by it. The gate now accepts an armed DSpark drafter as the
  draft source (exactly the forward's own arming predicate). On a
  DSpark-only boot the MTP-chain-only mode-1 probe is skipped and
  reported n/a, `DS4_CONT_MTP_ACCEPT` is required, and the draft source
  is announced. Engagement receipts: run E records the finished sequence's
  draft and hit counts and every forced-draft prefix-causality run its
  draft count (the per-run accept lines carry the hits), and a run with
  zero drafts at D ≥ 1 is rc=2 with a named cause, so a disarmed drafter
  can never pass as lossless by luck.
  On a DSpark boot the gate's effective depth is the block drafter's
  (`DS4_DSPARK_VERIFY_DEPTH`; `DS4_CONT_MTP_DEPTH` does not apply, D=0 is
  not a DSpark shape) and the forced-draft hook reaches the DSpark
  block-draft fill, so the prefix-causality legs really do push wrong
  drafts through the verify: the forced runs' accept stats now differ
  from the unforced run (hits 0 at the first position), the N=8
  sensitivity leg diverges with rollback off, and the rollback self-check
  stays byte-exact. GB10 receipts on the ship pairings: Vision-Exp base +
  its drafter at N=1 (D=4 and D=1) and N=8, and the 0731 pairing at N=1,
  all rc=0.

## v0.6.5 — 2026-08-27

- **`--tool-call-reminder on|off`, default on** (env
  `DS4_TOOL_CALL_REMINDER=0` disables,
  `DS4_TOOL_CALL_REMINDER_MIN_BYTES` retunes the gate) — the
  reference-faithful fix for the deep tool-protocol slip. At depth the
  2.4-bpw quant answers some tools-armed turns with a prose completion
  report instead of a tool call (measured: 50% of turns at 70-80K
  prompt tokens; every observed slip fired right after a successful
  tool result). Past ~96KB of rendered conversation (~30K+ tokens),
  every tool result in a tools-armed chat now carries a short protocol
  reminder. Measured: 0/72 sampled slips with the reminder vs 6/72
  without at the exact captured slip states, and the failing agent task
  end to end went from three dead runs to submitted and
  harness-resolved in 57 calls with zero slips — with reasoning traces
  kept, no resample spent, and no format deviation. The reminder is
  injected on every qualifying result (not just the last) so the
  rendered history is byte-stable across turns and warm prefix reuse is
  unaffected; shallow conversations are never touched, so
  chat-with-tools flows that legitimately answer in prose after a tool
  result see no change. Disclosure: default-on means benchmark runs
  carry it unless disabled; runs reporting benchmark numbers should say
  which position the knob was in.

## v0.6.4 — 2026-08-26

The issue-#18 residual-gap release: with `--reasoning-replay drop
--tool-slip-resample`, a SWE-rebench agent instance that llama.cpp
resolves and that this engine previously failed three ways
(harness-graded, identical scaffold and request shape) is now submitted
and resolved identically to llama.cpp.

- **`--reasoning-replay keep|drop`** (env `DS4_REASONING_REPLAY=drop`) —
  agent scaffolds that echo assistant messages verbatim re-send
  `reasoning_content`, and the tool-context chat render re-emits it
  inside `<think>` blocks. That is the V4 reference format: the
  reference encoding disables thinking-drop whenever tools are present,
  and DeepSeek's API requires the echo in tool loops (it returns 400
  when `reasoning_content` is not passed back). llama-server's template
  default drops it, a deviation from the reference format, and on the
  identical conversation that deviation runs 16-28% shallower per turn.
  Depth is where the 2.4-bpw quant's tool-protocol adherence slips
  (measured: 50% of turns settle as prose completion reports instead of
  tool calls at 70-80K prompt tokens), so on low-bit quants the
  deviation pays. `drop` reproduces it opt-in: history assistant turns
  render in the lean `</think>` replay form; assistant turns being
  continued (after the last user-like message) always keep their
  reasoning. Measured steady-state cost: a one-turn-tail re-prefill
  (~270 tokens / ~0.6 s) absorbed by the cont bank's partial-prefix
  admission. The default stays `keep`, which is both the byte-identical
  prior behavior and the reference-faithful one; `drop` is the
  recommended setting for long agent loops on low-bit quants.
  (Correction note, 2026-08-26: this entry originally claimed
  DeepSeek's API rejects replayed reasoning; the opposite is true in
  tool loops, per their thinking-mode guide.)
- **`--tool-slip-resample`** (env `DS4_TOOL_SLIP_RESAMPLE=1`, off by
  default) — when a continuously-batched non-streaming tools-armed chat
  turn settles at `finish=stop` with no tool calls, the request is
  requeued once at the FIFO head for a fresh draw before anything is
  written to the client; the just-retired bank warm-admits the full
  prompt, so the retry costs one generation. Pinned seeds are perturbed
  (+1) for the redraw; `length`/`error` settles, streaming turns, and
  the serial lane are never resampled; retries are exempt from the
  queue-age shed while keeping the original arrival stamp for honest
  latency. Disclosure: this knob changes benchmark behavior by
  construction (it retries the engine's own sampler before the harness
  sees the turn) — in the validating run, 6 slips across 142 turns were
  all rescued invisibly and the task went from three dead runs to
  submitted-and-resolved. Runs that report benchmark numbers should say
  whether it was on.
- **`--tool-slip-dump DIR`** (env `DS4_TOOL_SLIP_DUMP_DIR`) — forensic
  instrument: every tools-armed chat completion that settles without
  tool calls dumps one self-contained JSON file (raw request body,
  full generated text, parse verdict, lane), on both the
  continuous-batching and serial lanes. Each dump is a byte-exact
  replay fixture. `--trace` usage text now documents that session
  tracing covers the serial lane only.

## v0.6.3.1 — 2026-08-25

- **Client reasoning_effort compat-mapping (field issue #18)** — the
  OpenAI-surface `reasoning_effort` field no longer reaches the
  checkpoint's prefixed tiers by default: client `high`/`xhigh`/`max`
  resolve to prefix-free `low`. The v0.5.3 tier rename made a client
  `"high"` start injecting DeepSeek's position-0 "write out your entire
  deliberation" preamble (verbatim reference-encoder text), and a
  controlled needle matrix shows that preamble degrades deep-context
  tool calling: 6/50 completion-protocol failures at ≥96K tokens with
  the prefix (including a deterministic greedy flip) vs 0/100 without,
  on both v0.5.2 and v0.6.2. Agent frameworks send the field meaning
  the OpenAI "think more" knob; llama.cpp and pre-rename engines no-op
  it, which is the behavior restored as the default. Operators opt back
  into the native tiers with `--reasoning-effort-native`
  (env `DS4_REASONING_EFFORT_NATIVE=1`); the `--reasoning-effort`
  operator default is honored as written either way. Applies to every
  client surface (OpenAI chat/completions/responses and the Anthropic
  `output_config.effort`); disabling thinking (`none`/`off`) stays
  client-reachable.
- **Mixed-spelling DSML tool-call parsing** — at depth the model
  sometimes frays tag spellings inside one tool block (measured live at
  ~96K: DSML envelope with plain-XML `<parameter>` tags; the issue-#18
  capture's death loop is the model retrying exactly such calls after
  the parser demoted them to content text and the harness answered "no
  tool calls found"). The generated-message parser now matches each
  element (block end, invoke open/close, parameter open/close)
  against all three spellings independently, and a parameter value runs
  to the earliest end-tag spelling, so mixed open/close pairs still
  terminate. Canonical DSML parses byte-identically to before
  (regression-tested).

## v0.6.3 — 2026-08-21

- **Numerics note** — the decode-dispatch change in the full-window
  work (below) shifts some temperature-0 generation trajectories: the
  frozen eval battery holds quality parity (needle retrieval perfect
  at every depth, code suites exact, everything else within the ±1
  band; receipts in the release notes), while benchmark scores that
  hinge on a handful of long scenarios can move a few points between
  kernel paths. Tool-eval-bench restamps at 83/100/82/80 on this
  lineage.
- **Best-fit trim victims** — admission-pressure reclaim used to
  walk victims purely in recency order, so a deep trunk could die
  for a deficit a small idle bank would have covered (measured:
  want 1,646 MiB, released a 2,896 MiB bank, +76% over-reclaim and
  the deepest warm context destroyed). When one victim's release
  covers the whole remaining deficit, the engine now picks the
  smallest such victim in the same validity class, and among equal
  releases the shallowest one (same bytes freed, less warm context
  destroyed); earlier victims in a multi-bank reclaim keep the
  recency order (they are consumed whole either way). The substitution is disclosed
  (`best-fit victim` line) and the trim summary reports released vs
  wanted. `DS4_BATCH_TRIM_BESTFIT=0` restores the pure recency
  walk. (A prefix-preserving tail trim was prototyped and refuted
  on receipt: VMM page granularity plus the raw-ring warm-fork
  floor cap its yield below one page per slab at any context, so
  whole-bank release with a better-chosen victim is the honest
  fix.)
- **Whole-prompt depth fence** — the whole-prompt probe modes
  (`DS4_METAL_PREFILL_CHUNK<=0`, or an explicit chunk wider than
  8,192) could submit single forwards of unbounded depth into
  kernels that are unqualified past 8,192 rows, failing as a crash
  or silently wrong output. Such requests now get a typed refusal
  naming the lever (server: typed 503 before any allocation; engine:
  a named error for direct callers), and the boot log discloses the
  mode whenever it is active. `DS4_PREFILL_NOFENCE=1` lifts the
  fence for deliberate probe runs. The default chunked path cannot
  hit the fence. Also fixes doc drift: `DS4_CONT_PREFILL_CHUNK_LIVE`
  defaults to 512, not the documented 4096.
- **The full 1,048,576-token window, qualified to the last token** —
  the deepest ~32k tokens of the window (compressed rows past 7,936,
  first crossed at 1,015,936 resident) previously rode a fixed-size
  score buffer on the fallback decode paths: the head-group flash
  kernel was skipped past the cap, a captured fallback froze its row
  count and silently dropped the deepest rows on replay, and
  substrate callers refused outright. The dispatch now tries the
  uncapped head-group kernel first at depth and gives the online
  fallback live per-request scalars, so every supported decode shape
  serves the full window (audit Finding 1). An exactly-full persisted
  bank also restores now instead of being rejected by an off-by-one
  bound (Finding 2). Gate: tail-needle legs at 1,029,340 prompt tokens
  (needle at 99.9% depth, retrieved exactly) on both the default and
  forced-fallback paths, plus an exact-fill persist/restore leg.
- **Think-dial observability** — serving log lines now carry
  `think=<none|low|high|max>` (the request's effort dial; the THINKING
  flag remains the live inside-think state), and the continuous lane
  gains the per-request completion line it never had
  (`cont chat ctx=... gen=... think=... finish=...`); previously a
  cont-served request was log-invisible and seeing its effort needed
  serial debug plus `--trace`. A `think_modes` counter family lands in
  `/v1/stats` and `/metrics` (`ds4_requests_think_total{mode=...}`).
  The agent now warns when `--think-max` steps down below the 384K
  context floor (CLI and eval already warned; the server honors
  explicit levels at any context since v0.5.4).
- **Chunked request bodies** — the HTTP reader now decodes
  `Transfer-Encoding: chunked` request bodies (proxies such as
  llama-swap, gpustack, and Open WebUI chains re-frame bodies as
  chunked; previously the reader parsed only `Content-Length`, so a
  proxied request read as an empty body and died as a JSON error).
  The decoded body observes the same 64 MiB cap; chunk extensions and
  trailers are discarded; malformed framing answers 400.
  `DS4_SERVER_CHUNKED=0` restores the previous reader.
- **Typed refusal for schema-constrained output** — a `response_format`
  requesting `json_object` or `json_schema` (OpenAI surfaces),
  `text.format` (Responses), or `output_format` / `output_config.format`
  (Anthropic) now answers HTTP 400 with a message naming the mode, in
  the endpoint's native error envelope, instead of silently serving
  free text a client would try to parse as JSON. Plain `{"type":"text"}`,
  `null`, and omitted stay accepted; the string spelling of a schema
  mode is refused too rather than skipped. Structured output itself
  (constraining decode to a schema) remains unimplemented and tracked.

## v0.6.2 — 2026-08-19

Real budgets. v0.6.1 made admission charge what a request actually
commits; this release makes every remaining floor and margin a
measurement too, and makes the account prove itself continuously. The
anti-thrash floor was still denominated in virtual extents — pricing
two full-depth working sets at 26.71 GiB at `-c 786432` where they
really commit 6.68 — so the budget's capacity leg never engaged
honestly at depth; the boot bank count came from a frozen ladder; the
planning headroom was a constant tuned for one configuration; trim
picked victims by history length, keeping deep trunks immortal under
pressure while re-trimming recently-hot banks; and nothing checked the
box's raw memory drop against the engine's own ledger. This release
closes all of it: the shipped 262144 default now boots 9 banks where
the ladder froze 4, capacity governs the budget honestly at depth, a
stripped 1 GiB-floor box gets ~3 GiB of planning margin back, and an
idle-tick reconciliation line holds the unexplained residual to
2–19 MiB against a 256 MiB tolerance across the gate battery. One
disclosed constant remains: the boot-burst half of the derived
headroom ships as a 2048 MiB default pending its own measurement.

- Governed bank plan: the boot bank count is priced from the live
  memory budget instead of a frozen fundable-token ladder. Above 16k
  context the plan asks what the budget actually funds at the same
  per-token rate admissions are charged (so plan and admission cannot
  disagree): a 262144 default boot now grants 9 banks where the ladder
  froze 4, a 524288 boot grants 6 (08-18 box state; the count follows
  the box). At or below 16384 the measured 32-bank regime stands
  verbatim. An explicit `DS4_SERVER_COALESCE_MAX` still rules AND
  disarms the sizing — the operator's number is a deterministic pin;
  `DS4_BATCH_FIT_KV=0` restores the ladder end to end.
- The KV budget's anti-thrash floor is priced in committed terms: the
  guarantee was always "two full-depth working sets", but it was
  denominated in virtual per-bank extents (13.4 GiB at `-c 786432`)
  when a full bank actually commits ~2.9 GiB — the floor clamped the
  measured capacity answer up by ~4.6x and the budget's capacity leg
  never engaged honestly at depth. The floor now prices two full-depth
  banks at the packed rate admissions are charged (band included). The
  boot ledger names which floor ruled (`work floor=packed|virtual`) and
  prints its own line whenever the floor is what bound capacity.
  `DS4_BATCH_VMM_FLOOR_PACKED=0` restores the old denomination.
- The boot planning headroom derives from the operator floor: the
  static 6144 MiB was the shipped 4 GiB floor plus ~2 GiB of runtime
  growth margin bundled for one configuration only. It is now computed
  as `--mem-floor-gb` + `DS4_BATCH_FIT_BURST_MB` (default 2048) — the
  same 6144 at shipped defaults, ~3 GiB returned to the fundable pool
  on a stripped 1 GiB-floor box. `DS4_BATCH_FIT_HEADROOM_MB` still
  pins outright; `DS4_BATCH_FIT_HEADROOM_DERIVED=0` restores the
  static value.
- Trim victims are chosen like warm-record eviction and named in the
  log: invalid content first, then the longest-idle bank (shortest
  history breaks ties), replacing shortest-history-first — which kept
  deep trunks immortal under budget pressure while re-trimming
  recently-hot small banks. Each trim now logs one line per victim
  (bank, bytes released, history length, recency) ahead of the summary
  line, and the boot ledger discloses the active order.
  `DS4_BATCH_TRIM_VICTIM=hist` restores the old order.
- Reconciliation line (the zero-headroom law made continuous): at every
  idle tick the server reconciles the box's raw available-memory drop
  since boot settle against what its own allocation census plus the
  named one-time charges explain, and logs the signed residual instead
  of silently absorbing it — a future phantom or leak surfaces as a
  named number, not a field report. Fresh copies on `/v1/stats` and
  `/metrics`; `DS4_MEM_RECONCILE_TOL_MB` (default 256) flags the line,
  `DS4_MEM_RECONCILE_STRICT=1` emits a distinct token for gate scripts.
  The ~650 MiB census-invisible first-admit warmup (driver module
  loads, JIT, host allocator growth) is captured once as a named
  one-time charge (`DS4_MEM_RECONCILE_WARMUP_MB` pins it); the serial
  session's memory row now publishes the allocator's own measured bytes
  instead of the estimate once the graph commits.
- Default context raised to 262144 on CUDA builds (was 32768; Metal/CPU
  keep 32768). Rationale: with the v0.6 memory model a deep window is
  demand-mapped and nearly free until used, while the old 32k default
  was a footgun — a defaults boot could not fit even one long agentic
  request (prompt plus the 32768-token decode budget assumed when
  `max_tokens` is omitted). At 262144 the governed bank plan still
  funds a healthy bank count (9 on the reference box). Users chasing
  maximum concurrency for shallow batch work set `-c` low explicitly,
  as they already tune the bank count. The default resolves after flag
  parsing, so `--metal`/`--cpu` on a CUDA build get the conservative
  default.
- Weight-server manifest content identity (closes the limitation
  chartered at v0.6.0): the manifest now carries a per-model content
  fingerprint (`content <id> <size> <algo> <hex>`, strided FNV-1a
  sample: full head and tail windows plus one page per 16 MiB) and the
  engine verifies it against its own mapping of the model file before
  importing any range. A weight server left holding a superseded
  checkpoint of identical size is now refused loudly at boot instead
  of feeding stale bytes silently. Old manifests without the record
  still import, with a one-line notice; unknown fingerprint algos
  downgrade to the same notice; `DS4_WEIGHT_FP_CHECK=0` disables
  verification. This is mixup detection, not tamper-proofing: bytes
  between sample windows are not covered.

## v0.6.1 — 2026-08-17

Memory truth. Admission charges what a request will actually commit —
measured, not feared. A field trace showed the projection charging
4.9 GiB for a 245k-token bank whose real packed commit was ~750 MiB
(~6.5x pessimism from charging virtual extents and phantom decode
budgets), refusing an admission the box could fund six times over and
evicting work onto the expensive serial lane. Seven increments delete
that class end to end; the concurrent charter shape — a 500k-token
admission in flight while two 245k admissions land beside it — now
serves with zero refusals on a 128 GiB box, and decode under ~1M
resident bank tokens runs within 2% of an empty box.

- Tranche decode credit (`DS4_CONT_ADMIT_TRANCHE`, default 32768;
  0 = legacy): admission credits the prompt plus one decode tranche
  instead of a ~393k-token phantom budget for every request that omits
  max_tokens; live rows extend credit tranche-by-tranche under the same
  funding verdict, and a refused extension finishes the row cleanly at
  its funded boundary (`finish_reason: length`) instead of rejecting
  the whole request up front.
- Honest serial lifecycle: the session graph's cost is published as a
  lease when committed and released when freed (no phantom intent), and
  an idle reaper (`DS4_SERIAL_IDLE_REAP_S`, default 120) returns the
  whole right-sized session — measured +5.3 GiB back — after idle,
  re-allocating lazily on the next serial request. Fixed a leak where
  freeing a serial session stranded its captured layer-graph
  executables until an unrelated pool resize.
- Defaults re-derived from measurement: ctx-aware bank grant (32 banks
  through 16k, halving against a ~1 Mi fundable-token depth, floor 4),
  flat 6 GiB fit headroom replacing the 8 GiB deep tier, prewarm gated
  on its real consumers, and the boot "context buffers" print labeled
  as the estimate it is (now including the FP8/FP4 mirror terms).
- Pool truth: captured-graph executables are census-visible
  (`graph_exec` class, driver-measured per slot, released verbatim at
  destroy) with an idle sweep on the pressure ladder; the q8-f16 cache
  budget and managed-KV routing read the typed observation instead of
  raw driver numbers.
- `--no-serial` (env twin `DS4_SERVER_NO_SERIAL=1`): opt-in cont-only
  serving; every serial-lane execution path answers a typed 503
  (`reason="lane_disabled"`). Fixed a latent `/v1/models` race with the
  reaper, and the cont bank plan now sizes from the configured ctx.
- Disclosed admission band (`DS4_CONT_ADMIT_BAND_X1024`, default 1045):
  the only margin left in the projection is the measured transient peak
  (leg-calibrated 1.02x sequential), applied uniformly so admission,
  extension, and lease refreshes charge one truth. Live commit-rate
  feedback exports observed vs packed bytes-per-token
  (`ds4_cont_commit_bytes_per_token`) with a loud one-shot warning if
  observed ever exceeds 2x physics — measured on the ship config:
  observed 4366 B/tok vs packed 4142 (1.054, page-floor amortization).
- Standing deep battery: `speed-bench/memcal_gate.sh` (E0-E5
  calibration: 4.67 KiB/tok all-in at 507k admitted tokens) and
  `speed-bench/deep_admit_ab_gate.sh` (the charter A/B: truth credit
  serves the concurrent deep shape, union 4692 MiB, loaded-decode
  1.019x; the legacy credit at a budget 1.2x that union reproduces the
  field refusal chain on demand) join six fast memory-truth gates in
  the release battery.

## v0.6.0 — 2026-08-17

The memory governor is authoritative. The shadow accounting arc built
across v0.5.6.x (typed observation, allocation census, epoch-coherent
snapshots, governed checks) now makes the real admission decisions for
all five memory consumers — engine boot, prewarm, the bank plan, the
serial session, and the per-call batch graph. `DS4_MEMGOV=observe` is
the one-word rollback to shadow-everywhere. (Development receipts for
this release carry historical v0.5.7 naming; the arc was renumbered at
cut prep.)

- Governor enforcement across all five consumer families: one typed
  observation, one lease ledger, one evaluator; every refusal is typed
  with a reason that carries retryability
  (`ds4_requests_rejected_total{lane,reason}`; legacy scalars frozen
  byte-identical). Two-phase server-ranked idle-bank reclaim lets the
  serial lane collect from the commons before refusing — exact-deficit
  trims, warm records preserved unless content is actually destroyed.
  Per-source residency policy is now typed (eager / lazy / host-mapped)
  and chosen by measurement rather than by name heuristics.
- FIX (affects every v0.5.x): the captured serial decode path baked its
  top-k scoring band at capture time, sized to the prompt — decode rows
  appended after capture were invisible to the attention selector and
  the stream guard clamped (the TOPK-BOUND-VIOL tripwire that found
  it stays in the binary). The band is now the capture-stable session
  cap at every launch site. Serial decode selection changes as a
  result; quality re-stamped through the full gate battery.
- FIX (aged servers): the CUDA graph pool's freed-graph reserve was
  released exactly once per process, at boot. Long-lived servers
  accumulated reserve that counted against their own admission checks
  and floor-rejected organic work beside it (soak-measured: an 11.5 h
  server sagged ~1.3 GB and refused ~6 requests/cycle; a restart
  zeroed both). Refusal paths now return engine-owned reserve before
  any live-memory refusal (`DS4_MEM_OWN_TRIM=0` opts out), counted and
  disclosed.
- Memory observability: /metrics, /v1/stats, and the board now render
  the raw estimate pair behind the availability answer (`cuda_free` vs
  `meminfo_available`) plus own-trim counters, all from one
  epoch-coherent snapshot per render. A low kernel estimate beside
  succeeding admissions is an accounting artifact, not a leak — the
  cross-boot experiment showed every kernel bucket flat across
  boot/kill cycles while the estimate re-scored page-cache state.
- The zero-config launch defaults are now checkpoint-generation aware
  (serving-gguf audit 2026-08-12: the compiled-in default file names
  were the pre-0731 set, so a raw `ds4-server -c N` on a box holding
  both generations silently served the old checkpoint). Resolution
  prefers the -0731 file names and falls back to the previous-
  generation names; the generation is detected from the resolved base
  file name — the same scheme as the ds4-on-spark installer, so
  GGUF_FILE overrides keep working in both directions. The 0731
  checkpoint has no MTP head (replaced upstream by the DSpark stages),
  so the legacy MTP gguf never auto-attaches beside a 0731 base —
  `--preset spark` included — and the boot line says so
  (`mtp=retired`). An explicit `--mtp` still wins, as do all existing
  flags and env overrides. Gate: `launch_defaults_gate.sh` new
  `gen_resolve` fake-layout leg plus 0731-shape assertions on the
  zero-config and preset legs.
- The DSpark trace-replay tool now defaults to the shipped quench
  controller, so an explicit replay reproduces serving behavior
  instead of a rejected prototype. The shipped break-even guard (2.16,
  unchanged since v0.5.0) was re-validated at corpus scale this cycle:
  59 traces, a flat plateau from 2.16 to 2.30, quench lifting the
  corpus floor from 0.731 to 0.962.
- Field builds are warning-clean on all three backends (CUDA, Metal,
  CPU), and the release gate now fails on any warning at all. The pass
  was annotation-only, proved by object-code identity (18 of 20
  objects byte-identical in their code sections, including the entire
  engine and every kernel), plus one real hardening fix: the daily
  update-check now handles a pathologically long `$HOME` instead of
  writing a truncated stamp path.
- Release evidence: 24 h enforcement soak, 48/48 cycles clean (six
  saturation pulses, observe-rollback slice mid-run, zero faults);
  bit-exact golden vectors across eager/lazy/mapped; width-384 live
  stamp; the full gate battery including the two new gates
  (`topk_band_gate.sh`, `memdrift_attrib.sh`).
- Known limitation, chartered as the first 0.6.x item: weight-server
  manifest imports currently trust path and size — a swapped file of
  identical name and size would be accepted silently. A content-
  identity check on import is designed but deliberately not rushed
  into this release.

## v0.5.6.3 — 2026-08-11

Memory-footprint fix for long-running serving on memory-tight boxes
(forum 378855 post 86 — thanks to emptysands for a field report with
the measurement discipline to bisect it: identical loads, both
process and system metrics, two engine versions back to back).

- The KV-cache growth budget is re-tethered to capacity. Since v0.5.5
  the budget was the bank plan's own allowance (every bank at its
  full cache extent) regardless of what the box could afford; on a
  32k boot that authorized more than free-minus-headroom, so
  demand-mapped cache pages marched into system memory with no brake
  engaging — measured as a steady MemAvailable drain under sequential
  load and, on a memory-tight single Spark, an eventual system OOM
  that v0.5.0 did not exhibit. The budget is now the plan allowance
  capped to measured capacity at boot-settle, floored at two full
  banks so pressure trims can never evict every warm record. Under
  the same 20-minute reproduction the drain now front-loads the bank
  working set and goes flat (0.02 GB in the second half); the trim
  gate confirms pressure is absorbed by page recycling with zero
  admission rejects and intact warm records.
- The boot ledger shows the decision: the batch vmm line now prints
  budget=[chosen] [plan X, capacity Y]. DS4_BATCH_VMM_BUDGET_MB
  still pins the budget explicitly; DS4_MEM_FLOOR_GB (default 4)
  remains the operator floor for admissions.
- Also documented from the same report: the ~8.2 GiB startup weight
  span cache has been a runtime default on GB10 since v0.5.6 (it is
  what the +~7 GB VmHWM and lower MemAvailable baseline vs v0.5.0
  are). It is deliberate — promotions land before the bank plan so
  every build plans honestly — and DS4_CUDA_NO_HBM_CACHE=1 opts out
  while keeping the plan honest via the outstanding-substrate charge.

## v0.5.6.2 — 2026-08-10

The proper Codex fix. v0.5.6.1 documented two caveats around OpenAI
Codex on custom providers: compaction summaries could be poisoned by
leaked tool-call markup, and Codex needed a hand-installed catalog
file to learn the context window at all. Both are now fixed
server-side, plus the stream-timeout failure behind long thinking
turns. Validated end-to-end: the codex compaction task that failed on
every earlier run now completes through multiple compactions, and a
ten-minute single thinking turn survives at Codex's stock timeouts.

- Tool-call syntax is terminal on requests that declare no tools. The
  model can open a DSML tool-call block after plain text (it did so on
  every observed compaction summary); previously that raw control
  syntax streamed to the client as text and the finish still flipped
  to tool_calls. Now the visible message ends exactly at the marker
  with an honest finish=stop, partial markers are held off the wire,
  and DSML quoted inside thinking stays inert. A no-tools client can
  never receive tool syntax, so transcript consumers (Codex compaction
  summaries included) stay clean.
- GET /v1/models can teach Codex the model directly: point
  DS4_CODEX_MODELS_FILE at the catalog file shipped with ds4-on-spark
  and the response carries Codex's ModelInfo schema alongside the
  OpenAI list (Codex tolerates the combined body; OpenAI clients
  ignore the extra field). Codex then self-configures — real context
  window, working auto-compaction, correct agent instructions — with
  nothing but the provider block in its config. The boot log warns if
  the file's context_window disagrees with the booted -c.
- Streams now heartbeat during silent decode stretches. A thinking
  turn with no reasoning deltas on the wire is minutes of silence, and
  Codex's ~300 s idle timer killed such turns (it counts parsed SSE
  events — comment keepalives do not reset it). A live Responses
  stream heartbeats with the API's own response.in_progress event
  every 5 s; Anthropic streams use the protocol's native ping;
  other surfaces get an SSE comment. Prefill keepalives unchanged.

## v0.5.6.1 — 2026-08-09

Fast-follow fix for a session-killing interaction found while testing
agent harnesses against v0.5.6, plus the first community Metal
contribution.

- Long agent sessions no longer die after a stream reconnect. After a
  tool turn, the serial session briefly reserves itself for that turn's
  continuation; in v0.5.6 the reservation refused any other request
  from the same client for the full 60-second grace window — longer
  than real clients keep retrying, so one dropped stream during a long
  thinking turn could end the whole session (Codex hit this in about
  25 seconds). The reservation now sheds competing work only for a
  short seat window (DS4_CONT_HOLD_SHED_S, default 5 s, with an honest
  Retry-After); after it, the newcomer is served and a late
  continuation falls back to the documented 409-and-replay contract,
  which interruption checkpoints keep cheap. A queued continuation's
  hard pin still protects its turn exactly as before. Gated by the full
  continuation-registry battery plus a live replay of the failing
  session: the identical run that died in 25 seconds served 2.57M
  cumulative tokens to its honest completion.
- Metal: the DSpark block drafter now runs end-to-end on Apple Silicon
  — three previously-stubbed ops plus four correctness fixes on the
  continuous/drafter path, contributed by robotnursenyc (PR #2) with
  measurements on their own hardware. Metal correctness on this fork's
  serving paths is community-maintained: gated here by compile and the
  isolated Metal kernel regressions; see METAL_DSPARK.md. Two CUDA-side
  amendments rode the landing (a mirrored signature and a
  scoped-commands capability) so GB10 serving is byte-identical.
- Known, documented for transparency: OpenAI Codex (0.144.x) never
  auto-compacts its transcript against custom providers (upstream
  issues #16068/#19185), so very long Codex sessions grow until they
  hit this server's honest capacity refusals. Not a server bug, and
  Claude Code compaction is unaffected. A partial local workaround
  ships with ds4-on-spark: a `model_catalog_json` catalog file that
  teaches Codex the real context window, validated live to make
  Codex's own compaction fire instead of the session dying at the
  capacity wall (see "Pointing OpenAI Codex at the box" in the
  ds4-on-spark README). Honest caveat from the same validation:
  post-compaction task retention depends on the transcript summary the
  model writes, and DeepSeek can leak DSML tool-call markup into those
  summaries, which degrades later turns. Suppressing that leak
  server-side, plus serving Codex's /v1/models schema so no catalog
  file is needed, is chartered for v0.5.7. For long agent sessions
  today, Claude Code remains the recommended harness.

## v0.5.6 — 2026-08-08

The first-class API release: Anthropic Messages and OpenAI Responses
are now first-class surfaces of the batched engine, served in the
continuous batch alongside OpenAI chat and completions instead of
being translated on the way out of the serial lane. The OpenAI-only
batched path was a deliberate first-version scoping choice; this
release retires it. Tool-call continuations become a first-class
engine object with an honest replay contract, and the serving
internals the promotion depended on were converged and measured.
Thanks again to OllieOllie and emX0r, whose v0.5.5-era reports shaped
several of the honesty fixes that shipped along the way.

- Anthropic and Responses ride the continuous batching lane: buffered
  and streaming, thinking (with signatures), stop sequences, and
  streaming tool calls all batch. Transport starts at admission, so
  time-to-first-byte no longer waits on a batch slot, and prefill
  keepalives use each protocol's native ping. Buffered tool turns
  deliberately stay serial: that lane can feed the model a corrective
  tool error after a failed parse, and the batch cannot yet, so
  correctness keeps the lane until an equivalent exists.
- Tool-call continuations are a first-class engine object. A completed
  tool turn publishes a continuation record binding its call ids to
  the exact engine state that produced them (serial session or batch
  bank); an output-only follow-up claims that state in place with a
  strict generation-and-frontier equality check, and anything stale
  answers a native 409 telling the client to replay full history.
  Retention is honest and bounded: a grace window after each turn, a
  soft TTL, a queued-continuation hard pin, and victim placement that
  never destroys a bank a live continuation is about to claim (a
  fully protected set sheds 503 with a truthful Retry-After).
- Serving honesty across every surface: engine-stranded streams end
  with the protocol's native stream-error event instead of a
  fabricated success finish; stop-sequence hits report the matched
  text (Anthropic stop_sequence was previously end_turn with a null
  match on the serial lane); an explicit max_tokens of 0 answers with
  zero tokens on every lane; negative token budgets are a 400 naming
  the client's own field instead of a silent zero; errors are native
  envelopes on all four surfaces with Retry-After on refusals.
- Usage is client-frame everywhere: cache read/write token details
  reflect the client's prompt even when a warm engine admission
  evaluates a longer effective prompt, and usage always matches the
  timings block. Responses reasoning_tokens is now counted from the
  generated tokens themselves rather than re-tokenizing the parsed
  reasoning text at finalize, and the streaming terminal reports it
  on tool turns too.
- Explicit admission bounds with native shedding: connection, queue
  depth, queue bytes, queue age, and slow-reader output caps each
  refuse with the endpoint's own envelope plus Retry-After, and every
  shed is counted by reason. A stalled reader can no longer park its
  stream in kernel socket memory beyond the server's accounting
  (DS4_SERVER_CLIENT_SNDBUF caps per-connection send buffering).
- One route decision: request requirements are computed once at
  enqueue and a single pure function decides the lane at every
  dispatch site, proven equal to the legacy predicate tree over an
  exhaustive table and observable via per-surface, per-lane, and
  per-reason counters.
- The serving internals converged: one semantic accumulator now feeds
  both the serial loop and the continuous per-token callback (text,
  thinking, tool-marker and stop scanning, sampling overrides), so a
  fix lands once instead of twice; the Responses stream machine
  renders from spans of the row's single text buffer, halving
  per-stream memory. Projection cost inside the batch loop was
  measured at ~70 microseconds per token (well under one percent of a
  decode step), so no offload machinery was added.
- Engine fix: the speculative accept path could retire a finishing row
  before committing its final accepted tokens to bank history, which
  made the published continuation frontier one step short and refused
  every bank continuation; retirement now happens strictly after the
  step commits.
- Memory governance groundwork (the "lite" arc): resident-set
  accounting no longer overcharges scale slabs (a false-reject
  vector), page-trim keeps ownership when the driver fails an unmap,
  the admission credit projection's page-interval union is extracted
  and unit-pinned, and the serial-fit reclaim path has its own gate.
  The 240k deep gate now also runs a zero-config leg, validating
  shipped defaults rather than a tuned environment.
- The new batched routing is the DEFAULT on every surface; nothing
  needs to be configured. Transitional escape hatches, kept for
  exactly one release then removed: setting
  DS4_SERVER_CONT_ANTHROPIC=0, DS4_SERVER_CONT_RESPONSES=0,
  DS4_SERVER_CONT_TOOLS_ANTHROPIC=0 or
  DS4_SERVER_CONT_TOOLS_RESPONSES=0 restores that surface's previous
  serial routing exactly, for comparison. Continuation retention
  knobs: DS4_CONT_GRACE_S / DS4_CONT_TTL_S / DS4_CONT_PIN_DEADLINE_S
  (defaults 60/300/60).
- Observability: continuation registry counters (published, resolved,
  missed, demoted, live), route decision and shed-reason vectors,
  queue and backlog gauges, projection-cost and /v1/batch wait
  counters. /v1/batch keeps its own contract (documented): it is a
  synchronous bench endpoint and waits out a live batch epoch; the
  measured wait equals the epoch's remaining drain, the same as
  behind a long serial request.
- Measured and published, per the acceptance list: with max_tokens
  omitted (the 384K default budget) a deep -c 524288 boot admits two
  concurrent batch rows, because admission promises every row its
  full output budget; further requests are refused honestly and served
  serially. Requests that state a realistic max_tokens admit at the
  width their budget funds. A renewable credit policy that would widen
  the omitted-budget case is designed but deliberately not shipped
  until real traffic asks for it.

## v0.5.5 — 2026-08-05

The illegal-access release: the intermittent CUDA crash that several
GB10 boxes hit under sustained agentic load is root-caused and fixed,
alongside four serving-honesty fixes from the same field docket.
Thanks to OllieOllie for the budget double-booking analysis and the
capture/topk bisect that pointed at the right kernel family, and to
emX0r for the serial-lane report that drove the reservation work.

- The illegal-access crash class is fixed (Xid 13 "Out Of Range
  Address"; forum 378855 post 30, forum 376884 posts 113-129): the
  deep-context top-512 selector's buffer-compaction decision read a
  shared counter that the next tile's append could still move on the
  skip path — a nanoseconds-wide race that let sibling warps disagree
  about compacting, corrupt the block's sort state, and write outside
  the shared-memory window. The decision value is now frozen under a
  barrier before it is read (selections are byte-identical; one extra
  barrier per tile). On the deterministic reproducer: 16 interleaved
  fresh-boot pairs, 5 crashes on the unfixed binary, 0 on the fix.
  The window explains every field trait — rare and intermittent,
  worse on hot or long-running boxes, and vanishing under every
  debugging mode that was tried.
- New DS4_SERIAL_RESERVE_CTX lever for deployments whose single-request
  lane matters more than batch depth (forum 378855 post 49): set it to
  the deepest serial prompt you need served and the server carves that
  much memory aside at boot, so cache growth can never leave the lane
  unable to allocate. It is off by default, and the measurement is
  worth stating plainly: a right-sized serial session graph costs about
  7.3 GiB, while deep batch serving at -c 524288 leaves about 6.8 GiB
  free on a 128 GiB box. The two lanes cannot both be funded at depth,
  so reserving statically does not remove the shortage, it only chooses
  who loses it, and on a default boot that must not be the batch path.
  The general fix is for the serial lane to reclaim cache pages when it
  needs them, which is the next release's work. When the lever is on,
  the reservation is still capped so it cannot take what the batch path
  needs to function, and the boot line reports the carve-out.
- The KV-cache page budget is now the bank plan itself (banks x full
  per-bank extent), not a boot-time free-memory sample. The sampled
  budget needed a grant-back and a live refresh to survive its own
  noise and could still trim warm records to a noise-derived number;
  all three are gone. Live-memory truth stays where v0.5.4 put it:
  the admission-time floor verdict, which now also charges other
  in-flight admissions' outstanding projections (the joint-admission
  overrun class analyzed by OllieOllie, forum 376884 post 129) and
  the serial reserve.
- A max_tokens cut that lands inside a tool-call emission now reports
  finish_reason "length" instead of "error" (agent harnesses treated
  "error" as retryable failure), returns the partial call as
  assistant text, and never decodes past the stated budget on the
  recovery path.
- The warm-record matcher ranks reuse candidates by the tokens they
  actually deliver, with an explicit guard: exact-token reuse of a
  committed prefix structurally beats re-deriving alignment from
  text, so byte-length comparisons no longer decide picks.
- Interrupted-work checkpoints (v0.5.4) extended: a bank's emit-keep
  state now follows fork copies and re-arms after being spent, fixing
  a slow accumulation where partial-fork history could block
  continuous-graph capture for the bank's life (the "gradual
  slowdown" field reports).
- GB10 build-target advisory + a reference Dockerfile (forum 378855
  post 48): a generic `make cuda` on GB10 silently drops the Spark
  weight-read cache — the boot log now names the faster target once;
  the Dockerfile builds the right one.
- Diagnostics that stay: a permanent bound tripwire in the top-512
  selector, fault-surviving instrumentation modes in the kernel
  microstress, and measurement gates for the serial reservation,
  finish-reason honesty, matcher picks, and tool-turn re-render
  alignment (measured: 1-token asymmetry on the hot path).

## v0.5.4 — 2026-08-03

Field-report release: every change traces to a report in the NVIDIA
developer forum threads. Thanks to GaelicThndr for the disk-restore
report and the single-Spark reasoning-effort question that drove two of
these.

- reasoning_effort reachable on every box (forum 378855 post 30): the
  server no longer silently downgrades high/max below 384K context —
  explicit request values are honored at any --ctx (a boot advisory
  replaces the downgrade), and --reasoning-effort low|high|max|off sets
  the default for requests that do not send the field.
- --version prints the release version on installer builds (forum
  378855 post 30's log): a tag-less installer clone made git describe
  return a bare commit hash, which the update check treated as older
  than any release — a daily spurious "update available" self-nag on
  up-to-date boxes. The VERSION file now takes over whenever tags are
  absent.
- A drafter failure disarms speculative decode instead of killing the
  continuous batch (found while chasing forum 376884 post 113): one
  loud line, then plain lossless decode for the rest of the process.
  Committed tokens only ever come from the base verify forward, so
  draft production is best-effort by construction.
- Operator memory floor --mem-floor-gb (default 4 GiB): admissions now
  gate against live free memory, not just the boot-time budget plan.
  KV-cache growth is trimmed or rejected before free memory drops below
  the floor, whatever the box has lost since boot (other processes,
  page-cache churn). The boot line reports the floor; 0 disables.
- Interrupted work now leaves checkpoints, not cold banks (forum 376884
  posts 126-127, forum 378855 post 41): an aborted or failed admission
  keeps the bank's committed chunk watermark and re-announces it as a
  warm record, so a retry resumes where the interruption landed instead
  of re-prefilling from zero (measured: a retry after a mid-replay
  disconnect reused 34,402 tokens that v0.5.3 threw away). Aborted or
  budget-exhausted reasoning requests likewise keep a prompt record —
  previously each retry of a timed-out ~96k summarization re-paid the
  full prefill.
- A failed admission chunk aborts one job, never the batch: the
  admission loop restores its pre-chunk counter snapshot, logs one loud
  line, and keeps serving the other tenants (previously any chunk
  failure killed the whole continuous batch).
- Client-disconnect detection now recognizes reset-style closes: a
  client that quits mid-stream with unread bytes sends RST, not a clean
  close, and the liveness probe treated that as alive until a failed
  write caught it much later.
- Continuous-batch responses report true cached_tokens: warm, fork, and
  partial admits reported 0 in usage.prompt_tokens_details for every
  admit shape (the serial path was already honest).
- Official logprob vectors refreshed against the 0731 model via the
  official DeepSeek provider (hard-pinned through OpenRouter): the unit
  battery runs all five vector cases with zero exclusions again.
- New release gates: update_check stamp leg, drafter_disarm_gate,
  mem_floor_gate (external-loss hog + deep ceiling-replay legs),
  restore_reject_gate covering the forum 378855 post 30 shape
  (restore -> budget-reject -> serial, clean across 11 cycles on two
  releases), and bank_mutation_gate (interrupted warm/truncate replays
  + mid-decode disconnects of reasoning rows, with watermark-reuse and
  usage-honesty receipts per cycle).

## v0.5.3 — 2026-08-02

Field-robustness and interface release: three engine increments plus two
community PRs.

- Deep partial-truncate regression gate (forum 376884 posts 113+122): the
  in-place truncate-reuse envelope is now pinned by speed-bench/
  truncate_gate.sh — seven sequential truncates of one bank ending at the
  reported crash geometry (cut < ctx/2 < committed), with a needle inside
  the replayed suffix and a FORK_PARTIAL=0 control. The reported v0.5.0
  crash did not reproduce across 18 attempts on v0.5.0 or v0.5.2; the
  remaining suspect is v0.5.0's adaptive drafter residency, removed in
  v0.5.1 — if you are on v0.5.0, upgrade.
- Responses API usage now counts reasoning tokens (forum 376884 post 115):
  output_tokens_details.reasoning_tokens was hardcoded 0.
- --version, --check-update, --upgrade, and a once-daily update check: a
  plain GET of the one-line LATEST file in ds4-on-spark shortly after the
  server starts listening; no payload, never blocks or fails boot; disable
  with --no-update-check or DS4_NO_UPDATE_CHECK=1.
- PR #5 (Fabio Pili): the three reasoning_effort levels the 0731 model
  card documents (low/high/max), reachable and distinct; "off" accepted;
  default rendering byte-identical.
- PR #6 (a-huk): the GB10 graph-fit gate trusts MemAvailable; the former
  pinned-file subtraction starved the fit estimate on tight boxes and
  503'd the serial lane.

## v0.5.2 — 2026-08-02

Field-robustness follow-up, chartered from the v0.5.0 announcement
thread's reports: three fixes for what deep-context and interrupt-heavy
deployments actually hit. No performance re-bases — 12k serving twins
byte-exact per increment and across the whole release.

- **Serial requests at deep `-c` are served, not 500'd** (`f1712b5`):
  the serial fallback session's lazy graph was sized by server `-c`, not
  the request — on a bank-holding deep boot every serial-path request
  (Anthropic/Responses APIs, non-streaming token-id echo, continuous
  admission rejects) demanded a graph that could never fit beside the
  banks (measured: an ~11 GiB ask for a 26-token `/v1/messages` job at
  `-c 250000`) and died with `lazy session graph alloc failed`,
  deterministically. The session is now re-created at the largest ctx
  the fit gate passes, bounded by prompt + budget + continuation
  headroom; a later, deeper serial request regrows it (the old graph is
  freed before its replacement is probed). When even a minimal output
  window cannot fit the answer is a clean, retryable 503 — never the
  doomed alloc. Serial-only deployments (no batch ctx) keep the v0.5.1
  contract untouched. `DS4_SERVER_SERIAL_RIGHTSIZE=0` opts out;
  `speed-bench/serial_rightsize_gate.sh` is the standing gate.
- **Armed speculation with no engine says so** (`4cb91f8`):
  `DS4_SERVER_COALESCE_MAX=1` (or `DS4_SERVER_COALESCE=0`) skips the
  persistent batch ctx at boot, and speculative decode only lives in
  the continuous engine — so a server with MTP/DSpark proudly armed
  silently decoded plain serial at roughly half speed. The boot now
  prints a loud warning naming the cause and the fix
  (`DS4_SERVER_COALESCE_MAX>=2`, or `--no-spec` to disarm instead).
- **Dead clients stop costing GPU** (`19e5ed1`): a killed streaming
  request already aborted within a step, but a killed **non-streaming**
  request decoded to its full budget (default 384K tokens) for nobody,
  and a client killed **mid-prefill** had its whole deep prompt
  prefilled (~80 s at 45k) and then began decoding. The zombie-reap
  socket probe now runs at every abort point: between admission prefill
  chunks (a new engine liveness hook — the pending bank resets to free
  at the next chunk boundary), at each continuous decode token, and at
  each serial decode step. Measured: a mid-prefill kill now wastes one
  chunk (14% of that prompt) instead of the full prefill plus a phantom
  decode. `DS4_SERVER_DISCONNECT_ABORT=0` opts out;
  `speed-bench/abort_paths_gate.sh` is the standing gate.

## v0.5.1 — 2026-08-02

The agentic-serving robustness fast-follow: four field-driven fixes for
the population running long-lived servers with disk KV, plus the removal
pre-announced in v0.5.0. No performance re-bases — every increment
shipped with byte-exact 12k serving twins against its kill switch.

- **Adaptive side-model residency removed** (`b8702cd`): the v0.5.0
  mlock/watchdog mechanism measured zero speed win twice (bitwise
  identical speculation counters; the hold zone and the refault-tax
  zone are disjoint on 121 GiB unified memory), so it is gone — exact
  inverse of `6f59a16`, byte-exact twins against the v0.5.0 binary.
- **Trim-on-evict for the demand-mapped KV pool** (`23eab49`): the
  v0.5.0 known issue. Evicting a bank under comp-cache budget pressure
  now unmaps the pages that lie entirely inside its extent (VA
  reservations stay, so captured graphs stay valid; the next tenant
  re-maps on first emit), preferring hist-invalid then cheapest-warm
  victims. Engagement gate: pinned budget that previously bounced 7
  admissions to the serial path now trims 7 banks (~168-210 MiB each)
  with zero rejects, needles exact; a control boot with
  `DS4_BATCH_VMM_TRIM=0` reproduces the old rejects. Corollary from
  gate development: at 16k ctx a bank's whole extent is its floor page,
  so shallow-context servers were never exposed — the creep is
  depth-proportional (the field report was at 131k).
  `speed-bench/vmm_trim_gate.sh` is the standing gate.
- **Disk-KV cross-boot replay survives divergence** (`6874572`): a
  stored record had to byte-prefix the incoming prompt exactly, so a
  length-truncated turn — whose next-turn re-render closes the turn —
  made its record permanently unmatchable: silent cold re-prefill
  forever, zero log lines (the deep-probe field shape). The disk tier
  now gets the same partial path live banks have had since P1: records
  rank by byte-LCP against the prompt (salvage must cover ≥1/8 of the
  record; restore streams ~10× faster than prefill), restore into a
  victim bank under the existing depth rules, and the normal token-LCP
  cut takes over. Gate: divergent replay went 89.9 s cold → 1.3 s
  partial restore, needle exact through the cut tensors; a control
  boot (`DS4_SERVER_DISK_PARTIAL=0`) reproduces the silent cold prefill.
- **Warm-admit checkpointing** (`6874572`): banks only persisted on
  evict/shutdown, so the most-reused agentic trunks — the ones that
  never evict — were the least durable; a crash lost everything since
  the last foreign admit. Pin-tier retires now persist
  (`reason=bank-checkpoint`), paced by the continued-store interval;
  checkpoints supersede their predecessors at budget eviction. Gate:
  exactly one checkpoint at a deep trunk's retire, a short turn 2 does
  not re-persist, and kill -9 → reboot restores the checkpoint warm
  (ttft 0.6 s, needle exact). `DS4_SERVER_BANK_CHECKPOINT=0` opts out.
- **MTP accept guard** (`18b0c98`): the MTP-2 draft arms had no yield
  governance (DSpark has the calibrated quench). Measurement refuted
  the feared failure mode first: a cross-generation pairing (0731 base
  + legacy MTP) drafts at 51.9% accept — right at the D=1 break-even —
  and decodes at 22.3 tok/s vs 21.9 plain and 24.4 matched, because
  same-lineage checkpoints transfer and the ggufs carry no generation
  metadata to refuse by. What ships is a floor: after 256 drafts,
  cumulative accept under 15% (~random = genuinely foreign or corrupt
  support model) trips a terminal process-wide disable with a loud
  line; decode continues plain and lossless. The MTP load line now
  announces the armed guard; `DS4_MTP_ACCEPT_GUARD=0` disarms.
  `launch_defaults_gate.sh` grew four legs, including a benign-verdict
  tripwire that flips if a future refresh makes cross-pairing harmful.
- **Gate hardening**: `bank_persist_gate.sh` + `launch_defaults_gate.sh`
  boots now wait for memory reclaim (the recurring "transient" serial
  failure was the gate racing its own dying server: 8.8 GiB allocatable
  vs ~9.1 needed while pages drained), and the persist gate grew the
  divergent-replay and crash legs above.
- DSML-markup-in-reasoning (field report, legacy era): does not
  reproduce on 0731 under natural tool use (0/5 probes; reasoning is
  terse and clean). It appears only when the prompt explicitly asks the
  model to discuss its invocation syntax — where stripping would
  corrupt requested content — so no filter ships.

## v0.5.0 — 2026-08-01

The deep-prefill/decode substrate release, shipped together with the
DeepSeek-V4-Flash-**0731** weights refresh. Two stories in one cut: the
engine closes its flat-pool arc (every remaining per-layer activation
requantize retired, bit-exactly) and unlocks CUDA-graph capture at every
depth; the model refresh re-bases the quality baselines dramatically
upward on the identical ship-recipe quant.

**Engine — flat-pool arc (all bit-exact, each gated separately):**

- **Fused own `out_a`** (`9339bf0`): attention out_a epilogue becomes one
  fused kernel, retiring a pack + cuBLAS pair. 515K prefill record
  679.3 s = 759.6 t/s (+2.3%) at its ship — since re-taken at the
  v0.5 tip: **517,963 tokens / 667.1 s = 776.4 t/s** (+2.2% again, the
  stream-topk tier working inside deep prefill selection); ABBA 2k
  +4.4%, 64k +3.5%.
- **q8_1 dual-emit** (`26662e0`): out_a epilogue dual-emits the q8_1 of
  low, retiring out_b's input quantize (verify 0/16384 diffs).
- **MoE gate/up Y-indirect staging** (`9c1c4cb`): quantize once per
  token + ids_src map folded into one prologue register per thread
  (trip-invariance; NT64 SASS byte-identical; verify 172/172 clean).
- **Norm triple-emit** (`76f5266` + `507921d`): the RMS-norm producers
  emit f32 + f16 + the q8 D4 activation in one pass behind a
  pointer-keyed producer registry — all FIVE per-layer K=4096 input
  quantizes retired, zero dispatch-site changes. Verify 344/344 +
  172/172 clean; ABBA 64k +1.33% / 2k +2.77%, above the GPU-time bill
  because 215 retired launches per window also relieve the CPU-bound
  shallow window. Bench records: 2k 958–960, 64k ~933 t/s.
- Eager-launch jitter is BANKED AS A NAMED FLOOR (13.8 ms/win at ~4,070
  launches/win, launch-rate-insensitive; the lever is prefill capture,
  future work).
- Credit: parts of the prefill kernel line trace inspiration to Marco's
  MIT-licensed GB10 fork
  ([DS4-GB10-GX10-DSpark-CUDA](https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA));
  ported pieces carry commit credit and Portions-Copyright headers
  in-tree.

**Engine — deep substrate:**

- **Exact mxf4 scorer select at all depths** (`3af807b` + follow-ups):
  the indexer scorer's mxf4 selection chain is exact at every context
  depth on the serving path.
- **Streaming top-512 at every depth** (`9922a98`): the topk dispatcher
  takes a capture-stable streaming tier for all widths and both regimes
  above 8192 compressed rows; the >8192-row CUDA-graph capture
  exclusion is LIFTED. Deep decode at 240K: **−4.7% per verify step**
  (132.1 vs 138.5 ms), −10.9% ms/tok on the gated leg. The legacy
  chunked tree survives only as a forensic escape
  (`DS4_CUDA_NO_TOPK_STREAM`); an in-tree dual-run instrument
  (`DS4_CUDA_TOPK_STREAM_VERIFY`) byte-compared stream vs tree across
  ~5,600 live launches: zero diffs.
- **Boot prewarm** (`d872609`) + **quality/test hardening** (`07dea00`):
  goldens re-based cross-box (abs=0), per-step vector exclusion and a
  golden record mode in the tests, build-config guard
  (`.ds4-cuda-config.mk`).
- **Quench recal for the 0731 identity**: `dspark_shadow_guard`
  2.10 → **2.16** (same-day 12k pair, 166-step sample: plain 50.5
  ms/step, spec 109.3 → C 2.165; spec worth 1.42× at 12k). The deep
  break-even ramp (measured ~2.37 at 240K) remains open, tracked for
  v0.6. Gates: teb fast 86 (top of band; score and spec-hit totals
  identical to the 2.10 control) and think 83 in band, counters clean.

**Model — DeepSeek-V4-Flash-0731 refresh:**

- Ship quant: `...chat-v2-imatrix-0731.gguf` (86.7 GiB, published
  upstream — same recipe the fork's baselines are stamped on). Perf
  parity with the old quant confirmed by ABBA (+0.07%/+0.15%).
- Quality re-base on the identical harness/corpus (2,248 items, zero
  request errors; frozen caps, with the June cap-correction addendum
  repeated — both bases quoted per the reporting convention):
  **MMLU 79.5% vs 63.5** (the 2-bit-expert knowledge-recall weakness
  largely closed) · GSM8K 96.4 frozen / 96.8 corrected (band) ·
  HumanEval 88.4 frozen / **89.0 corrected** (above the old 88.4) ·
  MBPP 90.0 (exact) · **needle 70/70** (baseline-exact through 130K) ·
  think-GSM8K 39/40 · IFEval strict 74.3 frozen / **82.6 corrected**
  vs the old 83.4/86.9. The IFEval frozen drop is mostly a cap
  artifact: 0731 writes ~30% longer on open instructions and hit the
  frozen 768-token cap 2× as often (177/541 vs 85); 143 of 180 capped
  items pass strict when allowed to finish. The residual ~−4 pts
  corrected-vs-corrected is a real 0731 style delta (longer, looser
  instruction adherence), model-side, not engine.
- **MTP retired for 0731**: the upstream checkpoint replaced the
  single-block MTP module with the DSpark stages; there is no 0731 MTP
  head. Speculation is DSpark-only; the ds4-on-spark launcher and
  installer refuse the legacy-MTP × 0731 pairing and manage the weight
  upgrade (optional old-weight removal, prompted).
- **0731 DSpark drafter** re-extracted from the new checkpoint
  (Q2K-Q8, 6.97 GB): accept 73.7% / 3.08 tok/step at 12k — parity
  with the old pairing's band. Published at
  `bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF` as
  `DSpark-drafter-Q2K-Q8-0731.gguf` (the installer's default fetch).
- Model-behavior notes: 0731 answers deep-context prompts tersely
  (deep gates' long-generation samples shrank 512 → 8 tokens), and the
  tool-eval safety surface moved (new advisory TC-58; the old model's
  known TC-60 no longer fires).

**Known issue (fix scheduled v0.5.1):** the demand-mapped comp/index
cache pool is grow-only; long-running agentic serving at deep ctx can
walk banks to their maximum extent and squeeze the weight page cache
(field-reported throughput cliff). Workaround: pin the pool with
`DS4_BATCH_VMM_BUDGET_MB` (admissions beyond it reject cleanly). The
v0.5.1 trim-on-evict releases evicted banks' pages under pressure.

Also in this release: **adaptive side-model residency** (`6f59a16`) —
the server mlocks the drafter at load and a dedicated watchdog thread
releases all locks one-way under memory pressure. Post-ship A/B
measurement banked ZERO speed win (two ABBA nulls with bitwise-identical
speculation counters; the hold zone and the refault-tax zone are
disjoint on 121 GiB unified memory), so the mechanism is scheduled for
removal in v0.5.1 alongside trim-on-evict — the wedge-forensics and
capacity laws it produced are the keeper.

## v0.4.2 — 2026-07-24

Community fix: thinking-mode conversations now reuse KV on the
continuous path. Contributed by [@fabiopili](https://github.com/fabiopili)
in [#4](https://github.com/Entrpi/ds4/pull/4) — an excellent two-symptom
diagnosis and a fix that mirrors the session path's own design.

- **The bug**: `cont_warm_retire()` bailed on every thinking-mode row,
  so thinking conversations never built warm records. Two symptoms:
  every turn cold-re-prefilled the full history (measured 40 s vs 0.8 s
  on turn 2 of a 29K-token thinking agent preamble), and thinking banks
  never persisted to the disk KV tier (`kv_cache_store_bank` requires a
  warm-valid bank) — the root cause of
  [ds4-on-spark#4](https://github.com/Entrpi/ds4-on-spark/issues/4).
- **The fix**: port the session path's visible-transcript checkpoint
  key to the continuous path. The warm record keys on the rendered
  visible transcript (`</think>` + content + end marker for toolless
  turns; verbatim reasoning replay under tool context), which is
  byte-identical to what the prompt renderer emits on the next turn.
  Safety is by construction: a full record match reuses the bank's
  exact committed tokens under the engine's prefix validation, and
  partial cuts are token-LCP against committed history, so a cut can
  never land inside hidden reasoning. Bonus (by design): a
  `deepseek-chat` continuation of a `deepseek-reasoner` conversation
  byte-matches the thinking bank's visible record and reuses it.
- **Gates**: deep 240K all-stages pass, turn 2 **54.2 ms/tok** at 2.69
  tokens/verify-step with a warm-in-place reuse of the full 240K prefix
  (`warm admit cached=240156`; v0.4.1 stamp 57.3 at 2.76); deep 12K
  36.9 ms/tok at 2.95 (v0.4.1: 36.6 at 2.95), reuse + needles exact,
  counters clean. tool-eval-bench fast: score 88 ×5 runs, zero
  failures/serial/rejects/illegal ×5, admit streams byte-identical
  across runs and emit/draft totals conserved (19453/16206); spec-step
  tiling wobbles ±1 counter with arrival timing — a same-day control
  on the v0.4.1 binary bounded the class, and the non-thinking retire
  path is byte-identical code to v0.4.1. Thinking gate on the merged
  tree (DGX Spark, 4 legs): toolless turn-2 reuse (`fork admit
  cached=4140`), preserved-reasoning reuse (cached=4473), non-thinking
  control, and clean-SIGTERM persist (12 bank lines, 6 disk files,
  previously always empty for thinking) with reboot restore (`bank
  restore hit tokens=4232 load=159.5 ms`) and post-restore reuse —
  plus a needle planted in the reused prefix answered exactly on every
  leg. make test tensor equivalence: all-zero deltas.

## v0.4.1 — 2026-07-22

The quench recalibration patch. The terminal yield-quench controller's
break-even guard was still the v0.1.1 calibration (2.22, from a measured
spec-step cost C=2.17); v0.4's substrate work cut the verify cost to a
measured C of 2.03–2.08 on the 2k–64k band (2.40/2.38 at 240K/515K), so
the shipped guard sat 8% above break-even and terminally quenched
winners. One constant changes: `dspark_shadow_guard` 2.22 → **2.10**.

- **Method** (tools tracked in-repo): `DS4_DSPARK_TRACE=1` always-spec
  collections on four shapes (W&P and code low bands; 240K and 515K
  turn-2 via `deep_ctx_gate.sh`, which gains a `SPEC=0` plain-reference
  knob), replayed with `tools/dspark_trace_replay.py` (validate: 74/74
  engine SHADOW lines bit-match the replayer). Break-even C is
  identity-calibrated per band: C = geo(yield) / geo(measured same-run
  speedup). Candidate families swept under per-request-cost economics:
  flat guards, depth-ramped guard(pos0), non-terminal re-arm. Re-arm
  LOSES even at an optimistic zero-cost bound (floor 0.926); the depth
  ramp ties flat 2.10 at Δ2e-4 and is deferred (reopening trigger:
  observed deep content yielding between the guard and C_deep ≈ 2.4).
- **Measured** (GB10, fresh same-day plain reference both sides):
  code-corpus band geomean vs plain 1.084 → **1.103**; adversarial W&P
  band 1.02 → **1.044**; deep stamps 12K **36.6 ms/tok** at 2.95
  tokens/verify-step (identical to v0.4.0), 240K **57.3** at 2.76,
  515K **59.9** at 2.79 (v0.4.0: 62.1 at 2.75) — the deep gate corpora
  yield 2.6–2.9, above either guard, so no deep behavior change;
  forced-quench identity 1.0014 (12K ABBA) and 1.004 (41K); make test
  tensor equivalence all-zero deltas.
- **Floor honesty** (measured; replaces the "~0.96 floor" line): the
  quench worst case is pre-quench learning debt — minev (4) speculative
  steps times the window's yield deficit — which lands short (tg 128)
  adversarial generations at 0.93–0.97x plain depending on the draw,
  under ANY guard at or above break-even (the old guard 2.22 measured
  0.932 on 2026-07-22 draws at the very point where v0.4.0's release
  leg drew 0.97). Post-quench serving is identical to plain (the
  identity above). The README now states the mechanism and the typical
  range instead of a single draw.
- tool-eval-bench fast ×2: score 88 both runs (previous band 81–86;
  the pair is bit-identical — spec counters 14173/200 twice). Counters
  legitimately move off v0.4.0's 13659/201: the recalibrated guard
  retains more speculation.

## v0.4.0 — 2026-07-21

The deep-decode substrate release. Serving decode over deep
conversations drops 27–35% release-over-release (240K: 76.3 → **55.8
ms/tok**, 515K: 95.2 → **62.1 ms/tok**, turn-2 512-token generations
at 2.57/2.75 tokens per verify step; 12K stamps 36.6 ms/tok), the
static 64K depth gate on speculative decode is gone, and a
community-contributed server fix reaps queued requests whose client
disconnected before any prefill is spent. Quality held at the release
battery: evals band-exact vs v0.3.0 (HumanEval 149/164 identical,
IFEval strict 453 vs 443, needles 70/70), tool-eval-bench 86 twice
with bit-identical engine counters, tensor equivalence all-zero.

- **Head-group flash-decode for dense mixed attention** (`17a7d76`).
  The deep-decode depth tax was the ratio-128 mixed-attention family:
  eight heads now share one f32-staged KV tile (fp8 decoded once per
  head-group instead of once per head), with q held in registers,
  warp-per-head dots, online-softmax partials over row chunks, and a
  fixed-order combine. Isolated width-5 verify launches went 1876 →
  391 µs. Escape: `DS4_CUDA_NO_ATTN_HG`.
- **Indexed-path gather flash-decode at serving widths** (`e0ed742`).
  The same head-group idea for indexed attention: 12k 42.6 → 37.7,
  240K 66.2 → 64.4, 515K 73.5 → 71.5 ms/tok at landing time.
- **Speculative decode armed at every depth** (`b5a9ab1`). The static
  64K KV-depth gate defaults off; the terminal yield-quench
  controller (v0.2) is now the only governor of when drafting pays.
  240K 64.4 → 59.4, 515K 71.5 → 65.5 ms/tok at landing time. Setting
  `DS4_DSPARK_MAX_KV>0` restores a hard cap.
- **Dense verify tier: aligned Q8_0 at widths 1–8** (`7ffe821`). The
  aligned SoA dense artifacts previously served only single-token
  decode; verify steps fell back to raw mmvq (~90–200 GB/s). A
  width-templated kernel reads the aligned stream once per row with
  per-token accumulator columns: 240K 59.4 → 55.2 ms/tok. Escape:
  `DS4_CUDA_NO_Q8_ALIGNED_NC`.
- **Indexer scorer token-loop** (`18b40d7`). The WMMA scorer launched
  one CTA per (row-tile, token), re-staging the same K tile per
  verify token; one CTA now stages K once per sequence run and loops
  the tokens in-CTA, bitwise-identical to the per-token kernel. 240K
  55.2 → 50.6, 515K 66.1 → 61.7 ms/tok at landing time. Escape:
  `DS4_CUDA_NO_IDX_V5E`.
- **MoE gate_up first-owner expert dedup at verify widths**
  (`6e27e1e`). Live routing shows ~40% expert overlap across a
  width-5 verify's 30 slots and the 124 MB working set defeats L2;
  the first CTA owning an expert now accumulates every matching slot
  as extra q8_1 columns, so duplicate expert weights are read from
  DRAM once. Bitwise-identical outputs; win is routing-dependent
  (deeper on prose/code). The sibling dedup for the q2k down
  projection was prototyped and measured net-negative (its 2.68
  MB/expert distinct set is already L2-absorbed) — not integrated,
  verdict recorded in the proto. Escape: `DS4_CUDA_NO_MOE_DEDUP`.
- **Server: reap queued requests whose client disconnected before
  admission** (`881bcca`, contributed — PR #3 by @guptaavi, from a
  production GB10 report). A client that disconnects while its
  request is still queued used to hold an admission slot invisibly
  and then burn a full prefill into a dead socket; under continuous
  batching the zombies compound while every metric reads healthy. A
  zero-byte `MSG_PEEK` probe (peer-FIN only — a slow but connected
  client can never be false-positived) now gates all three FIFO
  exits, and dead jobs finish through the existing failure machinery
  before any engine work: visible as `outcome=failed` in `/metrics`.
- **ds4-bench: fixed illegal access sweeping past 32K at large
  `--ctx-alloc`** (`9caaa50`, reported on the NVIDIA developer
  forum). The release frontier sweep runs 2048–65536 clean at
  `--ctx-alloc 131072`.
- README GB10 leader assets refreshed from this release's runs:
  frontier sweep CSV/chart (prefill peaks 817 t/s, 535 at 65K; gen
  21.6 → 16.7 t/s across the band) and CLI speed-table rows.



Deep-context serving gets a tensor-core scorer and a durable KV tier.
Batched deep decode is 13–21% faster (240K→515K context), deep
conversations now survive bank eviction and server restarts (an
80K-token resume takes ~3 s instead of ~2 minutes of re-prefill), and
disk checkpoints store packed rows natively at ~2.3× smaller — with
cross-config restores proven bitwise exact.

- **Durable pinned banks: continuous-batching KV survives eviction and
  restarts.** Until now only the serial session's KV could persist to
  `--kv-disk-dir`; a continuous-batching bank (the per-conversation KV a
  warm record points at) evaporated whenever its bank was recycled or
  the server restarted, and a deep conversation paid its full prefill
  again. Banks now serialize through the same walkers and the same wire
  format as serial checkpoints (a bank record IS a valid serial
  checkpoint — the payload-integrity gate leg restores one in
  serial-only mode and byte-compares two fresh-boot continuations).
  Policy, with no new knobs: deep records at or above the existing pin
  threshold persist when a foreign admit destroys their bank
  (`bank-evict`); every valid record persists at graceful shutdown
  (`bank-shutdown`). At admission, a request no live record serves is
  checked against the disk tier and restored directly into a free bank
  — preferring no-value banks, with a deep-over-shallow displacement
  rule mirroring the eviction tiers so churn can't lock deep trunks out
  — after which the normal warm matching, engine-side frontier
  validation, and suffix prefill proceed exactly as for a live record.
  Restores honor the packed-native v3 format (packed primaries upload
  mirrors verbatim; write-dead F32 pages stay unmapped). The
  persist/restore runs synchronously in the admission path (seconds at
  deep contexts, replacing minutes of re-prefill); asynchronous staging
  is future work, as is carrying tool-call maps through bank restores
  (a post-restart tool-turn re-render currently degrades to a cold
  prefill, never to wrong output). New standing gate:
  `speed-bench/bank_persist_gate.sh` — which also flushed out a latent
  serial-mode bug on its first deep serial restore: the fp8 predecode
  scratch resized in single-row steps as the compressed count grew,
  eventually attempting an allocation inside an active capture window
  and killing generation a few tokens in. It is now sized once by the
  per-layer compressed cap (fixed in the preceding commit).

- **Batched-decode indexer scoring rewritten on tensor cores — deep
  decode ~13% faster.** The multiseq scorer (every continuous-batching
  decode step, all widths including width 1) was a scalar
  per-(row, token) kernel that was latency-bound, not bandwidth-bound:
  33 block-wide barriers and a 32 KB q re-read per compressed row put
  its cost at ~16 ms/step at 240K context, and the fp4 mirror's 6.4×
  byte reduction moved nothing. The new kernel stages 32-row tiles of K
  as unscaled e2m1 levels (always fp16-exact; the scaled values
  underflow fp16 at deep-context block scales, which is what rules out
  a naive half conversion) and MMAs them against per-warp q tiles,
  folding the F32 block scales back per 32-wide chunk — chunk width
  equals scale-block width, so the fold is exact. Q levels divide by
  the exact commit scale via a new scale-only emit from the indexer-Q
  QAT (1 KiB/token); the F32-primary config re-derives K block scales
  in-kernel with the emit QAT's own derivation, so fp4-primary and
  F32-primary configs stay bit-identical on the same binary. Measured:
  scorer 744→188 µs/launch at the 240K shape (3.97×, .15 proto ladder,
  top-k flips 0/2048 on every leg, 4-token batch ≡ single-token
  launches bitwise), 240K turn-2 deep decode 87.6→76.3 ms/tok
  (−12.9%), 12k turn-2 unchanged in its 44–48 band, needles exact at
  12k/240K/250K/500K. Wins at every tested shape (43→21 µs even at 12k
  context), so dispatch is unconditional. Serial (non-batched) decode
  keeps the scalar kernel: serial refuses deep contexts by design, and
  at shallow depths the scorer is under 1 ms/step.

- **Disk-KV payloads store packed rows natively (format v3).** Packed-
  primary sessions serialize the fp8/fp4 mirror codes+scales verbatim
  instead of dequant-expanding to F32: checkpoints shrink ~2.3× at 16K
  context (263→115 MB, growing toward ~3× at depth as compressed rows
  dominate), saves drop the expansion pass, and a packed-primary restore
  uploads the codes verbatim — no re-encode, and the write-dead F32
  pages stay unmapped. Cross-config restores remain exact in every
  direction (an F32-primary reader dequant-expands packed payloads
  through the emit scratch; a packed reader of an F32-row payload
  re-encodes as before), v2 payloads remain readable forever, and the
  kv_crossmode gate proves the full 6-leg matrix byte-identical. Found
  and fixed along the way: the e4m3 decode table was only initialized
  when `DS4_CUDA_FP8_KV` was enabled, so an F32-primary server
  dequanting packed codes read an all-zeros table (every non-RoPE lane
  restored as 0.0) — it is now initialized unconditionally.

- **Disk-KV restores into packed-primary servers are exact and enabled
  again.** v0.2.4 refused them because the restore-time mirror re-encode
  measurably drifted; the mechanism is now root-caused and fixed. The
  quantizer scale derivation `exp2f(ceilf(log2f(amax/T)))` is wrong at
  exact power-of-two ratios under `--use_fast_math`: `lg2.approx.f32`
  errs one-sided above at `2^-n`, `ceilf` rounds the exact negative
  integer up, and the derived scale doubles. A live encode essentially
  never presents an exactly-pow2 ratio, but a re-encode of committed
  values does routinely (a committed block max is codec level × pow2
  scale) — with the doubled scale, bottom-of-grid values fell off
  (measured: 10.5% of fp4 indexer lanes, including level-1 values
  collapsing to zero; a thin deep-subnormal fp8 tail). All QAT scale
  sites (six CUDA kernels, the CPU reference, three Metal kernels) now
  share an exact frexpf/ldexpf derivation, and codec sign tests use
  `signbit` so `-0.0` survives the round trip bitwise. Re-encode is now
  bitwise idempotent for any on-grid committed input — including
  checkpoints written by older binaries. Repro + fix proof:
  `cuda/mmq/test/proto_kv_reencode_idem.cu` (0 value mismatches / 84M
  lanes on the fixed path); end-to-end proof: `kv_crossmode_gate.sh`
  legs 3/5 now assert packed restores hit and byte-match F32 restores.
  The `DS4_KV_DISK_PACKED_RESTORE` investigation override is gone.
  Note: absolute speculative-counter values shift vs older binaries
  (committed values change in the boundary band where the old formula
  was wrong); same-binary packed-vs-F32 identity is unchanged.

## v0.2.4 — 2026-07-18

Packed FP8/FP4 compressed KV becomes the primary storage. Deep-context
decode gets 19–26% faster, every context size fits in ~3× less KV memory,
and the change is bit-lossless. The MTP head is no longer loaded when a
DSpark drafter is armed.

- **MTP-droppable is now the launch default.** With a DSpark drafter
  armed, the MTP head is fully shadowed: teb fast MTP-less is
  byte-identical in every speculative counter (13577/197) with a
  slightly better wall, and back-to-back 240K-deep stamps read 86.9
  (MTP-less) vs 87.2 ms/tok (full stack) — equal or better everywhere,
  for ~3.55 GiB of weights plus spec scratch back. The launch defaults
  therefore skip the MTP auto-attach beside an armed drafter and say so
  on the boot line (`mtp=dropped`). An explicit `--mtp` always wins,
  `--preset spark` still demands the full stack, and disarming the
  drafter (`--no-dspark`, `DS4_CONT_DSPARK=0`) restores the MTP
  auto-attach — MTP-2 remains the fallback speculation when no drafter
  is present. The `ds4-on-spark` wrapper no longer passes `--mtp` in its
  full-stack mode.

- **FP8 comp-KV + FP4 indexer primaries default ON.** The attention
  compressed-KV row stores the model's own e4m3-quantized values as 448
  1-byte codes + F32 rotary tail (704+28 B vs 2048 B F32); the indexer row
  packs e2m1 nibbles with per-32-lane scales (64+16 B vs 512 B). Because
  the F32 caches only ever held model-quantized values at F32 width, the
  packed forms are bit-lossless: teb temp-0 seed-7 speculative counters
  are byte-identical to F32 on every leg (crash/fast/think). The F32 rows
  go write-dead inside the VMM demand-mapped slabs, so their pages never
  become resident — that is where the memory comes back.
  `DS4_CUDA_FP8_KV=0` / `DS4_CUDA_FP4_INDEX=0` restore F32-primary.
- **Deep-context decode wins** (GB10 sm_121, 512-token turn-2 stamps,
  fresh boots): 120.9 ms/tok at 516K depth vs 162.7 F32 (−26%); 87.6 vs
  107.4 at 240K (−19%); +5.6 GiB MemAvailable after the 766K-token
  charter gate (needles exact to 518K both configs). The same charter
  shape at F32 now sits on a capacity cliff — on a degraded box it takes
  a clean 503 budget reject where the packed primaries pass with >6 GiB
  spare, so the flip is robustness as much as reach. Kernel levers that
  paid for it: the e4m3 decode table moved off `__constant__` (divergent
  per-code indexing serialized up to 32-way replays), address hoisting on
  all FP8 read sites, and a pair-lane layout (uchar2 codes + shared scale
  reads, quad dot) in both the dense and indexed decode-mixed kernels.
- **Shallow contexts carry a ≤2% decode tax** (12k: ~+2% ABBA-clean; teb
  mid-ctx wall +2.2%), the named floor being the in-kernel scalar FP8 dot
  kept for spec-verify counter identity plus attribution noise; prefill
  is flat at both ends. Ledgers and the elimination lever live in
  `local/docs/briefs/brief-kv-efficiency-arc.md`.
- **VMM-availability guard**: the primaries refuse to engage — loud boot
  line — when the batch comp slabs cannot be VMM demand-mapped (device
  without VMM support, `DS4_BATCH_VMM_COMP=0`, or slab-poison
  diagnostics). Eager F32+packed would be strictly worse than F32-only,
  so the server stays F32-primary there. A/B recipe note: an empty env is
  no longer the F32 control — pass explicit `=0`s.
- **Disk-KV checkpoints are now safe across storage modes** (found by the
  new `speed-bench/kv_crossmode_gate.sh`). The local session serializer
  was missing the packed-row expansion its distributed sibling got in the
  P2 era, so a packed-primary server checkpointed the write-dead F32 comp
  rows — an F32 boot restoring such a file produced visibly deranged
  output. Saves now expand the packed rows to their exact F32 values
  (file format unchanged); gate-proven by F32 boots restoring
  packed-written checkpoints byte-identically to F32-written ones.
  Restores *into* a packed-primary server are refused with a loud
  warning for now — the restore-time mirror re-encode is measurably not
  bit-exact yet, and a refused hit costs one prefill where a drifting
  hit silently changes output. Exact packed restore (persist
  codes+scales, or prove re-encode idempotence) is queued for v0.3;
  `DS4_KV_DISK_PACKED_RESTORE=1` overrides for investigation. Disk KV is
  a serial-path feature — continuous-batch serving is unaffected.
- **Counter-identity made structural.** The fp8-vs-F32 byte-equality of
  teb temp-0 speculative counters — the arc's losslessness tripwire —
  turned out to be a compiled-form coincidence: after the decode-kernel
  rewrites, think-leg `spec_hits` drifted by single digits in **both**
  configs (F32 moved −1 with its source untouched; fast-math is free to
  re-schedule the dot chains on any recompile) while every transcript
  stayed byte-identical, i.e. the drift lives entirely at draft-accept
  margins the verifier corrects. The config-branched comp dot chains in
  the decode-mixed and indexed-mixed kernels (fp8 *and* F32 branches) are
  now pinned to one in-order chain with `__fmaf_rn`/`__fmul_rn`, so
  same-binary cross-config counter identity no longer depends on
  compiler mood. Verified: fp8 and F32 think legs land identical
  `spec_hits` on the pinned binary.

## v0.2.3 — 2026-07-17

One-command serving: the launch defaults move into the engine binary.

- **Launch defaults: `ds4-server -c N` boots the full stack on a standard
  install.** With `-m` omitted the server resolves the base model from
  `$DS4_GGUF_DIR` (default `~/gguf`); when the MTP head and/or DSpark
  drafter GGUFs sit beside the base model and `--mtp`/`--dspark` are not
  given, they are attached and MTP-2 + DSpark speculative decode is
  enabled. Every auto choice is reported on one boot line — never silent.
  Explicit flags and env (`DS4_CONT_MTP_MODE`, `DS4_CONT_DSPARK`,
  `DS4_DSPARK_MODEL`) always win; file names follow the `ds4-serve`
  wrapper's env (`GGUF_FILE`/`MTP_FILE`/`DSPARK_FILE`). New flags:
  `--dspark FILE` (CLI form of `DS4_DSPARK_MODEL`), `--preset spark`
  (require the full stack, fail loudly if any piece is missing),
  `--no-mtp` / `--no-dspark` / `--no-spec` (per-component opt-outs).
  `DS4_CONT_DSPARK=0` (or empty) now reads as OFF — it was
  presence-tested before, so `=0` counter-intuitively armed the drafter.
  Gate scripts pass `--no-mtp` on their `MTP=""` legs so MTP-droppable
  legs stay genuinely MTP-less; the serial-path repro boots `--no-spec`.
  The `ds4-on-spark` v0.2.3 pin updates its `ds4-serve` wrapper to
  forward `--no-spec`/`--no-dspark` to the server (its wrapper-level
  downgrade flags previously relied on the server not auto-detecting).
  Gated: `launch_defaults_gate.sh` (zero-config / `--no-spec` /
  `--preset spark` legs) ALL PASS on GB10 sm_121 with live speculative
  engagement, plus teb on the release binary — crash 83+100, fast 86,
  think 83 (band 81–86), 0 serial starts, 0 admission rejects.

## v0.2.2 — 2026-07-16

Closes a silent performance-tier cliff between weight-server and standalone
boots — found chasing a latency-table discrepancy, disclosed on the
announcement thread the same day — plus a request-compatibility alias.

- **Every boot now builds the aligned fast-path artifacts** (`73c9727`).
  The fast decode and prefill dispatches (aligned-SoA D2R tiers) read
  derived repack artifacts that only `ds4_weight_server` built, so a
  standalone (self-load) boot silently fell to the raw-layout tier:
  decode 13.9 vs 17.8 tok/s p50, prefill 488 vs 853 tok/s, TTFT roughly
  doubled at the pp≈2048/tg=256 bench shape — with zero log tells. The
  one-command installer boots exactly that way. The repack builders now
  live in one shared library (`cuda/mmq/ds4_repack.{h,cu}`) compiled into
  both the engine and the weight server; manifest-less boots build the
  same artifacts in-process (78.7 GiB in ~22–26 s on GB10) and register
  them through the same lookup the import path uses. Precedence: manifest
  import > in-process build > raw fallback. `DS4_CUDA_BUILD_ARTIFACTS=0`
  opts out of the boot-time build; `DS4_CUDA_NO_DERIVED_WEIGHTS` still
  disables derived artifacts entirely. Gated: 474/474 FNV-1a artifact
  bit-identity vs the weight-server build, byte-identical accept traces
  per tier pair, and decode/prefill/TTFT parity with the weight-server
  control (17.8 tok/s / 868 tok/s / 5.3 s) on a standalone boot. The
  deep-context gate re-stamped on the release binary also improved:
  deep decode 163 ms/tok at 517K (was ~177 on the raw tier) and the
  cold half-million-token admit ~33 min (was ~41).
- **The active perf tier is never silent again.** One canonical boot line
  (`built in-process` / `imported from weight server` / `none` plus the
  reason), `ds4_derived_artifacts{source=…}` and
  `ds4_derived_artifact_bytes` gauges on `/metrics`, and
  `artifact_source` in `/v1/stats`.
- **`enable_thinking` is accepted as a `think` alias** (`f090ed2`) — the
  Qwen/vLLM convention, honored at all three request-parse sites alongside
  the existing `think` and `thinking.type` forms.

## v0.2.1 — 2026-07-16

Serving observability plus a models-list fix, both requested by users on the
v0.2 announcement thread within a day of posting. No kernel or placement
changes; quality surfaces are untouched.

- **Observability: one metrics core, three porcelains** (`118592e`). A single
  `ds4_metrics` registry (relaxed-atomic counters plus a 60-second rolling
  window) feeds every user-facing surface; readers never take the generation
  lock, so metrics stay pollable even during a minutes-long deep-context
  prefill. The surfaces:
  - a **`timings` block next to `usage`** in every response (and on the final
    streaming event with `stream_options.include_usage`): TTFT, prefill
    tokens with the cached/computed split, prefill and decode tok/s, and
    speculative acceptance + tokens-per-step when DSpark is active.
    Inapplicable fields are omitted, not null.
  - **`GET /metrics`**: Prometheus text exposition (hand-rolled, no
    dependencies) — request outcomes, token totals, rolling decode tok/s,
    live banks, KV pages, admission classes, speculation and quench counters.
  - **`GET /v1/stats`**: human-readable sectioned status text (JSON with
    `Accept: application/json`); `watch curl` works as a status board.
  The release gates now assert health from `/metrics` counter deltas as
  primary, with the stderr greps retained as fallback for one release.
  Overhead gated at zero: fresh-boot A/B produced byte-identical accept
  traces at 51.3 vs 51.4 ms/tok.
- **`/v1/models` lists only the loaded model** (`74928f0`). The endpoint
  hardcoded both `deepseek-v4-flash` and `deepseek-v4-pro`, so a Flash-only
  box advertised two ids as if selectable when the `model` field never
  switches weights (it is a label plus the `deepseek-chat` /
  `deepseek-reasoner` thinking toggle). `GET /v1/models/{id}` stays
  permissive for both known ids.

## v0.2 — 2026-07-15

The robust-serving release. v0.1.1 was held back when our own tool-calling
gate exposed two ship-path CUDA crashes; v0.2 ships those fixes plus the two
serving capabilities the agent workloads actually needed — speculation for
thinking/tool traffic and deep-context capacity — behind standing release
gates that run on every ship candidate.

- **Crash class fixed: cont admission-chunk OOB mirror reads** (`f16820c`).
  On ≥128-row admission chunks the token-tile comp-mirror kernel eagerly
  decoded `[0, max-over-banks n_comp)` of a *shallow* bank's demand-mapped
  mirror — an unmapped READ (dead ctx / zeroed-value corruption) that was
  placement-deterministic, not a race. Fix: bank-true `n_comp` for the
  single-run path. Warm + partial-prefix admission defaults re-enabled.
  Companion hardening: in-flight work now drains before sticky-scratch
  growth frees (`86a25e0`).
- **Speculation for agents: DSML sampler override on the continuous path**
  (`353c749`). Tools + thinking (or any temperature) no longer fall off to
  the serial path: the per-token structural/payload sampler swap that
  tool-call grammar needs now runs inside continuous batching, so agent
  traffic rides cont+DSpark. Tool-eval-bench thinking leg: same score band,
  all-batched, −27 % wall clock; Hermes agent end-to-end: every generate leg
  speculative at 80–97 % accept, 3.4–4.5 tok/step, zero serial fallbacks.
  Lossless at any temperature (delta-proposal speculative sampling — the
  verify-row logits + the request's own sampler/RNG are the only token
  source).
- **`DS4_SERVER_DEFAULT_TEMP`** (`d32e9a6`): default temperature for requests
  that omit one (agent frameworks usually do); explicit temperatures are
  untouched.
- **Deep-context capacity: 766K tokens served concurrently on one GB10**
  (`010cc08`, `32e1dab`). Admission/placement fixes for the multi-agent deep
  shape — a 518K-token orchestrator + 248K subagent concurrently at ctx
  524288, needles exact to 518K, warm in-place turn-2 TTFT 1.2 s (vs ~41 min
  cold prefill), deep decode 146–177 ms/tok at 248–519K:
  - batch geometry now rejects configs exceeding the uint32 absolute-row ABI
    instead of wrapping (`010cc08`);
  - comp-cache page budget refreshes live at admission time when free memory
    has grown since boot (pinned deterministic via `DS4_BATCH_VMM_BUDGET_MB`);
  - `DS4_SERVER_PIN_MIN_TOKENS` (default 65536): warm records above the
    threshold form a pinned LRU tier — a deep orchestrator trunk is never
    evicted by short-lived tenants while shallow victims exist;
  - deep-trunk fork guard: warm *fork* placement is refused above the pin
    threshold (fork-by-copy re-maps the whole committed extent, ~10 MiB per
    1K tokens at F32 — a 518K fork projected 6.8 GiB); the in-place warm path
    rides existing pages instead. The same rule now covers *partial-prefix*
    forks, keyed on the cut extent: sequential unique deep prompts sharing a
    long prefix previously stacked a fresh multi-GiB bank per request until
    a scratch allocation aborted the server (observed at six ~130K banks
    under a pinned page budget); past the threshold the trunk is truncated
    in place instead of copied.
- **Standing release gates** (`749dada`, `83f3ae5`, plus this release's
  additions in `speed-bench/`): `teb_gates.sh` (69-scenario tool-calling
  crash/fast/think legs, band 81–86, health+engagement gated; opt-in
  hardmode 73/100, error-injection @20 % 82/100 — both twice — and pass^k
  trials: mean 84.7 ± 2.3, pass@k 81.2, pass^k 71.0), `deep_ctx_gate.sh` (the
  766K/518K capacity gate above, one boot, three stages; re-passed on the
  release binary: ~750K concurrent, needles exact, warm turn-2 TTFT 1.7 s),
  `bank_churn_soak.sh` (pinned deep trunk + 12 cycling shallow tenants:
  PASS, 41 rounds/61 min, needle miss 0, deep evictions 0, memory drift
  1.0 GiB), `needle_sweep.sh` (formal retrieval matrix: **20/20 exact** —
  10 depths × {248K, 519K actual tokens}, cold TTFT flat across depth at
  843 s / 2434 s, memory flat across the 9-hour run). Quality restamp vs
  the June baseline on the release binary: GSM8K 484/500, MMLU 364/570,
  HumanEval 149/164, MBPP 178/200, IFEval strict 442/541 (a same-day
  control build scored 443 — the engine moves 1 item of 541; June 451),
  needle 45/45 inline + 64k 15/15 + 128k 10/10. Tool-eval-bench on the
  release binary: fast 82 / think 83 (band 81–86), crash legs clean, and
  `--mtp`-less fast 82 — exact parity, MTP is genuinely droppable.

## v0.1.1 — 2026-07-13

Decode is now net-positive by default across content and depth: the terminal
yield quench floors low-acceptance requests at ~0.96× plain while the kv-depth
gate handles >64k, so served decode ≈ max(speculative, plain) everywhere.
Frontier chart: `speed-bench/v011_decode_overlay.svg` (W&P prose floor line +
C-source favorable line, both at the ship config).

- **DSpark terminal yield quench** (`DS4_DSPARK_QUENCH`, default ON; `=0` to
  disable): per-request cumulative-regret controller — every verify step,
  `debt += guard − tokens_committed` (guard 2.22 ≈ the measured 2.17
  plain-step cost of one spec step); once debt exceeds a 4-plain-step budget
  with the yield EWMA below guard, speculation turns off for the REST OF THAT
  REQUEST (terminal, reset at admit), riding the kv-gate's lossless per-bank
  nd=0 path. Calibrated offline on 60 traced requests; the naive zero-clamped
  debt variant was measured to false-quench long bursty winners and rejected.
  Gates: forced-quench identity 1.000× vs plain; gsm8k 117/120, mbpp 37/40
  through the full serving path; W&P frontier floor 0.72× → 0.96× vs plain with
  shallow wins (1.2–1.7× structured) intact; suite holds 0.99 of always-spec.
  Tunables `DS4_DSPARK_SHADOW_{GUARD,ALPHA,MINEV,BUDGET,CREDIT_CAP}`;
  `DS4_DSPARK_QUENCH_FORCE_STEP` for identity testing. Supersedes
  `DS4_DSPARK_ADAPT_GATE` when both are set.
- **DSpark per-step trace + offline policy replayer**
  (`DS4_DSPARK_TRACE=1` + `tools/dspark_trace_replay.py`): per-request
  per-step yield/comparisons/drafts/latency telemetry, validated to reproduce
  `CONT_MTP_ACCEPT` aggregates exactly; the replayer calibrates quench
  parameters against recorded traces (`validate` / `replay --grid` /
  `inspect` / `selftest`).
- **Q2K drafter is the ship default** (was Q4K): equal throughput and
  acceptance in A/B (accept ±3pp, mean within noise), 6.49 vs 10.71 GiB in the
  weight server — the freed 4.2 GiB removes the deep-context boot knife-edge —
  and required for ~1M-token KV.
- Rollback checkpoint capture now derives from actually-packed draft rows
  (skipped when a step packs no drafts, e.g. every request's first MTP step);
  the restore pass provably no-ops there, so no-draft steps match plain cost.
- **DSpark kv-depth auto-gate** (`DS4_DSPARK_MAX_KV`, default 65536, 0 = off):
  speculative decoding is auto-disabled per sequence once its kv frontier crosses
  the threshold — acceptance decays with depth while the multi-row verify forward's
  cost grows with kv, netting a loss at 64k+ on prose (0.75–0.90×). Gated banks
  decode plain (verify = 1 row, no draft/injection); lossless by construction.
  Default set by the 2026-07-11 probes: spec still wins at 49k on both prose
  (1.10–1.49×) and code (1.19×); raise further for code-heavy serving.
- **DSpark adaptive kv gate** (`DS4_DSPARK_ADAPT_GATE=1`, opt-in, experimental):
  replaces the static cutoff with a runtime measure-and-switch controller past
  `DS4_DSPARK_ADAPT_START` — times the settled mode, probes the alternative,
  keeps the faster with hysteresis, re-probing periodically. Correct decisions
  8/8 in probes; costs ~5–12% vs oracle-best in probe overhead, hence opt-in.
  Solo-stream only; ring injection stays on during spec-off windows so spec can
  re-enter safely. See `misc/cuda-env-vars.md`.
- Fix serial-path lazy graph alloc OOM under bank starvation (`1da9467`): cont
  token-id echo, session-graph fit gate (`DS4_SESSION_GRAPH_FIT`,
  `DS4_SESSION_GRAPH_HEADROOM_MB`), allocation early-bail.

## v0.1.0 — 2026-07-10

384 fork commits on `batched-serving`, released as branch `release/v0.1.0`.
Headline numbers measured on GB10 (DGX Spark class, sm_121),
DeepSeek-V4-Flash IQ2XXS-mixed GGUF; methodology in the release notes.

### Serving & API

- **Continuous batching** (`DS4_SERVER_CONTINUOUS=1`, default): mid-flight admit/evict,
  per-bank KV state, FCFS pending-prefill interleave (`DS4_CONT_PREFILL_CHUNK_LIVE`),
  chunked cold admission (~1.9× admit, `DS4_CONT_PREFILL_CHUNK=4096`).
- **Request coalescing** default-on for non-streaming groups (`b5c6a83`), budgeted by
  prompt+output token footprint (`9339ab2`).
- **Warm start & fork**: per-bank prefix warm start (TTFT ~7×), D2D bank-clone fork
  fan-out (N=4 TTFT ~49×), partial-prefix fork (`8a929a1`); victim order
  invalid > superseded > LRU.
- Stops + tool calls ride the batched path (tools batch greedy-only); OpenAI tool-argument
  streaming with incomplete-call rejection.
- Budget-computed bank fit from MemAvailable (not cudaMemGetInfo); KV ≈ 9.46 KiB/token +
  ~94 MiB bank floor; lazy single-session graph allocation (boot-time GPU footprint ~0).
- Batched forward scales 6.2× from batch 1→128; batchable stops/tools/MTP all supported.

### Prefill engine (CUDA) — 12k cold prefill 305 → ~800 tok/s

Each landing independently gated (bit-parity or value-parity → same-boot ABBA → nsys →
slice evals) and reversible by env switch:

- sm_121 native arch build (`98c55ad`, `make cuda-spark`) — 306→339
- mm_ids case-1 fast path + heads8 occupancy pin + SoA-direct mmq tile loaders
  (`86c5d4d`/`2e519d0`/`42cf8ca`) — 339→420
- Sanitize/rms_norm/hc_expand folds into GEMM activation converts (`6e415a7`/`a574241`/
  `57e0821`) — 420→446
- mm_ids W8192 smem-cliff kill, two-pass no-smem (`c9da2fa`; default chunk stays 4096)
- **D2R (decode-to-registers) MoE GEMM family** — direct-to-register dequant kernels
  reading the weight server's aligned SoA artifacts in place:
  - Q2_K down-GEMM (`49329aa`) — 427→493
  - IQ2_XXS gate/up pair GEMM (`e5668be`) — 493→557
  - Expert-major CTA schedule, L2-reuse grid order (`2e68f52`) — 674→768
  - Dense-Q8 16-warp m128n128 kernel on the kind-5 aligned artifact for shared-expert +
    q_b projections (`154174e`) — 779→800
- **Token-tile HMMA attention** (indexed prefill `47438d7` + decode-mixed `9de3044`),
  replacing heads8_online — 557→640
- Memset audit: blanket GEMM output zeroing dropped, bit-exact gated (`9adc3df`) — 640→678

### Decode engine (CUDA) — plain cont+capture w1 48.9 ms/tok @HEAD (C1-era 54.0); DSpark ship 3.02 tok/step

- M1/M2 fused decode kernels: aligned-SoA IQ2_XXS + Q8_0 decode paths, fused
  gate+up+SwiGLU, fused HC stage, fused router (top-6), fused compressor pair+store,
  fused QKV-post (head_rms+rope / kv-rope+fp8+store), q8 activation folds.
- Q2K moe-down aligned row-pair-SoA repack, bit-exact decode twin (`e221241`, default ON).
- **C1: per-layer CUDA-graph capture** of the batched decode step (`d8cf4f9`, default ON;
  45.1→37.8 ms/tok ship).
- C2: bank-agnostic cont graphs (state lanes, on-device bank resolve), multi-live
  PLAIN + VERIFY capture (`749a1e4`/`23dcb1f`/`46ab301`).
- C3: batched fused router/compressor/HC at small widths (`cd8ad1a`/`0a48aac`/`ee3da19`),
  indexer-producer gating (`e82cbc7`).
- Decode width tiers: NATIVE_F16 / MMVQ_DECODE_MAX_TOKENS=8 dispatch, expert-vec split
  for all decode widths, multi-column output_a widths 2–8.

### Speculative decoding

- **MTP verifier paths** (top-2 verify, fused two-token MoE down); mode-2 batched
  spec-decode opt-in (`DS4_CONT_MTP_MODE=2`).
- **DSpark block-draft spec decode, lossless** (`82b2622` + Phase D chain): on-device
  Markov refine, multi-seq inject with per-bank KV slabs, prefill-region injection,
  auto verify-depth, concurrency auto-gate (nlive≤1); ~2.0 tok/step at 85.7% accept
  on eval workloads.
- **WS-served drafter** (`d8cc99d`): 58.8→44.3 ms/tok in DSpark ship config at 72.3%
  accept; ships Q4K, with the Q2K drafter variant available (−4.2 GiB, 87% accept;
  required for ~1M-token KV).

### Weight server

- CUDA weight-server lifecycle with VMM backend: allocate/upload model ranges, broker
  fds to clients, direct-I/O uploads, scoped imports, manifest staleness rejection,
  ownership locks, telemetry (`f2f424e`…`e7f7ce1` chain).
- **Aligned-SoA repack artifacts** (iq2 + q8 default-on `6de508d`, q2k default-on
  `6f44a5e`): the artifacts both decode-vec and D2R GEMM kernels read in place.
- Parallel aligned-repack builders (WS boot tax 63→21 s, `802e4f3`).
- Drafter serving with never-split range plan.

### Long context & KV

- Compressed-KV storage tiers: FP8 codes primary (`74a617a`), FP4 e2m1 indexer
  primary (`e7c4826`) — bit-lossless vs F32 storage (caches hold model-quantized values).
- VMM demand-mapped comp/index slabs; 128k contexts served correctly (~4.9 GB/bank at
  139k ctx, max_seq fit-reduces); DSA prefill context-flat.
- Needle 8k–128k: 70/70 at the June baseline; HEAD re-stamp 45/45 + 15/15 + 10/10
  (128k tier at `63d9d5e`).

### Fixed

- **Token-tile attention launch failure at 94k+ context** (`63d9d5e`, found by the v0.1.0
  needle gate): union-builder smem opt-in now budgets static+dynamic; prompts in the
  ~94k-98k band no longer fail chunked prefill.
- **Silent truncation of coalesced batched generation** (`9339ab2`, found by the v0.1.0
  eval gate): batch ring now sized for the generation horizon; budget clamps log loudly;
  coalesce groups bounded by footprint.
- Cross-stream use-after-free of pool scratch (cont BOS-spam root, `66d6dc3`);
  mm_ids_helper warp race + DS4_CUDA_DETERMINISTIC (`491f584`); quantizer unwritten-tail
  nondeterminism; mmvq −1 router ids; ring-lane addressing by live-slot ordinal
  (`cd00600`); MTP validator F32 ffn_gate_inp; free-on-grow graph-cache invalidation;
  macOS Metal-stub link fixes (`26b8816`, `259075d`).

### Method & infrastructure

- Engine proof runner + weight-server proof flow; stdlib-only resumable eval harness
  (GSM8K/MMLU/HumanEval/MBPP/IFEval/needle, inline scoring, watchdog-supervised).
- Measurement discipline: same-boot ABBA with SM-clock medians for <5% deltas; proven
  path engagement + finish-reason shape asserts in eval gates; ncu counter law
  TF = 906·IPC/inst-per-MMA; negative results recorded in ledgers.
