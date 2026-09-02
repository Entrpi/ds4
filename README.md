# Entrpi/DwarfStar

**Entrpi/DwarfStar** (`ds4`) is a small native inference engine for **DeepSeek
V4 Flash**, with support for **DeepSeek V4 PRO** on very high-memory
machines. It is intentionally narrow: not a generic GGUF runner, not a
wrapper around another runtime, completely self-contained. It provides
DS4-specific loading, prompt rendering, tool calling, KV state handling
(RAM and on-disk), an HTTP server API, and an integrated coding agent,
plus tools for GGUF and imatrix generation and for quality and speed
testing.

This repository, [Entrpi/ds4](https://github.com/Entrpi/ds4), is a fork
of [antirez/ds4](https://github.com/antirez/ds4) (the original
DwarfStar) that has diverged nearly since the project's inception: it
builds the CUDA/Linux side into a **batched multi-request serving
engine**. The
reference machines are the DGX Spark (GB10, `sm_121`) and the RTX PRO
6000 Blackwell (`sm_120`). Upstream's heart is a single-user CLI/agent
engine, Metal first; the fork keeps all of that working. The fork story
and its measured results are in [About this fork](#about-this-fork)
near the end of this README. On a DGX Spark, the fastest path is the
packaged installer at
[Entrpi/ds4-on-spark](https://github.com/Entrpi/ds4-on-spark).

Backends:
* **NVIDIA CUDA** is the fork's optimization target, with special care for the DGX Spark.
* **Metal** (starting from MacBooks with 96 GB of RAM) is upstream's primary target. It is kept building and passing its vectors here; correctness on the fork's serving paths is community-maintained.
* **AMD ROCm** is only supported in the [rocm](https://github.com/antirez/ds4/tree/rocm) branch, rebased by the community as needed.

This project would not exist without **llama.cpp and GGML**, make sure
to read the acknowledgements section, a big thank you to Georgi Gerganov
and all the other contributors.

## Quick start

Download a model, build, serve. (Full detail in the sections below; on
a DGX Spark, [ds4-on-spark](https://github.com/Entrpi/ds4-on-spark)
does all of this, plus the DSpark drafter setup, with one command.)

```sh
./download_model.sh q2-imatrix    # 96/128 GB machines; see Model Weights
make cuda-spark                   # DGX Spark / GB10; plain make = macOS Metal
./ds4-server --host 0.0.0.0       # CUDA defaults: ctx 262144, banks sized from free memory
```

On a standard install that is the whole launch: the base model resolves
automatically, a speculative drafter sitting beside it is attached and
armed, and every automatic decision is stated on one boot line, never
silently. Since v0.6 unused context is demand-mapped, so a deep `-c`
costs almost nothing until a request actually uses it; the CUDA default
of 262144 exists so that a long agentic session fits a defaults boot,
and `-c` raises it (524288 is the installer default; the model's full
1M window at `-c 1048576` is proven with a 975k-token conversation;
see [Memory and capacity](#memory-and-capacity)).

Then talk to it with any OpenAI or Anthropic client:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello!"}]}'
```

Agent clients (Codex, Claude Code, opencode, Pi) are covered in
[Agent Client Usage](#agent-client-usage). The interactive CLI is
`./ds4` (see [CLI](#cli)), and `./ds4 --help` / `./ds4-server --help`
list every flag.

## Model Weights

This implementation only works with the DeepSeek V4 Flash and PRO GGUFs published for
this project. It is not a general GGUF loader, and arbitrary DeepSeek/GGUF files
will not have the tensor layout, quantization mix, metadata, or optional MTP
state expected by the engine. The 2 bit quantizations provided here are not
a joke: they behave well, work under coding agents, call tools in a reliable way.
The 2 bit quants use a very asymmetrical quantization: only the routed MoE
experts are quantized, up/gate at `IQ2_XXS`, down at `Q2_K`. They are the
majority of all the model space: the other components (shared experts,
projections, routing) are left untouched to guarantee quality.

Download one main model. **Prefer the imatrix versions.**

```sh
./download_model.sh q2-imatrix   # 96/128 GB RAM machines, imatrix-tuned q2
./download_model.sh q2-q4-imatrix  # 96/128 GB RAM machines, q2 with last 6 layers q4
./download_model.sh q4-imatrix   # >= 256 GB RAM machines, imatrix-tuned q4
./download_model.sh pro-imatrix  # 512 GB RAM machines, PRO imatrix quant
```

Legacy GGUF files are still available if you specifically need the older
non-imatrix quants:

```sh
./download_model.sh q2           # 96/128 GB RAM machines, legacy non-imatrix
./download_model.sh q4           # >= 256 GB RAM machines, legacy non-imatrix
./download_model.sh pro          # 512 GB RAM machines, legacy non-imatrix PRO
```

The script downloads from `https://huggingface.co/antirez/deepseek-v4-gguf`,
stores files under `./gguf/`, resumes partial downloads with `curl -C -`, and
updates `./ds4flash.gguf` to point at the selected main model. The plain q2 XXS
weights are produced with the weights importance vector only, without an
imatrix. The imatrix variants are preferred.
Authentication is optional for public downloads, but `--token TOKEN`,
`HF_TOKEN`, or the local Hugging Face token cache are used when present.

If you want to regenerate GGUF files or collect a new imatrix, see
[gguf-tools/README.md](gguf-tools/README.md). Those tools are meant for offline
model-building work and can take a long time on the full DeepSeek V4 Flash
weights. Flash GGUF generation is supported by the local tools. PRO GGUF
production currently still depends on the external `llama.cpp`-based workflow;
native tooling can be added later.

`./download_model.sh mtp` fetches the optional speculative decoding support
GGUF for Flash. It can be used with q2-imatrix, q4-imatrix, q2, and q4, but must be
enabled explicitly with `--mtp`. The current MTP/speculative decoding path is
still experimental: it is correctness-gated and currently provides at most a
slight speedup, not a meaningful generation-speed win.

Then build:

```sh
make                  # macOS Metal
make cuda-spark       # Linux CUDA, DGX Spark / GB10
make cuda-generic     # Linux CUDA, other local CUDA GPUs
make cpu              # CPU-only diagnostics build
```

`./ds4flash.gguf` is the default model path used by both binaries. Pass `-m` to
select another supported GGUF from `./gguf/`. Run `./ds4 --help` and
`./ds4-server --help` for the full flag list.

Building in a container? Use [docker/Dockerfile](docker/Dockerfile) as the
reference: it builds with `make cuda-spark` (on a GB10 a generic
`make cuda CUDA_ARCH=...` build serves at a fraction of the speed) and its
header documents the run flags — in particular, never give the container a
memory limit, because the mmap'd weights live in the host page cache.

## Server

Start a local OpenAI/Anthropic-compatible server:

```sh
./ds4-server --ctx 524288 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Use `--chdir /path/to/ds4` when launching `ds4-server` from another directory,
so relative runtime files such as `metal/*.metal` resolve from the project tree.

The server keeps one mutable backend/KV checkpoint in memory,
so stateless clients that resend a longer version of the same prompt can reuse
the shared prefix instead of pre-filling from token zero.

Request parsing and sockets run in client threads. Since the fork's
v0.1.0 line, inference runs **continuous batching by default**:
concurrent requests are admitted mid-flight into per-request KV banks
and decoded together, with chunked prefill interleaved into live decode
(see [Memory and capacity](#memory-and-capacity) for how admissions are
funded). `DS4_SERVER_CONTINUOUS=0` restores the upstream serialized
behavior, where concurrent requests wait their turn on one live
graph/session.

Supported endpoints:

- `GET /v1/models` (lists the loaded model)
- `GET /v1/models/deepseek-v4-flash`
- `GET /v1/models/deepseek-v4-pro`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`
- `GET /metrics` (Prometheus text: request outcomes, token totals, rolling
  decode tok/s, live banks, admission classes, speculation counters; since
  v0.6.0 also the memory families — allocation census by class and domain,
  the availability observation with both raw estimates behind it, governor
  decisions per consumer, reclaim outcomes, and typed request rejections
  labelled by lane and reason)
- `GET /v1/stats` (human-readable status board; JSON with
  `Accept: application/json` — `watch -n2 curl -s :8000/v1/stats` works)

The Flash and PRO model endpoints are compatibility aliases. They both report
the model currently loaded from the GGUF passed with `-m`; the endpoint name does
not select a different model.

Every response also carries a `timings` block next to `usage` (TTFT, prefill
tokens with the cached split, prefill and decode tok/s, and speculative
acceptance when DSpark is active); streaming responses include it on the
final event when `stream_options.include_usage` is set.

`/v1/chat/completions` accepts the usual OpenAI-style `messages`,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`seed`, `stream`, `stream_options.include_usage`, `tools`, and `tool_choice`.
Tool schemas are rendered into DeepSeek's DSML tool format, and generated DSML
tool calls are mapped back to OpenAI tool calls.

`/v1/responses` accepts OpenAI Responses-style `input`, `instructions`,
`tools`, `tool_choice`, `max_output_tokens`, `temperature`, `top_p`, `stream`,
and `reasoning`. It is the preferred endpoint for Codex CLI. The server keeps
Responses continuations bound to live state when possible, and can fall back to
the same DSML rendering and KV prefix reuse used by chat completions.

`/v1/messages` is the Anthropic-compatible endpoint used by Claude Code style
clients. It accepts `system`, `messages`, `tools`, `tool_choice`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`, and thinking
controls. Tool uses are returned as Anthropic `tool_use` blocks.

Default sampled API generation uses `temperature=1`, `top_p=1`, and
`min_p=0.05`, so the default filter is relative probability rather than
nucleus mass. In thinking mode DS4 uses those fixed sampling defaults and
ignores client sampling knobs, matching DeepSeek's fixed-thinking API behavior.

The chat, Responses, and Anthropic endpoints support SSE streaming. In thinking
mode, reasoning is streamed in the native API shape instead of being mixed into
final text. OpenAI chat streaming
also streams tool calls as soon as the DSML invocation is recognized: the tool
header is sent first, then parameter bytes are forwarded as
`tool_calls[].function.arguments` deltas while generation continues. The
Anthropic endpoint streams thinking and text live, then emits structured
`tool_use` blocks when the generated tool block is complete.
The Responses endpoint streams the Responses event lifecycle expected by Codex,
including `response.output_text.delta`, function-call argument events, and
terminal `response.completed` / `response.incomplete` / `response.failed`
events.

Chat-completion SSE accepts a `return_token_ids: true` request field that
adds per-token IDs to each streamed delta, placed at the choice level to
match the wire shape vLLM and llama-benchy-style benchmark harnesses expect.
The emission limit is snapped to token boundaries so partial UTF-8 never
desynchronises the IDs from the text. Helpful when an external evaluation
loop needs raw token IDs alongside the rendered text.

For browser JavaScript clients served from another origin, start the server with
`--cors` to emit `Access-Control-Allow-*` headers. This only changes HTTP
headers; it does not expose the server on the LAN. Use `--host 0.0.0.0`
explicitly when remote machines should be able to connect.

### Tool call handling and canonicalization

DeepSeek V4 emits tool calls as [DSML text](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/README.md). Agent clients do not send that
same text back on the next request: they send normalized OpenAI/Anthropic JSON
tool-call objects. **If the server re-rendered those objects slightly
differently, the rendered byte prefix would no longer match the live KV
checkpoint** and the next turn would have to be rebuilt.

The first line of defense is exact replay. Every tool call gets an unguessable
API tool ID, and the server remembers `tool id -> exact sampled DSML block` in
a bounded in-memory map backed by radix trees. When the client later sends that
tool ID back, the prompt renderer uses the exact DSML bytes the model sampled,
not a freshly formatted approximation. This map can also be saved inside KV
cache files, so exact replay survives server restarts for cached histories.

**Canonicalization is only the backup path**. If the exact DSML block is missing,
or exact replay is disabled with `--disable-exact-dsml-tool-replay`, the server
renders a deterministic DSML form from the JSON tool object. After a tool-call
turn, it compares the live sampled token stream with the prompt that the next
client request will render. If needed, it rewrites the live checkpoint, or
falls back to an older disk KV snapshot and replays only the suffix. This keeps
the model continuation aligned with the stateless API transcript.

During generation, the server also treats DSML syntax differently from payload.
When the model is emitting stable protocol structure such as DSML tags,
parameter headers, JSON punctuation, or closing markers, sampling is forced to
`temperature=0` so the tool call stays parseable. This greedy mode does **not**
apply to argument payloads: `string=true` parameter bodies and JSON string
values, including file contents and edit text, use the request's normal sampling
settings. That separation is important: deterministic decoding is helpful for
syntax, but can create repeated text when applied to long code or file bodies.

### Replayed reasoning and agent-loop robustness

Deep agent loops on low-bit quants have a characteristic failure: past
roughly 65K tokens the model sometimes answers a tools-armed turn with a
prose completion report ("all tests pass, done") instead of a tool call —
measured at up to 50% of turns in the 70-80K band, always right after a
successful tool result. Agent harnesses reject those turns and their
retry rules can convert a few of them into an abandoned task with real
work discarded. Three knobs address it, one default and two optional
levers (measured on SWE-rebench-class workloads; receipts in the
changelog):

**`--tool-call-reminder on|off`, default on** (v0.6.5; env
`DS4_TOOL_CALL_REMINDER=0` disables) — the default fix. Past ~96KB of
rendered conversation (~30K+ tokens), every tool result carries a short
protocol reminder. At the exact captured slip states the reminder
measured 0/72 sampled slips vs 6/72 without, and it fixed the failing
agent task end to end (submitted and harness-resolved, zero slips) with
reasoning traces kept and no format deviation. Shallow conversations are
never touched, so chat-with-tools flows that legitimately answer in
prose after a tool result are unaffected, and the injection is
byte-stable across turns, so warm prefix reuse is unchanged. All three
API surfaces are covered, including the live tool-result continuation
fast lanes; the one exception is a Responses continuation that sends
only tool outputs without history, whose conversation depth is not
renderable, so it skips the reminder.

**`--reasoning-replay keep|drop`** (env `DS4_REASONING_REPLAY=drop`) —
an optional depth lever. Most OpenAI-style agent scaffolds echo each
assistant message back verbatim, including `reasoning_content`, and that
is what DeepSeek specifies for this model family: the V4 reference
encoding keeps reasoning for every turn whenever tools are present, and
DeepSeek's API requires the echo in tool loops (it returns 400 when
`reasoning_content` is not passed back). The default `keep` honors that
format, re-emitting the reasoning inside `<think>` blocks, which also
keeps the rendered prefix byte-aligned with the live KV (warm in-place
reuse, no re-prefill). llama-server's template default drops the echo
instead — a deviation from the reference format that runs the identical
conversation 16-28% shallower per turn, which delays the depth band
where slips live. `drop` reproduces that deviation opt-in, rendering
history assistant turns in the lean `</think>` replay form; the bank's
partial-prefix admission absorbs the divergence at the cost of a
one-turn-tail re-prefill (~270 tokens measured). An assistant turn being
continued (after the last user-like message) always keeps its reasoning.
With the default reminder on, most agent loops no longer need this.

**`--tool-slip-resample`** (env `DS4_TOOL_SLIP_RESAMPLE=1`, off by
default) — a second line of defense for slips that still get through.
A continuously-batched non-streaming chat turn that settles at
`finish=stop` with no tool calls is requeued once for a fresh draw
before anything reaches the client; the just-retired bank warm-admits
the full prompt, so the retry costs one generation. `length`/`error`
finishes, streaming turns, and the serial lane are never resampled.
Note the obvious disclosure, which applies to the default reminder as
well: these change benchmark behavior, so results should say which
knobs were on.

### Continuation registry and trust domain

For the Anthropic and Responses endpoints — whose protocols let a client send
an output-only follow-up (just the `tool_result` / `function_call_output`)
instead of replaying the whole conversation — the server keeps a continuation
registry: one record per tool-call turn, binding the turn's tool-call IDs to
the engine state that produced them (an engine-authoritative content
generation plus committed token frontier). A follow-up that references those
IDs is revalidated by pure equality at admission; if the state has moved on,
the server answers a native `409` asking for a full-history replay rather than
silently continuing from the wrong frontier. Records that lose their live
state remain replayable (exact sampled DSML is retained under a bounded LRU),
and short grace/TTL windows (`DS4_CONT_GRACE_S`, `DS4_CONT_TTL_S`,
`DS4_CONT_PIN_DEADLINE_S`) protect a just-published turn from being evicted
before its follow-up arrives.

**Streaming tool turns ride the batched lane.** Anthropic and Responses
STREAMING tool requests are served on the continuous-batching lane: the tool
turn's registry record is owned by its KV bank, and an output-only follow-up
continues that bank in place after the same generation/frontier equality
check. Buffered tool requests deliberately stay on the serial lane — its
model-visible corrective retry (feeding a malformed tool call back to the
model as a tool error) has no per-row equivalent in the batched loop, and a
parse-time boolean cannot predict a semantic failure discovered after
generation. A mixed client that streams the tool turn but sends the follow-up
buffered gets the honest `409` + full-replay path. Per-surface kill switches
`DS4_SERVER_CONT_TOOLS_ANTHROPIC=0` / `DS4_SERVER_CONT_TOOLS_RESPONSES=0`
restore the previous all-serial tool behavior (kept for one release, like the
Inc 3 stateless switches they compose with).

**The whole server is one trust domain.** There is no tenant or auth
namespace: tool memory and the continuation registry are global, and any
client that knows (or guesses) a tool-call ID may continue that conversation
or shed other clients' work through the grace window. Tool-call IDs are minted
unguessable, but they travel in responses — do not expose one `ds4-server` to
mutually untrusted clients expecting isolation. Put an authenticating proxy in
front, or run one server per tenant, until an authenticated namespace exists.

Minimal OpenAI example:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

### Agent Client Usage

`ds4-server` can be used by local coding agents that speak OpenAI-compatible
chat completions. Start the server first, and set the client context limit no
higher than the `--ctx` value you started the server with:

```sh
./ds4-server --ctx 524288 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

For long agent loops on quantized weights: since v0.6.5 the deep
tool-protocol reminder is on by default (the measured fix for agent
tasks dying to prose slips at depth), and two optional levers remain,
`--reasoning-replay drop` (keep scaffold-echoed reasoning out of the
prompt; conversations run 16-28% shallower at the cost of llama.cpp-style
format deviation) and `--tool-slip-resample` (one retry when a
tools-armed turn still settles as prose). See
[Replayed reasoning and agent-loop robustness](#replayed-reasoning-and-agent-loop-robustness)
for the mechanics and measurements.

You can use larger context and larger cache if you wish. Full context of
1M tokens is going to use more or less 26GB of memory (compressed indexer
alone will be like 22GB), so configure a context which makes sense in
your system. With 128GB of RAM you would run the 2-bit quants, which are
already 81GB, 26GB are going to be likely too much, so a context window
of 100~300k tokens is wiser. However users reported being able to run 2bit
quants with 250k ctx window in a Macs with just 96GB of system memory: make sure
to kill processes that use too much memory, if you plan doing so ;)

On this fork's CUDA server, v0.6 made the memory side of that choice
easy: unused context is demand-mapped and each admission is charged
only what it will actually use, so a deep `--ctx` no longer pre-pays
anything. See [Memory and capacity](#memory-and-capacity) for the
knobs and the proven numbers.

The `384000` output limit below avoids token caps since the model is able
to generate very long replies otherwise (up to 384k tokens). The server
still stops when the configured context window is full.

For **opencode**, add a provider and agent entry to
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ds4": {
      "name": "ds4.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "limit": {
            "context": 524288,
            "output": 384000
          }
        }
      }
    }
  },
  "agent": {
    "ds4": {
      "description": "DeepSeek V4 Flash served by local ds4-server",
      "model": "ds4/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

For **Pi**, add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ds4": {
      "name": "ds4.c local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dsv4-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "deepseek",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "deepseek-v4-flash",
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 524288,
          "maxTokens": 384000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Optionally make it the default Pi model in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "ds4",
  "defaultModel": "deepseek-v4-flash"
}
```

For **Codex CLI**, use the Responses wire API:

```toml
[model_providers.ds4]
name = "DS4"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

Then run:

```sh
codex --model deepseek-v4-flash -c model_provider=ds4
```

For **Claude Code**, use the Anthropic-compatible endpoint. A wrapper like this
matches the local `~/bin/claude-ds4` setup:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${DS4_ANTHROPIC_BASE_URL:-http://127.0.0.1:8000}"
export ANTHROPIC_AUTH_TOKEN="${DS4_API_KEY:-dsv4-local}"
export ANTHROPIC_MODEL="deepseek-v4-flash"

export ANTHROPIC_CUSTOM_MODEL_OPTION="deepseek-v4-flash"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="DeepSeek V4 Flash local ds4"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="ds4.c local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Claude Code may send a large initial prompt, often around 25k tokens, before it
starts doing useful work. Keep `--kv-disk-dir` enabled: after the first expensive
prefill, the disk KV cache lets later continuations or restarted sessions reuse
the saved prefix instead of processing the whole prompt again.

## Memory and capacity

Since v0.6.0 one memory governor decides every allocation that can
grow. Engine boot, prewarm, the bank plan, the serial session, and the
per-call batch graph all ask the same evaluator, which weighs each ask
against one availability observation and one ledger of what every other
lane holds. Nothing is reserved up front: context lives in virtual
banks whose pages materialize only as requests fill them, each
admission is charged what it will actually use, and idle banks return
their pages to the pool after a couple of minutes of quiet. When an ask
truly does not fit, the refusal is typed with a reason that says
whether a retry can succeed, and counted per lane in `/metrics`. Before
refusing for lack of memory the engine first collects what it can
return: idle banks' pages via trim, then its own unused CUDA graph-pool
reserve (`DS4_MEM_OWN_TRIM=0` opts out).

Since v0.6.2 the account also proves itself. Every floor and margin in
the plan is derived from a measurement rather than a constant: the
bank count is priced from the live budget at boot, the anti-thrash
floor prices working sets at what they actually commit (not their
virtual extents, which overclaimed 4x at deep context), and the
planning headroom derives from the operator's memory floor. The boot
ledger prints the arithmetic behind each of these decisions, and while
serving, an idle-tick reconciliation line checks the box's raw memory
drop since boot against what the engine's own ledger explains, logging
the signed residual (also on `/v1/stats` and `/metrics`) — an
unexplained phantom or leak surfaces as a named number, not a field
report.

The proving runs for v0.6.1: a zero-config boot at `-c 786432` admitted
three ingestions of about 755 thousand tokens each, back to back, and
held **2.26 million tokens of context resident and warm at once** on
one 128 GB Spark, with zero refusals and the 4 GiB floor intact. A
fourth deep ingestion was funded by reclaiming an idle bank (the
cheapest to restore), not refused. Decode measured at parity with an empty box at
450k-token depth and within 15 percent at 755k. The observed all-in
cost was about 4.3 KiB per token of resident context at the deep shape
(4.8 at 450k banks; deeper banks amortize the page floors).

The measured ceiling sits higher. With the admission floor lowered to
1 GiB on a dedicated box, the same governor held **3,019,176 tokens of
active context** — four ingestions of about 755 thousand tokens each,
a needle retrieved exactly from every one, and honest, instant
refusals for every further ask with the floor intact. The disclosed
cost: at that full squeeze decode runs 2.6x slower than on an empty
box (the OS starts reclaiming file-backed weight pages), where the
2.26M shipped-floor stamp is 1.14x. The step-by-step recipe is in the
[ds4-on-spark README](https://github.com/Entrpi/ds4-on-spark#reaching-3m-tokens-of-active-context).

The window itself is now qualified to its edge: at `-c 1048576` (the
checkpoint's exact YaRN window, 65536 x 16) a single prompt of
**1,029,340 tokens** was ingested with a needle at 99.9% depth and
retrieved exactly — on the default decode path and again with the
accelerated attention path disabled, the fallback that previously
truncated the deepest rows silently past 1,015,936 resident tokens.
A 975,246-token conversation was separately admitted and continued
warm in place, with a 2.0 s time to first token and 453 tokens
decoded at 88 ms per token at that depth, speculative decoding still
accepting 63 percent of its drafts. A bank persisted at exactly its
token bound now restores cleanly across a restart.

Two decisions cover most operator needs: the context limit `-c` (the
per-request ceiling; prompt plus decode budget must fit under it) and
the bank count (how many requests hold warm context at once; an
admission beyond it evicts the least-recently-used idle bank rather
than refusing). The knobs:

| Knob | Default | What it does and when to change it |
|---|---|---|
| `-c` / `--ctx` | `262144` on CUDA, `32768` on Metal/CPU (ds4-on-spark's `ds4-serve` passes `524288`) | The per-request context ceiling. Each request must fit its prompt plus its decode budget under this, or it gets a typed 400. Raise it for deeper documents (a 1,029,340-token prompt at `-c 1048576`, the model's full window, is the deepest proven); unused context is demand-mapped, so a deep ceiling costs almost nothing until a request fills it. |
| `DS4_SERVER_COALESCE_MAX` | unset: sized from the live memory budget at boot — 32 through 16k context (the measured regime); above that, as many full-depth-fundable banks as the budget covers (floor 4, cap 32), priced at the same per-token rate admissions are charged. The boot ledger prints the arithmetic (`kv plan` line). Where no memory answer exists (Metal/CPU), a static halving ladder rules instead. | How many requests can hold warm context at once (the bank count). Set it (1..64) to override — an explicit value also disarms the budget sizing, so your number rules (e.g. more, shallower banks for high-concurrency batch work). Boot may still reduce the count to fit memory, never raise it, and it is fixed until restart. A request beyond the bank count evicts the least-recently-used idle bank. |
| `--mem-floor-gb` (env `DS4_MEM_FLOOR_GB`) | `4` | The engine never admits work that would leave the system under this many GiB of free memory: it reclaims idle cache first and refuses with a typed error if that is not enough. Lower it (down to 1) on a dedicated serving box to fund more context; raise it on a machine you also work on. The boot planning headroom now derives from this floor plus a boot-burst margin (`DS4_BATCH_FIT_BURST_MB`, default 2048; boot line `batch fit headroom:`), so lowering the floor also returns planning margin to the fundable pool — `DS4_BATCH_FIT_HEADROOM_MB` pins the headroom outright, `DS4_BATCH_FIT_HEADROOM_DERIVED=0` restores the old static 6144. |
| `max_tokens` (request field, not a flag) | `32768` assumed when the client omits it | The decode budget a request is charged at admission, on top of its prompt. Agents that omit it are charged the full 32768, so set it explicitly when a deep prompt must fit: prompt + `max_tokens` must stay under `-c`. An oversized value is clamped and reported as `length`, never an error. |
| `DS4_CONT_ADMIT_BAND_X1024` | `1045` | Admission charges each request its measured memory need times a small safety margin, expressed in 1024ths: 1045/1024 means about 2% above the measurement, absorbing allocation transients. Set `1024` to charge exactly the measured need; raise it if admitted work ever brushes the floor. |
| `DS4_MEMGOV` | unset (the governor's verdicts are binding) | Set to `observe` to fall back to the pre-v0.6 memory formulas: the governor keeps evaluating and reporting on `/metrics`, but stops deciding. The one-word escape hatch if a memory decision ever looks wrong. |
| `DS4_MEM_RECONCILE_TOL_MB` | `256` | When idle, the server reconciles the box's available-memory drop since boot against what its own allocation ledger explains and logs the residual (`mem reconcile:` line, also on `/v1/stats` and `/metrics`); a residual beyond this many MiB is marked `FLAGGED`. `DS4_MEM_RECONCILE_STRICT=1` adds a distinct `mem reconcile STRICT` line for gate scripts to assert on; `DS4_MEM_RECONCILE_WARMUP_MB` pins the named one-time warmup charge instead of letting the first idle pass self-calibrate it. Pure reporting — no admission decision reads it. |
| `DS4_CONT_PREFILL_CHUNK` / `DS4_CONT_PREFILL_CHUNK_LIVE` | `4096` / `512` | Long prompts are ingested this many tokens at a time so a big admission never blocks the server. The `_LIVE` value applies while other requests are actively decoding: smaller keeps live decode smoother, larger ingests faster. |
| `DS4_SERVER_CONTINUOUS` | `1` (continuous batching on) | Set to `0` to serve one request at a time on the old serial path. Only worth considering for single-user, latency-critical setups. |
| `DS4_BATCH_VMM_BUDGET_MB` | unset: sized automatically (the bank plan's allowance, capped to measured capacity at boot, floored at two full-depth **packed** working sets — what two full banks actually commit at the admission-charged rate, not their virtual extents; `DS4_BATCH_VMM_FLOOR_PACKED=0` restores the old virtual-extent floor) | Hard cap on the KV pool, in MiB. Set it to pin the pool to a known size; either way the boot ledger prints `budget=[chosen] [plan X, capacity Y]` plus the work floor that applied, and a separate line whenever the floor is what ruled. |
| `DS4_BATCH_VMM_TRIM` | `1` (reclaim allowed) | When an admission does not fit, the engine may release idle banks' memory to fund it; the reclaimed conversation then needs a disk restore or re-prefill when it returns. Set to `0` to forbid that: resident context is never sacrificed, and the admission is refused instead. Victims are chosen like warm-record eviction — invalid content first, then the longest-idle bank (shortest history breaks ties) — and the log names each victim with its bytes, history length, and recency; `DS4_BATCH_TRIM_VICTIM=hist` restores the old shortest-history-only order. When one victim's release would cover the whole remaining deficit, the engine now picks the smallest such victim in the same validity class instead of the first in recency order — so a deep trunk no longer dies for a deficit a small idle bank could fund (the `best-fit victim` log line discloses the substitution, and the trim summary reports released vs wanted); `DS4_BATCH_TRIM_BESTFIT=0` restores the pure recency order. |
| `DS4_SERIAL_RESERVE_CTX` | unset (no reserve) | Set to a token count to reserve memory at boot for the single-request serial lane, for deployments where that lane matters more than batch depth. The boot line reports the carve-out. |
| `DS4_WEIGHT_FP_CHECK` | `1` (verify) | Weight-server imports verify the manifest's content fingerprint against the local model file and refuse a mismatch (a stale weight server serving different bytes). Set to `0` to skip the check. |

Observability: `/metrics` carries the memory families, an allocation
census by class and domain, the availability observation with both raw
estimates behind it (`ds4_memory_observation_bytes{kind=...}`),
governor decisions per consumer, reclaim outcomes, and typed request
rejections labelled by lane and reason. One reading note for long-lived
servers: the kernel's available-memory estimate drifts downward as the
mapped model's page-cache residency grows, even though allocations
still succeed and nothing is leaking. A restart resets the reading, and
the two raw estimates on `/metrics` make the artifact identifiable.

If you are coming from vLLM or SGLang, the assumption contrast and a
flag-by-flag comparison table are in the
[ds4-on-spark README](https://github.com/Entrpi/ds4-on-spark#if-you-come-from-vllm-or-sglang),
along with the max-capacity recipe for
[reaching 3M+ tokens of active context](https://github.com/Entrpi/ds4-on-spark#reaching-3m-tokens-of-active-context)
on a stripped headless Spark.

## Thinking Modes

DeepSeek V4 Flash has distinct non-thinking, thinking, and Think Max modes.
The server defaults to thinking mode. The checkpoint's high and max tiers
are not just decode budgets: they inject DeepSeek's effort preamble
(verbatim reference-encoder text) at position 0, ahead of the system
prompt.

Since v0.6.3.1, a client-sent `reasoning_effort` field cannot reach those
prefixed tiers by default: client `high`/`xhigh`/`max` compat-map to the
prefix-free level. Agent frameworks send the field meaning the OpenAI
"think more" knob, and a controlled needle matrix showed the injected
preamble measurably degrades deep-context tool calling (6/50
completion-protocol failures at 96K+ tokens with the prefix vs 0/100
without — the field issue #18 regression). llama.cpp and pre-v0.5.3
engines silently ignore the field for this GGUF, which is the behavior the
default restores. Operators opt back into the native tiers with
`--reasoning-effort-native` (env `DS4_REASONING_EFFORT_NATIVE=1`), and the
operator's own `--reasoning-effort low|high|max|off` server default is
always honored as written: setting it is the opt-in for that level.
Disabling thinking stays client-reachable either way.

For direct replies, use `thinking: {"type":"disabled"}`, `think:false`,
`reasoning_effort:"off"`, or a non-thinking model alias such as
`deepseek-chat`.

## Disk KV Cache

Chat/completion APIs are stateless: agent clients usually resend the whole
conversation every request. `ds4-server` first tries the cheap exact token-prefix
check, then falls back to comparing rendered prompt bytes with decoded
checkpoint bytes. The live in-memory checkpoint covers the current session; the
disk KV cache makes useful prefixes survive session switches and server
restarts.

For RAM reasons there is currently only one live KV cache in memory. When a new
unrelated session replaces it, the old checkpoint can only be resumed without
re-processing if it was written to the disk KV cache. In other words, memory
cache handles the active session; disk cache is the resume mechanism for
different sessions.

Enable it with:

```sh
./ds4-server --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

> **Long-running serving note.** How the per-bank KV pool grows, what
> funds each admission, the memory floor, and every related knob are in
> [Memory and capacity](#memory-and-capacity). The short version: pages
> map on demand, eviction releases them back, each admission is charged
> its measured need against live free memory, and a typed refusal
> arrives before the box is ever squeezed.

The cache key is the SHA1 of the rendered byte prefix, and files are named
`<sha1>.kv`. The DS4 payload still stores the exact token IDs and graph state
for that prefix. This matters for continued chats: the model may have generated
one token whose decoded text is later sent back by a client as two canonical
prompt tokens. A rendered byte-prefix hit can still reuse the checkpoint and
tokenize only the new suffix.
The file is intentionally written with ordinary `read`/`write` I/O, not
`mmap`, so restoring cache entries does not add more VM mappings to a process
that already maps the model.

Tool calls also keep a bounded exact-DSML replay map keyed by unguessable tool
IDs, so client JSON history can be rendered back to the exact sampled text. The
RAM map keeps up to 100000 IDs by default; tune it with `--tool-memory-max-ids`.
Use `--disable-exact-dsml-tool-replay` to disable this and fall back to
canonical JSON-to-DSML rendering.

On disk, a cache file is:

```text
KVC fixed header, 48 bytes
u32 rendered_text_bytes
rendered_text_bytes of UTF-8-ish token text
DS4 session payload, payload_bytes from the KVC header
optional tool-id map section
```

The fixed header is little-endian:

```text
0   u8[3]  magic = "KVC"
3   u8     version = 1
4   u8     routed expert quant bits, currently 2 or 4
5   u8     save reason: 0 unknown, 1 cold, 2 continued, 3 evict, 4 shutdown
6   u8     extension flags, bit 0 = appended tool-id map
7   u8     reserved
8   u32    cached token count
12  u32    hit count
16  u32    context size the snapshot was written for
20  u8[4]  reserved
24  u64    creation Unix time
32  u64    last-used Unix time
40  u64    DS4 session payload byte count
```

The rendered text is the tokenizer-decoded text for the cached token prefix.
It is both the human-inspectable prefix and the lookup identity: its SHA1 is
the filename, and a file is reusable only when those bytes are a prefix of the
incoming rendered prompt. After load, the exact checkpoint tokens from the DS4
payload remain authoritative, and only the incoming text suffix after the cached
bytes is tokenized.

The optional tool-id map is present only when header extension bit 0 is set.
Appended sections use fixed bit order, so future extension bits can add fields
without ambiguity. The map stores unguessable API tool call IDs back to the
exact DSML block the model sampled. Only mappings whose DSML block is present
in the rendered cached text are stored. This lets restarted servers render
later client history byte-for-byte like the original model output, even if the
client reorders JSON arguments.

The current tool-id map section is:

```text
0   u8[3]  magic = "KTM"
3   u8     version = 1
4   u32    entry count

For each entry:
0   u32    tool id byte length
4   u32    sampled DSML byte length
8   bytes  tool id
... bytes  exact sampled DSML block
```

The section is auxiliary replay memory, not model state. A cache hit restores
the session payload first, then loads the map if present. Before rendering a
request, the server can also scan cache files for the tool IDs present in the
client history and load just those mappings, so an exact DSML replay can survive
server restarts even when the matching KV snapshot is not the one ultimately
used for the rendered-prefix hit.

The DS4 session payload starts with thirteen little-endian `u32` fields:

```text
0   magic = "DSV4"
1   payload version = 2
2   saved context size
3   prefill chunk size
4   raw KV ring capacity
5   raw sliding-window length
6   compressed KV capacity
7   checkpoint token count
8   layer count
9   raw/head KV dimension
10  indexer head dimension
11  vocabulary size
12  live raw rows serialized below
```

Then it stores:

- `u32[token_count]` checkpoint token IDs.
- `float32[vocab_size]` logits for the next token after that checkpoint.
- `u32[layer_count]` compressed attention row counts.
- `u32[layer_count]` ratio-4 indexer row counts.
- For every layer: the live raw sliding-window KV rows, written in logical
  position order rather than physical ring order.
- For compressed layers: live compressed KV rows and compressor frontier
  tensors.
- For ratio-4 compressed layers: live indexer compressed rows and indexer
  frontier tensors.

The logits are raw IEEE-754 `float32` values from the host `ds4_session`
buffer. They are saved immediately after the checkpoint tokens so a loaded
snapshot can sample or continue from the exact next-token distribution without
running one extra decode step. MTP draft logits/state are not persisted; after
loading a disk checkpoint the draft state is invalidated and rebuilt by normal
generation.

Distributed coordinator sessions use the same `DSV4` payload. Worker-owned
layer tensors are pulled during save and merged into the normal layer-ordered
tensor stream; during load the coordinator splits that stream into the current
route and pushes the relevant layer tensors back to the workers. The saved file
does not retain the distributed topology.

The tensor payload is DS4-specific KV/session state, not a generic inference
graph dump. It is expected to be portable only across compatible `ds4.c`
builds for this model layout.

The cache stores checkpoints at four moments:

- `cold`: after a long first prompt reaches a stable prefix, before generation.
- `continued`: when prefill or generation reaches the next absolute aligned frontier.
- `evict`: before an unrelated request replaces the live in-memory session.
- `shutdown`: when the server exits cleanly.

Cold saves intentionally trim a small token suffix and align down to a prefill
chunk boundary. This avoids common BPE boundary retokenization misses when a
future request appends text to the same prompt. The defaults are conservative:
store prefixes of at least 512 tokens, cold-save prompts up to 30000 tokens,
trim 32 tail tokens, and align to 2048-token chunks. The important knobs are:

Continued saves use the same alignment and are written only when the live graph
naturally reaches an absolute frontier. With the defaults this means roughly
every 10k tokens, independent of where the first cold checkpoint landed, so long
generations leave restart points behind without persisting the fragile final few
tokens.

- `--kv-cache-min-tokens`
- `--kv-cache-cold-max-tokens`
- `--kv-cache-continued-interval-tokens`
- `--kv-cache-boundary-trim-tokens`
- `--kv-cache-boundary-align-tokens`
- `--tool-memory-max-ids`
- `--disable-exact-dsml-tool-replay`

By default, checkpoints may be reused across the 2-bit and 4-bit routed-expert
variants if the rendered prefix matches. Use `--kv-cache-reject-different-quant`
when you want strict same-quant reuse only.

The cache directory is disposable. If behavior looks suspicious, stop the
server and remove it. You can investigate what is cached with hexdump as
the kv cache files include the verbatim prompt cached.

## CLI

One-shot prompt:

```sh
./ds4 -p "Explain Redis streams in one paragraph."
```

No `-p` starts the interactive prompt:

```sh
./ds4
ds4>
```

The interactive CLI is a real multi-turn DS4 chat. It keeps the rendered chat
transcript and the live graph KV checkpoint, so each turn extends the previous
conversation. Useful commands are `/help`, `/think`, `/think-max`, `/nothink`,
`/ctx N`, `/read FILE`, and `/quit`. Ctrl+C interrupts the current generation
and returns to `ds4>`.

The CLI defaults to thinking mode. Use `/nothink` or `--nothink` for direct
answers. `--mtp MTP.gguf --mtp-draft 2` enables the optional MTP speculative
path; it is useful only for greedy decoding, currently uses a confidence gate
(`--mtp-margin`) to avoid slow partial accepts, and should be treated as an
experimental slight-speedup path.

## Native agent

DwarfStar features a native coding agent that works in a different way
than most other systems: the inference is controlled from within the agent
itself, without socket/API boundaries, so the session is represented
by the on-disk KV cache itself. Moreover the tools and the system prompt
are all designed vertically for DeepSeek v4 Flash and PRO. This provides a
few advantages:

* Low latency experience, bounded mainly by the prefill speed limits. Displaying of generated text, tool calling, start of a new session are always instantaneous.
* Live progress bar during prefill time.
* No DSML tool calling conversion, the tools are handled natively in the LLM format.
* KV cache mismatch are impossible by construction, the current state is always the truth.
* Everything is tuned for this model.
* Ability to switch saved sessions with `/list` and `/switch`; full KV sessions resume without a prefill stage.

Agent sessions are stored in `~/.ds4/kvcache`. Use `/save` to persist the
current session, `/list` to show saved sessions sorted by recent update time,
and `/switch <sha>` to resume one of them. The session ID is stable across
future saves and is derived from the first user prompt and creation time.
`/del <sha>` removes a saved session. `/strip <sha>` keeps the rendered
conversation text and title but removes the heavy KV payload; switching to a
stripped session rebuilds the KV cache by prefilling the saved text.

Use `--chdir /path/to/ds4` when launching `ds4-agent` from another directory,
so relative runtime files such as `metal/*.metal` resolve from the project tree.

However while the system already works, there is a lot of work to do
in order to make it ready for prime time. When finally the agent will reach
the wanted shape, we will *likely* split the server and the client creating a stateful
session-based protocol that can recreate all that in a client-server way.

## Benchmarking

`ds4-bench` measures instantaneous prefill and generation throughput at context
frontiers instead of reporting one whole-run average. It loads the model once,
walks a fixed token sequence to frontiers such as 2048, 4096, 6144, and uses
incremental prefill so each row measures only the newly-added token interval.
After each frontier it saves the live KV state to memory, generates a fixed
greedy non-EOS probe, restores the memory snapshot, and continues prefill.

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

The example file is a cleaned public-domain Project Gutenberg text of
Alessandro Manzoni's *I Promessi Sposi* (ebook #45334), with the Gutenberg
header and footer removed: <https://www.gutenberg.org/ebooks/45334>.

Use `--step-incr N` for different linear spacing, or `--step-mul F` for
exponential sweeps. Output is CSV with one row per frontier: latest prefill
interval tokens/sec, generation tokens/sec at that frontier, the
steady-state generation throughput (`gen_tps_ss`, which excludes the
first-token amortisation cost so the column compares apples-to-apples
across short and long generations), the first-token latency, and
`kvcache_bytes`. Committed sweeps live under `speed-bench/`; the included
[pro6000_blackwell_ts.svg](speed-bench/pro6000_blackwell_ts.svg) and
[gb10_spark_ts.svg](speed-bench/gb10_spark_ts.svg) are generated from
those CSVs via `python3 speed-bench/plot_speed.py`.

Sessions prefill long prompts in 4096-token chunks by default. Set
`DS4_METAL_PREFILL_CHUNK=N` to compare another chunk size, for example `2048`
to match the strict official-vector checkpoint path, or
`DS4_METAL_PREFILL_CHUNK=0` to prefill a prompt as one whole batch when memory
allows. Changing the chunk changes the KV checkpoint/logit path, so compare it
as an explicit run configuration. Single forwards wider than 8,192 rows are
fenced: that territory is unqualified (fixed grid and integer ceilings in the
one-shot kernels, with crash-or-silently-wrong failure modes), so the server
refuses such a request with a typed error naming the lever, and the boot log
discloses the mode whenever it is active. Set `DS4_PREFILL_NOFENCE=1` to lift
the fence for a deliberate probe run.
Chunked Metal prefill reuses the same range-capable layer-major graph for each
chunk, preserving absolute compressor/indexer boundaries while avoiding the old
per-layer chunk dispatch path.

## Capability Evaluation

`ds4-eval` is a small real-model integration benchmark. It is not a leaderboard
runner and should not be reported as an official GPQA, SuperGPQA, AIME, or
security benchmark score: the questions are an embedded 92-item subset chosen
to make local regression testing useful and visually inspectable. The program
loads the real GGUF,
renders DS4 chat prompts, streams sampled tokens in a split-screen TUI, grades
the final answer, and prints a per-question report with prompt tokens,
generated tokens, pass/fail state, the model answer, and the correct answer.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
```

The default run uses `--tokens 16000`, thinking mode enabled, and a soft/hard
`</think>` budget cutoff so the model has room to produce a visible answer.
`ds4-eval` sizes the context internally from the largest selected prompt plus
the generation budget, and refuses runs that would need more than 1M context
tokens. Press `p` to pause, `q` to exit and print the report, Up/Down to
inspect or select another question, and Enter to run the selected question next.
`--plain` disables the TUI.

Use `--regrade-trace /path/to/trace.txt` to replay the current answer
extractor and scorer against a prior `--trace` file without loading the model
or regenerating tokens. This is useful when auditing evaluator changes: it
shows which cases changed, the old picked answer, the new picked answer, and a
pass/fail summary.

For inference changes that can affect generation drift, keep this deterministic
q1..q4 token-count gate in the test plan:

```sh
./ds4-eval \
  -m ds4flash.gguf \
  --plain \
  --questions 4 \
  --tokens 2048 \
  --temp 0 \
  --seed 1
```

The generated-token counts must stay aligned with the baseline:

| Question | Expected state | Expected generated tokens | Expected given/correct |
|---:|---|---:|---|
| 1 | `PASSED` | 2048 | `B` / `B` |
| 2 | `PASSED` | 438 | `C` / `C` |
| 3 | `PASSED` | 666 | `70` / `70` |
| 4 | `FAILED` | 2048 | `A` / `C` |

The first 75 embedded questions are interleaved as 25 GPQA Diamond, 25 audited
SuperGPQA, and 25 AIME 2025 problems. The final 17 are an audited COMPSEC
subset of reduced single-function C/C++ vulnerability-localization questions.
The model is asked for the single best source line, or the smallest exact line
set only when the bug cannot be localized to one line; the scorer accepts small
audited ranges only when adjacent lines are equivalent locations for the same
bug. The order is
intentionally progressive: early questions are useful smoke tests, while later
questions are hard enough that a strong reasoning model should still miss some
of them. The SuperGPQA slice is curated rather than blind: upstream rows with
wrong keys, missing figures, or underspecified prompts are replaced with cleaner
rows.

The set should be treated as a hard capability regression suite rather than
a pass/fail unit test.

- **GPQA Diamond** contributes graduate-level science questions with
  multiple-choice answers. DeepSeek's model card reports strong results
  on full GPQA Diamond in thinking mode, but individual items still require
  careful physics, chemistry, or biology reasoning and are easy to lose with a
  small prompt/rendering or sampling regression.
- **SuperGPQA** contributes broad specialist knowledge and domain-transfer
  questions. The model-card SuperGPQA number is much lower than GPQA Diamond,
  so these items are expected to be uneven: some look mundane, others require
  niche professional knowledge or exact interpretation of a translated-style
  exam question.
- **AIME 2025** contributes exact-answer contest math. These are often the most
  unforgiving items in the set: no multiple-choice prior, no partial credit, and
  a single arithmetic or algebraic slip changes the grade.
- **COMPSEC** contributes single-function C/C++ security reasoning items
  reduced from public CVE writeups. These are not exploit prompts: the task is
  to identify the best source line where the defensive code flaw is introduced,
  or return `0` for a safe function.

In practice this means `ds4-eval` should not be expected to produce a perfect
92/92 run. It is meant to answer a more useful engineering question: after a
kernel, quantization, prompt-rendering, KV-cache, or tool-streaming change, does
DeepSeek V4 Flash still solve a representative mix of hard science, broad
knowledge, exact math, and security-code problems while using the same inference
path users run?

## Distributed Inference

Distributed inference lets DS4 **run a model that is too large for one machine** by
splitting transformer layers across multiple machines. The main example is the
full 4-bit Flash quant across two 128 GB MacBooks: each process maps only its
own layer slice, activations are sent over TCP, and the coordinator keeps normal
CLI/API behavior.

Distributed inference also allows to **speed up prefill** by
using multiple GPUs at the same time to process different micro-batches at
different layers, like in an assembly line. Only prefill can be accelerated this
way. Generation is purely autoregressive: each token must finish across the
route before the next token can start. The model work is the same as a single
process, plus coordination latency, so distributed generation is slower.

To build an initial mental model, here are the high level concepts:

1. You put the GGUF on every machine, but each one loads just a subset. `--layers` controls which tensors are mapped, so a worker with `--layers 20:output` does not load the earlier layers.
2. Layer ranges are inclusive: `10:20` means layers 10, 11, ..., 20. `N:output` means layer `N` through the final layer plus the output head.
3. You assign one of the machines the role of `coordinator`, the others the roles of `workers`. Workers will connect to the coordinator and will tell they are there and which layers they are able to process.
4. Each worker keeps its slice of the KV cache.
5. Communication is worker-to-worker, there is no need to use the coordinator as relay, so if your coordinator is `A`, and you make a request, activations will flow in `A -> B -> C -> back to A`.

### How it works and how to configure it

The prefill path is pipelined (this is why it can go faster than in a single machine).
For large prompts the coordinator can run its
slice on chunk N+1 while the worker is running its slice on chunk N. The
distributed rows below were measured with two M5 Max 128 GB MacBooks connected
by Thunderbolt 5, using the Q4 Flash GGUF and the default 4096-token
distributed prefill chunk. The single-process column is a reference run with
the Q2 GGUF on a single machine, so it actually is a bit faster since
the routed MoEs are smaller.

| Prompt | Single-process reference | Two MacBooks | Speedup |
| ---: | ---: | ---: | ---: |
| 9421 tokens | 421.70 t/s | 582.22 t/s | 1.38x |
| 28684 tokens | 405.30 t/s | 674.16 t/s | 1.66x |
| 63819 tokens | 353.62 t/s | 654.79 t/s | 1.85x |

Generation is different. **It is strictly autoregressive**: token N+1 cannot start
until token N has produced logits and sampling has selected the next token. That
means distributed generation cannot use the long prefill pipeline. It pays at
least one cross-machine activation hop per generated token, so generation is
slower than a single local process. On the same two-Mac Thunderbolt setup, a
12k-context control run with the 91 GB Flash quant went from 30.59 t/s
single-process to 24.67 t/s distributed, a 19.4% loss. Distributed inference is
therefore mainly for fitting larger models and speeding up long prefills, not
for making decode faster.

The measurements above use a Thunderbolt 5 cable. The implementation is plain
TCP and also works over slower links, including WiFi, but fast Ethernet or
Thunderbolt networking is strongly recommended. Slow links mostly hurt
generation latency and short prefills; large prefills can still benefit when
the layer split is balanced. In the normal performance path, the last worker
owns the output head and returns logits directly.

Minimal two-host configuration:

```sh
# Machine A: coordinator, owns tokenization, sampling, the prompt, and layers 0..19.
./ds4 \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --role coordinator \
  --layers 0:19 \
  --listen 169.254.43.68 1234

# Machine B: worker, connects to A and owns layers 20..output.
./ds4 \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --role worker \
  --layers 20:output \
  --coordinator 169.254.43.68 1234
```

Normally the final worker should own the output head too, for example
`--layers 20:output`. This avoids returning a full final hidden-state batch
after prefill and lets the final worker produce the logits directly. On very
slow or metered links, `--layers 20:42` is also supported: the coordinator will
load the output head and compute logits locally, trading extra coordinator work
for smaller per-token replies.

### Network Link Comparison

The table below shows the same two M5 Max hosts, the same 91 GB Flash quant,
coordinator `--layers 0:19`, worker `--layers 20:output`, an 8192-token prompt
from `speed-bench/promessi_sposi.txt`, and 128 generated tokens. WiFi and
Internet numbers vary with local conditions, but the shape is the important
part: high latency hurts generation directly, while lower bandwidth also pulls
down long-prefill speed.

| Link | Addresses | Ping avg | Prefill | Generation |
| --- | --- | ---: | ---: | ---: |
| Thunderbolt 5 | `169.254.43.68` -> `169.254.12.245` | 0.45 ms | 582.99 t/s | 25.09 t/s |
| WiFi | `192.168.1.57` -> `192.168.1.95` | 77.20 ms | 250.70 t/s | 10.70 t/s |
| Internet / VPN | `10.77.0.4` -> `10.77.0.3` | 152.10 ms | 114.88 t/s | 3.63 t/s |

The Internet/VPN case is not meant to be a good interactive experience. It is
still useful for collective testing: multiple people can temporarily combine
machines to run a larger model that would not fit on any single host, accepting
slow decode in exchange for being able to inspect the model at all.

Use the coordinator exactly like normal `./ds4`: interactive chat, `/read`,
and ordinary generation go through the same high-level session API. The same
distributed options are also wired into `ds4-agent`, `ds4-eval`, and
`ds4-bench`. For benchmarks, workers should already be running; `ds4-bench`
waits until a complete route is available.

Useful tuning and diagnostics:

```sh
./ds4-bench \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 32768 \
  --ctx-max 65536 \
  --step-incr 32768 \
  --gen-tokens 0 \
  --role coordinator \
  --layers 0:19 \
  --listen 169.254.43.68 1234 \
  --debug
```

`--debug` on the coordinator prints route formation and per-hop telemetry:
layer range, token span, local evaluation time, downstream wait time, socket
send time, and input/output byte counts. This is the current profiling tool for
deciding whether a split is balanced. `--dist-prefill-window N` controls how
many prefill chunks may be in flight end-to-end; the default is conservative
and bounded. `--dist-prefill-chunk N` exists for experiments, but the default
4096-token chunk is the canonical setting and should be used unless you are
explicitly validating a different chunk size.

By default DS4 sends hidden-state activations as 32-bit floats. To reduce
traffic, pass `--dist-activation-bits 16` or `--dist-activation-bits 8` on the
coordinator. This changes only the transport format between machines, not the
model weights or KV cache. 16-bit transport halves activation traffic and is the
first option to try on Ethernet or WiFi. 8-bit transport is more aggressive and
should be treated as an approximate/experimental mode unless you have validated
the output for your use case. However experimentally reduction activation
size didn't provide a significant improvement, so this option may be removed
in the future.

**If a worker disconnects, the coordinator removes that worker from the active
route**. The request already in flight can fail, and later calls report an
incomplete route until a compatible worker reconnects and sends a new
registration. For live sessions, the coordinator keeps the token history and can
rebuild worker KV state by replaying the prefix when the route is available
again. Workers also validate a rolling 64-bit token-prefix hash on every work
item, so a restarted worker at position 0 cannot silently accept work for
position N; it reports the mismatch and the coordinator replays the current
transcript. Ctrl+C in the CLI and agent is cooperative: DS4 waits for the
current distributed token or prefill chunk to drain before returning control,
which avoids coordinator-caused KV splits. Saved agent/server sessions use the
same KV file format as single-machine sessions: during save the coordinator
fetches worker-owned layer tensors and serializes one normal payload; during
load it splits that payload over the currently registered route.

### Distributed protocol overview

At the protocol level there are two kinds of connections. Workers keep a
control TCP connection open to the coordinator and send a `HELLO` with their
model ID, model family, quant profile, layer slice, context capacity, and data
port. The coordinator uses these registrations to build a route that covers all
layers. Work then moves over low-latency TCP data connections: the coordinator
computes the first slice, sends a `WORK` frame with session ID, token positions,
rolling token-prefix hashes before and after the span, route information, and
hidden-state payload, and each worker computes its slice. Middle workers can
forward directly to the next worker. The final worker returns logits to the
coordinator, or ACKs for non-final prefill chunks so the prefill pipeline can
stay full. `RESULT` frames echo the request ID and the post-span hash. A worker
status error is handled differently from a socket failure: KV/hash mismatch can
be recovered by replaying the token history on the same route, while transport
failure drops the route and waits for a replacement worker. For persistent KV,
the coordinator opens worker data connections and sends snapshot save/load
messages for each worker-owned layer range; the disk payload remains a single
agent/server cache file. The protocol has no
encryption or authentication, and is not release-stable yet; coordinator and
workers should be built from the same commit and used on trusted machines and
trusted networks.

## Reducing heat, power usage and fan noise

Long local inference runs can keep the GPU busy for extended periods. If you
care more about heat, fan noise, battery life on MacBooks, or reducing thermal
stress on the hardware than about maximum throughput, use `--power N`.

`--power 100` is the default and means full speed. Lower values ask DS4 to target
that percentage of GPU usage: `--power 70` targets about 70%, `--power 50`
targets about half usage, and so forth. DS4 does this by measuring GPU work time
and inserting small sleeps between work units: during prefill it sleeps between
layers, and during generation it sleeps between decoded tokens. This reduces
sustained load without changing model output.

The option is available on the CLI, server, agent, eval, and benchmark tools,
for example:

```sh
./ds4 --power 50
./ds4-agent --power 70
./ds4-server --power 40 --ctx 100000
```

## Backends

The default graph backend is Metal on macOS and CUDA in CUDA builds:

```sh
./ds4 -p "Hello" --metal
./ds4 -p "Hello" --cuda
```

On Linux, plain `make` prints the available build targets instead of selecting a
CUDA target implicitly. Use `make cuda-spark` for DGX Spark / GB10; it builds
with `nvcc -arch=sm_121`, the GB10's native architecture — an empty
`-arch` measured ~25% slower prefill on GB10. Use `make cuda-generic` for a
normal local CUDA build, or set `CUDA_ARCH` explicitly when cross-building or
when you need a known target:

```sh
make cuda CUDA_ARCH=sm_120
make cuda CUDA_ARCH=native
```

There is also a CPU reference/debug path:

```sh
./ds4 -p "Hello" --cpu
make cpu
./ds4
./ds4 -p "Hello"
```

Do not treat the CPU path as the production target. The CLI and `ds4-server`
support the CPU backend for reference/debug use and share the same KV session
and snapshot format as Metal and CUDA, but normal inference should use Metal or
CUDA.

## Steering

This project supports steering with single-vector activation directions; see the
`dir-steering` directory for more information. This follows the core idea of the
[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
paper. You can use it to make the model more or less verbose, less likely to
answer programming questions if it is a chatbot for your car rental web site,
and so forth, much faster than fine-tuning.
This is also useful for cybersecurity researchers who want to reduce a model's
willingness to provide dual-use or offensive security guidance.

## Test Vectors

`tests/test-vectors` contains short and long-context continuation vectors
captured from the official DeepSeek V4 Flash API. The requests use
`deepseek-v4-flash`, greedy decoding, thinking disabled, and the maximum
`top_logprobs` slice exposed by the API. Local vectors are generated with
`./ds4 --dump-logprobs` and compared by token bytes, so tokenizer/template or
attention regressions show up before they become long generation failures. The
C runner pins `DS4_METAL_PREFILL_CHUNK=2048` for this strict API-vector
comparison.

All project tests are driven by the C runner, with a small `ds4-eval`
extractor self-test run first:

```sh
make test                  # ./ds4-eval --self-test-extractors && ./ds4_test --all
./ds4_test --logprob-vectors
./ds4_test --server
```

## Debugging Notes

When a generation looks wrong, three small tools are usually enough to get a
first answer:

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 --dump-logits /tmp/logits.json --metal --nothink --prompt-file prompt.txt
./ds4-server --trace /tmp/ds4-trace.txt ...
./ds4-server --tool-slip-dump /tmp/ds4-slips ...
```

- `--dump-tokens` tokenizes the `-p` or `--prompt-file` string exactly as
  written, recognizes DS4 protocol specials, and then exits before inference
  starts. For example, the DSML tool close marker starts as two tokens: `</`
  and `｜DSML｜`.
- `--dump-logprobs` stores a greedy continuation with the top local
  alternatives at each step, which helps separate sampling choices from
  logit/model issues.
- `ds4-server --trace` writes the rendered prompts, cache decisions, generated
  text, and tool-parser events for a whole agent session. Serial lane only:
  continuously-batched rows (the default serving lane) do not trace yet.
- `ds4-server --tool-slip-dump DIR` covers the batched lane's blind spot for
  tool-protocol failures: every tools-armed chat completion that settles
  without tool calls dumps one JSON file with the raw request body, the full
  generated text, and the parse verdict. Agent harnesses discard rejected
  responses, so this is usually the only byte-exact record of what the model
  actually said; each dump replays directly against a server as a regression
  fixture.

## About this fork

This fork asks the question upstream deliberately leaves open: what
does DwarfStar look like as a multi-request serving engine on NVIDIA
hardware? Everything fork-side is default-on, each landing gated by
value-parity checks, same-boot A/B timing, and the full eval suite, and
each reversible with an env kill switch. The per-landing numbers and
the full story are in [CHANGELOG.md](CHANGELOG.md); the headline
results, each receipted there:

- **Continuous batched serving.** Mid-flight admission and eviction
  over per-request KV banks, chunked cold admission, pending prefills
  interleaved with live decode, warm start from cached prefixes (~7x
  TTFT), fork-by-copy fan-out (~49x TTFT on a 4-way branch), and
  durable banks that survive eviction and restarts. Stops, tool calls,
  thinking, and speculation all ride the batched path, on the OpenAI
  and Anthropic API surfaces alike. Aggregate decode 59 tok/s at 12
  concurrent requests (v0.5.0 stamp).
- **A rebuilt CUDA prefill engine.** D2R (dequant-to-registers)
  tensor-core MoE GEMMs reading aligned SoA artifacts in place,
  token-tile HMMA attention, an L2-reuse-aware expert-major CTA
  schedule, and a flat activation pool: ~1,010 tok/s cold prefill at
  12k on a GB10 (v0.5.0 stamp), 2.43x upstream at 2k and 3.30x at 64k
  on the same box, gguf, and harness, and a 515K-token admit sustained
  at 776 tok/s.
- **Decode rebuilt the same way.** Per-layer CUDA-graph capture at
  every depth, head-group flash-decode for dense and indexed attention,
  a tensor-core indexer scorer, and DSpark lossless speculative decode
  governed by a terminal yield-quench controller: 1.33-1.47x upstream
  across the 2k-128k frontier (v0.4.1 stamp), deep decode 45.7 ms/tok
  at 240K (v0.5.0 stamp).
- **Memory truth (v0.6).** One governor account for every allocation,
  every floor and margin measured, and a continuous self-audit of the
  ledger since v0.6.2: measured to 3M tokens of active context on one
  128 GB Spark (2.26M at zero config), typed refusals, a 4 GiB floor,
  idle trim. See [Memory and capacity](#memory-and-capacity).
- **Agent robustness (v0.6.3.1-v0.6.5).** A field report of agent tasks
  dying at depth was root-caused to three mechanisms and fixed with
  receipts at each step: client effort fields compat-map away from the
  checkpoint's injected preambles, deep tool results carry a protocol
  reminder by default (0/72 slips vs 6/72 without at the captured
  failure states), and two opt-in levers cover the rest. A previously
  failing SWE-rebench task now resolves at stock defaults, matching
  llama.cpp's outcome while keeping DeepSeek's reference format. See
  [Replayed reasoning and agent-loop robustness](#replayed-reasoning-and-agent-loop-robustness).
- **Ops.** A resident weight server imports the 81 GiB model into
  engine processes in seconds (VMM-backed IPC, manifest with a
  content-identity check), and standalone boots build the same aligned
  fast-path artifacts in-process at load, so both setups serve from the
  fast tier and say so on one boot line.

Every performance claim comes from same-boot A/B runs with SM-clock
logging; every default flip passed bit- or value-parity plus eval
slices with proven engagement of the changed path; the full quality
suite (GSM8K, MMLU, HumanEval, MBPP, IFEval, needle) is re-stamped at
each release commit. The release gates are standing scripts in
`speed-bench/`.

The v0.5 context-frontier sweep compares the ship defaults against the
v0.4.1 line on the GB10; below it, the v0.1.0 chart against upstream
main on both reference machines (kept for history):

![v0.5 context-frontier sweep](speed-bench/v050_sweep_overlay.svg)

![v0.1.0 context-frontier sweep](speed-bench/v010_sweep_overlay.svg)

What is *not* an optimization target: the Metal/macOS path, the CLI,
the agent, and the GGUF tooling are inherited from upstream and kept
building and passing their vectors (disk KV persistence is inherited
too, though the fork extends it with packed-native payloads and the
durable bank tier, while older checkpoints stay readable). Metal
correctness on the fork's serving paths is **community-maintained**:
contributions are welcome (see `METAL_DSPARK.md` for the
community-contributed DSpark drafter port), gated here by compile plus
the isolated Metal kernel regressions; end-to-end Metal measurements
are the contributors' own, as no high-memory Metal machine is on the
fork's test bench. Upstream credit for the engine this fork stands on
is gladly given: the inherited sections of this README (CLI, native
agent, distributed inference, disk KV cache, and more) describe that
shared foundation.

## Speed

These are single-run CLI numbers with `--ctx 32768`, `--nothink`, greedy
decoding, and `-n 256`. The short prompt is a normal small Italian story
prompt. The long prompts exercise chunked prefill plus long-context decode.
Mac entries use Metal; NVIDIA entries use CUDA. Q4 requires the
larger-memory machine class, so M3 Max Q4 numbers are `N/A`.

| Machine | Quant | Prompt | Prefill | Generation |
| --- | ---: | ---: | ---: | ---: |
| MacBook Pro M3 Max, 128 GB | q2 | short | 58.52 t/s | 26.68 t/s |
| MacBook Pro M3 Max, 128 GB | q2 | 11709 tokens | 250.11 t/s | 21.47 t/s |
| MacBook Pro M3 Max, 128 GB | q4 | short | N/A | N/A |
| MacBook Pro M3 Max, 128 GB | q4 | long | N/A | N/A |
| MacBook Pro M5 Max, 128 GB | q2 | short | 87.25 t/s | 34.27 t/s |
| MacBook Pro M5 Max, 128 GB | q2 | 11707 tokens | 463.44 t/s | 25.90 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | short | 84.43 t/s | 36.86 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | 11709 tokens | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | short | 78.95 t/s | 35.50 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | 12018 tokens | 448.82 t/s | 26.62 t/s |
| Mac Studio M3 Ultra, 512 GB | PRO q2 | 32768 tokens | 138.82 t/s | 9.56 t/s |
| RTX PRO 6000 Blackwell, 96 GB | q2 | short | 85.21 t/s | 53.28 t/s |
| RTX PRO 6000 Blackwell, 96 GB | q2 | 12461 tokens | 1920.66 t/s | 41.10 t/s |
| DGX Spark GB10, 128 GB | q2 | 2048 tokens | 989.95 t/s | 22.66 t/s |
| DGX Spark GB10, 128 GB | q2 | 14336 tokens | 1008.01 t/s | 19.69 t/s |

![M3 Max t/s](speed-bench/m3_max_ts.svg)
![PRO model M3 Ultra t/s](speed-bench/pro_model_m3_ultra_ts.svg)

## Motivations

Now, back at this project. Why do we believe DeepSeek V4 Flash deserves a
standalone engine? Because after comparing it with powerful smaller dense
models, we can report that:

1. DeepSeek V4 Flash is the practical target of the project: it can run on
   96/128GB machines while still feeling much larger than local dense models.
2. DeepSeek V4 PRO is supported too, as a side path for 512GB Mac Studio class
   machines. It is heavier, but it shares the same engine ideas and can be
   useful when the hardware is available.
3. In thinking mode, if you avoid *max thinking*, Flash produces a thinking
   section that is a lot shorter than other models, even 1/5 of other models in
   many cases, and crucially, the thinking section length is **proportional to
   the problem complexity**. This makes DeepSeek V4 Flash usable with thinking
   enabled when other models are practically impossible to use in the same
   conditions.
4. The models feature a context window of **1 million tokens**.
5. Being so large, Flash knows more things if you go sampling at the edge of
   knowledge. For instance asking about Italian show or political questions soon
   uncovers that 284B parameters are a lot more than 27B or 35B parameters. PRO
   pushes further when you can run it.
6. Flash writes much better English and Italian. It *feels* a quasi-frontier
   model. PRO is stronger still, especially for tasks such as translation.
7. The KV cache is incredibly compressed, allowing long context inference on
   local computers and **on disk KV cache persistence**. The compression isn't
   only about fitting in RAM: in our testing the model's speed survives deep
   contexts (>100k tokens) far better than similarly-sized recent peers like
   MiMo-V2.5, so long-context inference stays fast as the window fills.
8. Both DeepSeek V4 variants work well with 2-bit quantization, if quantized in
   a special way (read later). This allows Flash to run on MacBooks with 128GB
   of RAM (and many people reported it working with 96GB as well, even at 250k
   context window!), and PRO on 512GB machines.
9. We expect DeepSeek to release **updated versions of V4 Flash and PRO** in the
   future, even better than the current ones.

That said, a few important things about this project:

* The local inference landscape contains many excellent projects, but new models are released continuously, and the attention immediately gets captured by the next model to implement. This project takes a deliberately narrow bet: one model at a time, official-vector validation (logits obtained with the official implementation), long-context tests, and enough agent integration to know if it really works. The exact model may change as the landscape evolves, but the constraint remains: local inference credible on high end personal machines or Mac Studios, starting from 96/128GB of memory.
* This software is developed with **strong assistance from GPT 5.5** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.
* This implementation is based on the idea that compressed KV caches like the one of DeepSeek v4 and the fast SSD disks of modern MacBooks should change our idea that KV cache belongs to RAM. **The KV cache is actually a first-class disk citizen**.
* Our vision is that local inference should be a set of three things working well together, out of the box: A) inference engine with HTTP API + B) GGUF specially crafted to run well under a given engine and given assumptions + C) testing and validation with coding agents implementations. This inference engine only runs with the GGUF files provided. It gets tested against officially obtained logits at different context sizes. This project exists because we wanted to make one local model feel finished end to end, not just runnable. However this is beta quality code, so probably we are not still there.
* The optimized graph path targets **Metal on macOS** and **CUDA on Linux**. The CPU path is only for correctness checks and model/tokenizer diagnostics. For CPU-only Linux builds, use `make cpu`; it builds the normal `./ds4` and `./ds4-server` binaries without CUDA or Metal. On macOS, **warning: current macOS versions have a bug in the virtual memory implementation that will crash the kernel** if you try to run the CPU code. Remember? Software sucks. It was not possible to fix the CPU inference to avoid crashing, since each time you have to restart the computer, which is not funny. Help us, if you have the guts.
* The project supports both Flash and PRO variants, but Flash remains the main
  focus because it is the model that makes sense on 96/128GB personal machines.
  **PRO support is experimental**: it is useful and welcome, but today it is
  naturally limited to people with 512GB Mac Studio class hardware.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## License and attribution

This fork keeps upstream ds4's MIT license. The batched-serving fork
modifications are Copyright (c) 2026 Entrpi <entrpi@proton.me>, MIT. Lineage
of the code in this tree, so reusers know who built what:

- The engine, CLI/agent, Metal path, and session serving are
  [antirez/ds4](https://github.com/antirez/ds4) (upstream).
- The quantized-matmul kernel family under `cuda/mmq/` is vendored from
  llama.cpp (MIT); the exact upstream pin and per-file inventory are in
  `cuda/mmq/VENDOR.md`.
- The batched server, continuous-batching engine, KV banks, D2R/MMQ prefill
  tiers, token-tile HMMA attention, indexer/scorer work, and DSpark
  integration on the CUDA path are this fork's additions.
- [xangel82/DS4-GB10-GX10-DSpark-CUDA](https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA)
  (Marco Palaferri, MIT) is a sibling fork built on both upstream ds4 and
  this fork's kernel work, and we re-integrate select portions of his work
  in return. Where a change follows his design it is credited in the commit
  message, and where his code is adapted directly it also carries his
  copyright notice in the source headers -- for example, the fused gate/up
  prefill pipeline in `cuda/mmq/ds4_mmq_d2r.cu` and `cuda/mmq/ds4_mmq.cu`
  (his `910501e`, our `da027a1`).

If you reuse this fork's modifications, keep this notice together with the
MIT license text, per upstream's terms.

## Status

The code and GGUF files are to be considered of **beta quality** because
inference and model serving is a complicated matter and all this exists
only for a few days. It will take months to reach a more stable form.
However, we try to keep the project in a usable state, and we are making
progress. If you have issues, make sure to use `--trace` to log the
sessions, and open issues including the full trace.

The `ds4-agent` is alpha quality, the project was later added.

## More Documentation

If you are looking for very specific things, we have other
sub-README files:

- [CONTRIBUTING.md](CONTRIBUTING.md): correctness and speed regression testing
  guide for contributors. **Read this before sending a pull request**.
- [gguf-tools/README.md](gguf-tools/README.md): offline GGUF generation,
  imatrix collection, quantization tooling, and quality checks.
- [gguf-tools/imatrix/README.md](gguf-tools/imatrix/README.md): how the
  routed-MoE imatrix is collected and used.
- [gguf-tools/imatrix/dataset/README.md](gguf-tools/imatrix/dataset/README.md):
  how the calibration prompt corpus is generated.
- [gguf-tools/quality-testing/README.md](gguf-tools/quality-testing/README.md):
  how local GGUFs are scored against official DeepSeek V4 Flash/PRO continuations.
- [dir-steering/README.md](dir-steering/README.md): directional steering data,
  vector generation, and usage.
- [misc/cuda-env-vars.md](misc/cuda-env-vars.md): CUDA backend env-var
  reference and Q8_0 dispatcher behavior.
- [misc/cuda-mtp/README.md](misc/cuda-mtp/README.md): CUDA MTP enablement,
  DGX Spark / GB10 notes, optimization flags, and benchmark method.
- [misc/proof-harness/README.md](misc/proof-harness/README.md): generalized
  engine proof harness and weight-server lifecycle.
- [speed-bench/README.md](speed-bench/README.md): benchmark commands, charts,
  and CSV generation.
- [tests/test-vectors/README.md](tests/test-vectors/README.md): official
  continuation vectors used for regression checks.
