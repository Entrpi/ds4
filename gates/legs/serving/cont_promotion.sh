#!/bin/bash
# serving/cont_promotion.sh — v0.5.6 API Inc 3 live gate: batched-lane surface
# promotions (plan §5 Inc 3, §6.3 route engagement + semantic equivalence).
#
# Inc 3a legs (boot A = zero-config, cont on):
#   anth_cont      buffered /v1/messages rides the CONTINUOUS lane: native
#                  message schema AND route_requests{anthropic_messages,
#                  continuous} increments (the promotion, observed)
#   anth_stop_cont stop_sequences match on the CONT scan sites: stop_reason
#                  stop_sequence + the matched text, lane continuous
#   anth_warm_cont turn-2 continuation built from the SERVED t1 answer
#                  (GATE-DESIGN LAW: t1 repeats never warm-match): usage
#                  cache_read_input_tokens > 0 AND == timings prefill_cached
#                  -- the Inc 2d engine-verdict law on the promoted surface
#   anth_zero_serial   explicit max_tokens:0 (prewarm) STAYS SERIAL:
#                  need_prefill_only decision + serial lane entry
#   anth_tools_serial  tool-bearing buffered request STAYS SERIAL:
#                  need_continuation_publish decision recorded
#   oa_cont_sanity an OpenAI chat request still rides continuous
#
# Inc 3b legs (same boot A):
#   resp_cont      buffered /v1/responses rides continuous; ALSO proves the
#                  reasoning_tokens gap fix (cont now retokenizes at
#                  finalize exactly like serial -- 0 would be the old gap)
#   resp_warm_cont the client-frame usage law on the Responses surface:
#                  input - cached == timings prefill_tokens, cached > 0
#   resp_tools_serial  Responses tool gen stays serial (continuation
#                  publish)
#
# Inc 3c legs (boot C = fresh zero-config boot; the seed-floor trim):
#   oa_zero_cont   chat max_tokens:0 SERVED on cont: completion_tokens 0,
#                  empty content, finish length.  Pre-fix (measured
#                  2026-08-07) the cont lane leaked the engine's seed-floor
#                  token: completion_tokens=1, content "Okay", where serial
#                  answered 0/"" for the same request.
#   resp_zero_cont responses max_output_tokens:0 SERVED on cont:
#                  incomplete + output_tokens 0 + output []
#   oa_stream_zero streaming zero on cont: role preamble + finish length +
#                  [DONE] and NO content/reasoning delta (pre-fix the seed
#                  streamed out as a reasoning_content delta)
#
# Inc 3d legs (boot D = DS4_SERVER_CONTINUOUS=0, the static-coalescing lane):
#   anth_static    a thinking-disabled buffered Anthropic request is WRITTEN
#                  BY THE STATIC BATCH (lane cell +1 exact -- a single-member
#                  collapse records serial and fails the leg) with the
#                  native message shape
#   resp_static    same for Responses (reasoning effort none)
#   oa_static      the same blast forms ONE mixed three-surface batch
#                  (blocker occupies the worker; gather takes the queued
#                  trio) -- static_no_cont decisions recorded
#
# Kill switches (boot B = both DS4_SERVER_CONT_ANTHROPIC=0 and
# DS4_SERVER_CONT_RESPONSES=0, plan §7; the reason-table unit proves they
# are per-surface independent):
#   anth_kill      the same plain request runs SERIAL (surface decision
#                  recorded, zero new cont entries) and still serves the
#                  native 200 -- semantic equivalence with boot A asserted
#                  loosely (same stop_reason, nonempty text; cont temp-0 is
#                  documented not run-to-run deterministic, so no byte diff)
#   resp_kill      same for Responses (zero new cont entries, completed)
#
# Runs FROM the Mac over SSH like the other gates.  Boots kill any running
# ds4-server on $R.  End state: ds4-server killed, box left free.
#
# Env overrides: R (${DS4_BOX33:?}) BINDIR (${DS4_BOX_TREE:?})
#                PORT (8000) TUNNEL_PORT (18000) CTX (16384)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/cont_promotion_gate
OUT=${OUT:-/tmp/cont_promotion_gate_$$}
mkdir -p "$OUT"
BASE="http://127.0.0.1:$TUNNEL_PORT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; exit 1; }

tunnel_up(){
  curl -s -m 5 "$BASE/v1/models" >/dev/null 2>&1 && return 0
  ssh -f -N -L "$TUNNEL_PORT:127.0.0.1:$PORT" "$R" 2>/dev/null || true
  sleep 2
  curl -s -m 10 "$BASE/v1/models" >/dev/null 2>&1
}

wait_mem(){ # $1=min MemAvailable GiB
  local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge "$1" ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable ${got:-?}G never reached ${1}G"
    sleep 5
  done
}

boot(){
  SRV=$RWORK/srv.log
  log "boot: killing old ds4-server on $R"
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; mkdir -p $RWORK; rm -f /tmp/ds4.lock; exit 0"
  wait_mem 100
  ssh "$R" ": > $SRV; cd $BINDIR; env ${BOOT_ENV:-} setsid nohup ./ds4-server -c $CTX --port $PORT \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
      sleep 3
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" 2>/dev/null | tr '\n' ' ')"
    fi
    sleep 10; n=$((n+10)); [ $n -ge 1200 ] && fail "boot timeout"
  done
  tunnel_up || fail "tunnel :$TUNNEL_PORT unreachable"
  log "boot: up"
}

post(){
  curl -s -m 180 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
has(){ grep -q "$2" "$OUT/$1.json" || fail "$1: missing [$2] in $(head -c 300 "$OUT/$1.json")"; }
code_is(){ [ "$2" = "$3" ] || fail "$1: HTTP $2, want $3 ($(head -c 300 "$OUT/$1.json"))"; }
# route_requests / route_decisions cells from /metrics
lane(){ curl -s -m 10 "$BASE/metrics" | grep -F "surface=\"$1\",lane=\"$2\"" | grep -oE '[0-9]+$'; }
decision(){ curl -s -m 10 "$BASE/metrics" | grep -F "ds4_route_decisions_total{reason=\"$1\"}" | grep -oE '[0-9]+$'; }

# ==================== boot A1: the Anthropic block (Inc 3a) ================
# DS4_MEM_FLOOR_GB=2: on GB10 the weights mmap-fault in as the boot serves,
# so MemAvailable declines request by request and the default 4 GiB floor
# starts rejecting cont admissions ~8-10 requests into a 16k boot (measured
# 2026-08-07: bank credits 42 MiB, usable 0.0 beyond the floor; rejected
# rows fall back SERIAL, which has no bank-warm records -- the first 3b run
# failed its warm leg on exactly this, not on any warm-match defect).  This
# gate proves ROUTING + projection; memory governance has its own gate
# (memory/floor.sh).  The reject-delta guards below keep the distinction
# loud if the floor still engages, and each surface block runs on its own
# fresh boot (see boot A2) so no decisive leg lands on an exhausted boot.
# Inc 5b LAW: this gate proves ROUTING; the continuation-retention policy
# (grace/pin sheds after a tool-turn publication, plan 4.6) has its own gate
# (serving/cont_registry.sh boot B).  Zero it here so the tools-serial legs'
# published records cannot shed the later serial legs.
BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_CONT_GRACE_S=0 DS4_CONT_PIN_DEADLINE_S=0" boot
rejects(){ curl -s -m 10 "$BASE/metrics" | grep -F "ds4_cont_admit_rejects_total" | grep -oE '[0-9]+$'; }
serials(){ curl -s -m 10 "$BASE/metrics" | grep -F "ds4_requests_serial_total" | grep -oE '[0-9]+$'; }
# LANE-ENTRY TRAP (found 2026-08-07): route_requests counts lane ENTRIES --
# a floor-rejected cont attempt increments the continuous cell and THEN
# re-enters serial, so "cont counter grew" alone cannot prove cont SERVED
# the request.  Promotion legs must also assert requests_serial_total did
# not move across the leg.

# anth_cont: the promotion itself, schema + lane observed (SERVED on cont:
# the serial-total guard closes the lane-entry trap).
ac0=$(lane anthropic_messages continuous)
sl0=$(serials)
c=$(post anth_cont /v1/messages '{"model":"m","max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one ocean."}]}')
code_is anth_cont "$c" 200
has anth_cont '"type":"message"'
has anth_cont '"role":"assistant"'
has anth_cont '"stop_reason"'
has anth_cont '"input_tokens"'
ac1=$(lane anthropic_messages continuous)
sl1=$(serials)
[ "${ac1:-0}" -gt "${ac0:-0}" ] || fail "anth_cont: anthropic continuous lane never incremented (${ac0:-?} -> ${ac1:-?})"
[ "${sl1:-0}" -eq "${sl0:-0}" ] || fail "anth_cont: request FELL BACK to serial (${sl0:-?} -> ${sl1:-?}) -- cont entry alone is the lane-entry trap"
log "anth_cont PASS (native message SERVED on the continuous lane; lane counter ${ac0:-0} -> $ac1)"

# anth_stop_cont: the cont scan sites produce native stop honesty (2c pt2
# plumbed matched_stop on cont for exactly this increment).
sc0=$(lane anthropic_messages continuous)
c=$(post anth_stop_cont /v1/messages '{"model":"m","max_tokens":128,"temperature":0,"stop_sequences":["beta"],"messages":[{"role":"user","content":"Repeat exactly: alpha beta gamma"}]}')
code_is anth_stop_cont "$c" 200
has anth_stop_cont '"stop_reason":"stop_sequence"'
has anth_stop_cont '"stop_sequence":"beta"'
sc1=$(lane anthropic_messages continuous)
[ "${sc1:-0}" -gt "${sc0:-0}" ] || fail "anth_stop_cont: stop leg did not ride continuous (${sc0:-?} -> ${sc1:-?})"
log "anth_stop_cont PASS (stop_sequence + matched text from the cont scan)"

# anth_warm_cont: engine-verdict usage (2d) on the promoted surface.
c=$(post anth_warm_t1 /v1/messages '{"model":"m","max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one planet."}]}')
code_is anth_warm_t1 "$c" 200
python3 - "$OUT/anth_warm_t1.json" > "$OUT/anth_warm_t2_req.json" <<'PY' || fail "anth_warm_cont: t2 build failed"
import json, sys
t1 = json.load(open(sys.argv[1]))
ans = "".join(b.get("text", "") for b in t1["content"] if b.get("type") == "text")
assert ans.strip(), "t1 answer empty"
msgs = [{"role": "user", "content": "Name one planet."},
        {"role": "assistant", "content": ans},
        {"role": "user", "content": "Name another."}]
print(json.dumps({"model": "m", "max_tokens": 48, "temperature": 0, "messages": msgs}))
PY
rej0=$(rejects)
c=$(curl -s -m 180 -o "$OUT/anth_warm_cont.json" -w '%{http_code}' "$BASE/v1/messages" \
     -H 'Content-Type: application/json' -d @"$OUT/anth_warm_t2_req.json")
code_is anth_warm_cont "$c" 200
rej1=$(rejects)
[ "${rej1:-0}" -eq "${rej0:-0}" ] || fail "anth_warm_cont ENVIRONMENT: cont admission rejected on the memory floor (${rej0:-?} -> ${rej1:-?}); box page-cache state, not a warm-match defect -- reboot the box"
# Inc 3a frame law (found HERE, first run): the engine verdict counts
# EFFECTIVE-prompt positions -- a thinking-warm admit replays the bank's
# committed reasoning, so the verdict frame exceeds the client's terse t2
# prompt (measured: 16-token prompt vs 87 cached + 6 computed).  Usage is
# rendered in the CLIENT frame; the structural tie to engine stats is:
# input + cache_creation == timings prefill_tokens (the uncached portion is
# exactly what the engine computed fresh), with cache_read > 0 proving warm
# engagement.
ur=$(grep -oE '"cache_read_input_tokens":[0-9]+' "$OUT/anth_warm_cont.json" | head -1 | grep -oE '[0-9]+$')
ui=$(grep -oE '"input_tokens":[0-9]+' "$OUT/anth_warm_cont.json" | head -1 | grep -oE '[0-9]+$')
uw=$(grep -oE '"cache_creation_input_tokens":[0-9]+' "$OUT/anth_warm_cont.json" | head -1 | grep -oE '[0-9]+$')
pf=$(grep -oE '"prefill_tokens":[0-9]+' "$OUT/anth_warm_cont.json" | head -1 | grep -oE '[0-9]+$')
[ -n "$ur" ] && [ -n "$ui" ] && [ -n "$uw" ] && [ -n "$pf" ] \
  || fail "anth_warm_cont: usage/timings fields missing (read=${ur:-?} input=${ui:-?} creation=${uw:-?} prefill=${pf:-?})"
[ $((ui + uw)) -eq "$pf" ] \
  || fail "anth_warm_cont: uncached portion $((ui + uw)) != engine computed $pf (read=$ur)"
[ "${ur:-0}" -gt 0 ] || fail "anth_warm_cont: t2 continuation reported zero cache (no warm engagement)"
log "anth_warm_cont PASS (client frame: read=$ur input=$ui creation=$uw; uncached==computed=$pf)"

# anth_zero_serial: the prewarm contract stays serial (route_decide sends
# NEED_PREFILL_ONLY to serial before the surface stage).
pz0=$(decision need_prefill_only)
as0=$(lane anthropic_messages serial)
c=$(post anth_zero_serial /v1/messages '{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"prewarm this prompt"}]}')
code_is anth_zero_serial "$c" 200
has anth_zero_serial '"output_tokens":0'
pz1=$(decision need_prefill_only)
as1=$(lane anthropic_messages serial)
[ "${pz1:-0}" -gt "${pz0:-0}" ] || fail "anth_zero_serial: need_prefill_only decision missing (${pz0:-?} -> ${pz1:-?})"
[ "${as1:-0}" -gt "${as0:-0}" ] || fail "anth_zero_serial: serial lane entry missing (${as0:-?} -> ${as1:-?})"
log "anth_zero_serial PASS (prewarm stays serial; decision + lane recorded)"

# anth_tools_serial: tool generation keeps the serial continuation contract.
tp0=$(decision need_continuation_publish)
c=$(post anth_tools_serial /v1/messages '{"model":"m","max_tokens":128,"temperature":0,"tools":[{"name":"get_time","description":"Get the current time.","input_schema":{"type":"object","properties":{}}}],"messages":[{"role":"user","content":"What time is it?"}]}')
code_is anth_tools_serial "$c" 200
tp1=$(decision need_continuation_publish)
[ "${tp1:-0}" -gt "${tp0:-0}" ] || fail "anth_tools_serial: need_continuation_publish decision missing (${tp0:-?} -> ${tp1:-?})"
log "anth_tools_serial PASS (tool gen stays serial with the continuation-publish reason)"

# oa_cont_sanity: the incumbent surface still rides continuous.
oc0=$(lane openai_chat continuous)
c=$(post oa_cont_sanity /v1/chat/completions '{"max_tokens":32,"temperature":0,"messages":[{"role":"user","content":"Say ok."}]}')
code_is oa_cont_sanity "$c" 200
oc1=$(lane openai_chat continuous)
[ "${oc1:-0}" -gt "${oc0:-0}" ] || fail "oa_cont_sanity: openai chat lost the continuous lane (${oc0:-?} -> ${oc1:-?})"
log "oa_cont_sanity PASS"

# ================== boot A2: the Responses block (Inc 3b) ==================
# Its own boot ON PURPOSE: a 16k boot funds roughly 8-10 cont admissions
# before weight page-faulting walks MemAvailable down to the floor and
# admissions start rejecting to serial (measured 2026-08-07: box healthy at
# 121 GB the moment the server exits -- this is per-boot decline, not box
# state).  Each surface block gets a fresh funded window so its decisive
# warm leg is never adjudicated on an exhausted boot.
BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_CONT_GRACE_S=0 DS4_CONT_PIN_DEADLINE_S=0" boot

# resp_cont: promotion + the reasoning_tokens gap fix (cont now does the
# same finalize retokenize serial does; 0 would be the old gap WHEN the
# model actually reasoned).  Whether the model opens a <think> block at all
# is its own temp-0-jittery choice (measured on one prompt: 72 reasoning
# tokens one run, an unreasoned 36-token answer the next), and a buffered
# response exposes no reasoning ITEM to cross-check -- so the leg retries
# up to 3 times on a reasoning-inducing prompt and passes on the first
# attempt that reasoned (the finish_reason_gate attempt-loop pattern).
# Budget 400 so an opened block always closes.
rc0=$(lane openai_responses continuous)
sl0=$(serials)
rt=0
for attempt in 1 2 3; do
  c=$(post resp_cont /v1/responses '{"max_output_tokens":400,"temperature":0,"input":"Which is larger, 3^4 or 4^3? Reply with just the winning expression."}')
  code_is resp_cont "$c" 200
  has resp_cont '"status":"completed"'
  rt=$(grep -oE '"reasoning_tokens":[0-9]+' "$OUT/resp_cont.json" | head -1 | grep -oE '[0-9]+$')
  [ "${rt:-0}" -gt 0 ] && break
  # A zero can be an unreasoned answer (visible tokens ~= output_tokens) or
  # the retokenize gap (hidden tokens counted, none attributed).  Only
  # retry the former; the latter is the defect this leg exists to catch.
  ot=$(grep -oE '"output_tokens":[0-9]+' "$OUT/resp_cont.json" | head -1 | grep -oE '[0-9]+$')
  [ -n "$ot" ] && [ "$ot" -gt 24 ] && \
    fail "resp_cont: output_tokens=$ot with reasoning_tokens=0 -- hidden tokens unattributed (the cont retokenize gap)"
  log "resp_cont attempt $attempt: model answered without reasoning (output_tokens=${ot:-?}); retrying"
done
[ "${rt:-0}" -gt 0 ] || fail "resp_cont: reasoning_tokens=0 on 3 reasoned-prompt attempts"
rc1=$(lane openai_responses continuous)
sl1=$(serials)
[ "${rc1:-0}" -gt "${rc0:-0}" ] || fail "resp_cont: responses continuous lane never incremented (${rc0:-?} -> ${rc1:-?})"
[ "${sl1:-0}" -eq "${sl0:-0}" ] || fail "resp_cont: request FELL BACK to serial (${sl0:-?} -> ${sl1:-?}) -- reasoning_tokens came from the serial writer, not cont"
log "resp_cont PASS (completed SERVED on the continuous lane; reasoning_tokens=$rt; lane ${rc0:-0} -> $rc1)"

# resp_warm_cont: the client-frame usage law on the Responses surface.
c=$(post resp_warm_t1 /v1/responses '{"max_output_tokens":400,"temperature":0,"input":"Name one planet."}')
code_is resp_warm_t1 "$c" 200
python3 - "$OUT/resp_warm_t1.json" > "$OUT/resp_warm_t2_req.json" <<'PY' || fail "resp_warm_cont: t2 build failed"
import json, sys
t1 = json.load(open(sys.argv[1]))
ans = ""
for item in t1["output"]:
    if item.get("type") == "message":
        for blk in item.get("content", []):
            if blk.get("type") == "output_text":
                ans += blk.get("text", "")
assert ans.strip(), "t1 answer empty"
msgs = [{"role": "user", "content": "Name one planet."},
        {"role": "assistant", "content": ans},
        {"role": "user", "content": "Name another."}]
print(json.dumps({"max_output_tokens": 48, "temperature": 0, "input": msgs}))
PY
rej0=$(rejects)
c=$(curl -s -m 180 -o "$OUT/resp_warm_cont.json" -w '%{http_code}' "$BASE/v1/responses" \
     -H 'Content-Type: application/json' -d @"$OUT/resp_warm_t2_req.json")
code_is resp_warm_cont "$c" 200
rej1=$(rejects)
[ "${rej1:-0}" -eq "${rej0:-0}" ] || fail "resp_warm_cont ENVIRONMENT: cont admission rejected on the memory floor (${rej0:-?} -> ${rej1:-?}); box page-cache state, not a warm-match defect -- reboot the box"
ri=$(grep -oE '"input_tokens":[0-9]+' "$OUT/resp_warm_cont.json" | head -1 | grep -oE '[0-9]+$')
rcch=$(grep -oE '"cached_tokens":[0-9]+' "$OUT/resp_warm_cont.json" | head -1 | grep -oE '[0-9]+$')
rpf=$(grep -oE '"prefill_tokens":[0-9]+' "$OUT/resp_warm_cont.json" | head -1 | grep -oE '[0-9]+$')
[ -n "$ri" ] && [ -n "$rcch" ] && [ -n "$rpf" ] \
  || fail "resp_warm_cont: usage/timings fields missing (input=${ri:-?} cached=${rcch:-?} prefill=${rpf:-?})"
[ $((ri - rcch)) -eq "$rpf" ] \
  || fail "resp_warm_cont: uncached portion $((ri - rcch)) != engine computed $rpf"
[ "${rcch:-0}" -gt 0 ] || fail "resp_warm_cont: zero cache (no warm engagement)"
log "resp_warm_cont PASS (client frame: input=$ri cached=$rcch; uncached==computed=$rpf)"

# resp_tools_serial: Responses tool generation keeps the serial contract.
rt0=$(decision need_continuation_publish)
c=$(post resp_tools_serial /v1/responses '{"max_output_tokens":128,"temperature":0,"tools":[{"type":"function","name":"get_time","description":"Get the current time.","parameters":{"type":"object","properties":{}}}],"input":"What time is it?"}')
code_is resp_tools_serial "$c" 200
rt1=$(decision need_continuation_publish)
[ "${rt1:-0}" -gt "${rt0:-0}" ] || fail "resp_tools_serial: need_continuation_publish decision missing (${rt0:-?} -> ${rt1:-?})"
log "resp_tools_serial PASS (tool gen stays serial with the continuation-publish reason)"

# ===================== boot C: the Inc 3c zero-budget block ================
# A zero-budget request is the ONLY shape where the engine emits a token the
# wire must not show (the cont loop cannot retire a row without sampling --
# the ds4.c seed floor).  These legs hold the trimmed contract on the plain
# buffered path, the promoted Responses path, and the stream path, each with
# the LANE-ENTRY-TRAP guards: a serial fallback would ALSO answer 0/"" (the
# serial loop never samples at zero), so only the lane + serial counters
# prove the CONT lane produced the honored shape.
BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_CONT_GRACE_S=0 DS4_CONT_PIN_DEADLINE_S=0" boot

oc0=$(lane openai_chat continuous); sl0=$(serials)
c=$(post oa_zero_cont /v1/chat/completions '{"max_tokens":0,"temperature":0,"messages":[{"role":"user","content":"hi"}]}')
code_is oa_zero_cont "$c" 200
has oa_zero_cont '"completion_tokens":0'
has oa_zero_cont '"content":""'
has oa_zero_cont '"finish_reason":"length"'
oc1=$(lane openai_chat continuous); sl1=$(serials)
[ "${oc1:-0}" -gt "${oc0:-0}" ] || fail "oa_zero_cont: no continuous lane entry (${oc0:-?} -> ${oc1:-?})"
[ "${sl1:-0}" -eq "${sl0:-0}" ] || fail "oa_zero_cont: fell back to serial (${sl0:-?} -> ${sl1:-?})"
log "oa_zero_cont PASS (seed trimmed: 0 tokens, empty content, length; SERVED on cont)"

rc0=$(lane openai_responses continuous); sl0=$(serials)
c=$(post resp_zero_cont /v1/responses '{"max_output_tokens":0,"temperature":0,"input":"prewarm this prompt"}')
code_is resp_zero_cont "$c" 200
has resp_zero_cont '"status":"incomplete"'
has resp_zero_cont '"output_tokens":0'
has resp_zero_cont '"output":\[\]'
rc1=$(lane openai_responses continuous); sl1=$(serials)
[ "${rc1:-0}" -gt "${rc0:-0}" ] || fail "resp_zero_cont: no continuous lane entry (${rc0:-?} -> ${rc1:-?})"
[ "${sl1:-0}" -eq "${sl0:-0}" ] || fail "resp_zero_cont: fell back to serial (${sl0:-?} -> ${sl1:-?})"
log "resp_zero_cont PASS (incomplete, zero output, empty output array; SERVED on cont)"

oc0=$(lane openai_chat continuous); sl0=$(serials)
curl -s -N -m 60 -o "$OUT/oa_stream_zero.sse" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"max_tokens":0,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}'
grep -q '"finish_reason":"length"' "$OUT/oa_stream_zero.sse" || fail "oa_stream_zero: no length finish chunk"
grep -q 'data: \[DONE\]' "$OUT/oa_stream_zero.sse" || fail "oa_stream_zero: no [DONE]"
! grep -q '"reasoning_content"' "$OUT/oa_stream_zero.sse" || fail "oa_stream_zero: seed token leaked as a reasoning delta"
! grep -qE '"content":"[^"]' "$OUT/oa_stream_zero.sse" || fail "oa_stream_zero: seed token leaked as a content delta"
oc1=$(lane openai_chat continuous); sl1=$(serials)
[ "${oc1:-0}" -gt "${oc0:-0}" ] || fail "oa_stream_zero: no continuous lane entry (${oc0:-?} -> ${oc1:-?})"
[ "${sl1:-0}" -eq "${sl0:-0}" ] || fail "oa_stream_zero: fell back to serial (${sl0:-?} -> ${sl1:-?})"
log "oa_stream_zero PASS (no delta leak; length + [DONE]; SERVED on cont)"

# ===================== boot D: the Inc 3d static scoping ===================
# DS4_SERVER_CONTINUOUS=0 reverts to W3/W4 static coalescing -- the lane the
# 3d row opens to the promoted surfaces.  The static lane serves only
# needs-free shapes, so every target request disables thinking explicitly
# (the server default effort is LOW = NEED_THINKING = cont/serial only).
# Determinism: COALESCE_WAIT defaults to 0 (gather takes only already-queued
# jobs), so a serial blocker occupies the worker while the three targets
# queue; the worker then gathers them into ONE mixed three-surface batch.
# generate_batch_jobs records a static lane entry PER JOB -- the exact +1 on
# the anth/resp static cells is the engagement proof (a single-member
# collapse records serial instead, and would fail those asserts).
BOOT_ENV="DS4_SERVER_CONTINUOUS=0 DS4_CONT_GRACE_S=0 DS4_CONT_PIN_DEADLINE_S=0" boot

as0=$(lane anthropic_messages static)
rs0=$(lane openai_responses static)
os0=$(lane openai_chat static)
dsc0=$(decision static_no_cont)
curl -s -m 180 -o "$OUT/static_blocker.json" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"max_tokens":96,"temperature":0,"thinking":false,"messages":[{"role":"user","content":"Count from 1 to 40, comma separated."}]}' &
sleep 1
curl -s -m 180 -o "$OUT/anth_static.json" "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","max_tokens":32,"temperature":0,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"Name one ocean."}]}' &
curl -s -m 180 -o "$OUT/resp_static.json" "$BASE/v1/responses" \
     -H 'Content-Type: application/json' \
     -d '{"max_output_tokens":32,"temperature":0,"reasoning":{"effort":"none"},"input":"Name one river."}' &
curl -s -m 180 -o "$OUT/oa_static.json" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"max_tokens":32,"temperature":0,"thinking":false,"messages":[{"role":"user","content":"Name one mountain."}]}' &
wait
grep -q '"type":"message"' "$OUT/anth_static.json" || fail "anth_static: no native message shape ($(head -c 200 "$OUT/anth_static.json"))"
grep -q '"stop_reason"' "$OUT/anth_static.json" || fail "anth_static: no stop_reason"
grep -q '"object":"response"' "$OUT/resp_static.json" || fail "resp_static: no native response shape ($(head -c 200 "$OUT/resp_static.json"))"
grep -q '"status":"' "$OUT/resp_static.json" || fail "resp_static: no status field"
grep -q '"object":"chat.completion"' "$OUT/oa_static.json" || fail "oa_static: no chat.completion shape"
as1=$(lane anthropic_messages static)
rs1=$(lane openai_responses static)
os1=$(lane openai_chat static)
dsc1=$(decision static_no_cont)
[ "${as1:-0}" -eq $(( ${as0:-0} + 1 )) ] || fail "anth_static: static lane entry missing (${as0:-?} -> ${as1:-?}; single-member collapse ran it serial -- the mixed batch never formed)"
[ "${rs1:-0}" -eq $(( ${rs0:-0} + 1 )) ] || fail "resp_static: static lane entry missing (${rs0:-?} -> ${rs1:-?})"
[ "${os1:-0}" -ge $(( ${os0:-0} + 1 )) ] || fail "oa_static: static lane entry missing (${os0:-?} -> ${os1:-?})"
[ "${dsc1:-0}" -ge $(( ${dsc0:-0} + 2 )) ] || fail "static block: static_no_cont decisions did not record (${dsc0:-?} -> ${dsc1:-?})"
log "anth_static PASS (native message written by the STATIC batch; lane ${as0:-0} -> $as1)"
log "resp_static PASS (native response written by the STATIC batch; lane ${rs0:-0} -> $rs1)"
log "oa_static PASS (mixed three-surface static batch; static_no_cont ${dsc0:-0} -> $dsc1)"

# ===================== boot B: the §7 kill switches ========================
BOOT_ENV="DS4_SERVER_CONT_ANTHROPIC=0 DS4_SERVER_CONT_RESPONSES=0 DS4_MEM_FLOOR_GB=2 DS4_CONT_GRACE_S=0 DS4_CONT_PIN_DEADLINE_S=0" boot
ks0=$(decision surface)
kc0=$(lane anthropic_messages continuous)
c=$(post anth_kill /v1/messages '{"model":"m","max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one ocean."}]}')
code_is anth_kill "$c" 200
has anth_kill '"type":"message"'
has anth_kill '"stop_reason"'
ks1=$(decision surface)
kc1=$(lane anthropic_messages continuous)
[ "${ks1:-0}" -gt "${ks0:-0}" ] || fail "anth_kill: surface decision missing under the kill switch (${ks0:-?} -> ${ks1:-?})"
[ "${kc1:-0}" -eq "${kc0:-0}" ] || fail "anth_kill: request rode continuous DESPITE the kill switch (${kc0:-?} -> ${kc1:-?})"
# Loose semantic equivalence vs boot A (documented cont temp-0 jitter: no
# byte compare; the lane must not change the protocol result class).  Both
# sides run 400-token budgets so each reliably EOS's -- a tight budget made
# one side stop_reason=max_tokens while the other end_turn'd (measured).
sr_a=$(grep -oE '"stop_reason":"[a-z_]+"' "$OUT/anth_cont.json" | head -1)
sr_b=$(grep -oE '"stop_reason":"[a-z_]+"' "$OUT/anth_kill.json" | head -1)
[ -n "$sr_a" ] && [ "$sr_a" = "$sr_b" ] || fail "anth_kill: stop_reason diverged across lanes ($sr_a vs $sr_b)"
grep -q '"type":"text"' "$OUT/anth_kill.json" || fail "anth_kill: no text block"
grep -q '"type":"text"' "$OUT/anth_cont.json" || fail "anth_kill: boot A response had no text block"
log "anth_kill PASS (kill switch = serial with the surface reason; same result class)"

# resp_kill: the Responses switch, independently.
rk0=$(lane openai_responses continuous)
c=$(post resp_kill /v1/responses '{"max_output_tokens":400,"temperature":0,"input":"Name one ocean."}')
code_is resp_kill "$c" 200
has resp_kill '"status":"completed"'
rk1=$(lane openai_responses continuous)
[ "${rk1:-0}" -eq "${rk0:-0}" ] || fail "resp_kill: request rode continuous DESPITE the kill switch (${rk0:-?} -> ${rk1:-?})"
log "resp_kill PASS (kill switch = serial; completed result class)"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT (server killed, $R left free)"
