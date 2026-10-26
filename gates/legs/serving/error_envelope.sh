#!/bin/bash
# serving/error_envelope.sh — v0.5.6 API Inc 2b live gate: endpoint-native error
# envelopes + the per-surface decode-budget matrix (plan §5 Inc 2 / §6.2;
# surface-matrix defect 3).
#
# ONE default zero-config boot serves every leg (all legs are HTTP-level):
#
#   envelope legs
#     anth_parse   POST /v1/messages garbage body -> HTTP 400 + the documented
#                  {"type":"error","error":{"type":"invalid_request_error",...
#                  envelope (NO OpenAI shape on the Anthropic surface)
#     anth_neg     /v1/messages max_tokens:-1 -> 400 native + field message
#     oa_neg       /v1/chat/completions max_tokens:-1 -> 400 OpenAI envelope
#     oa_neg2      chat max_completion_tokens:-2 -> the message names the
#                  field THE CLIENT SENT
#     comp_neg     /v1/completions max_tokens:-1 -> 400
#     resp_neg     /v1/responses max_output_tokens:-3 -> 400 names the field
#     not_found    GET /nope -> 404 OpenAI envelope (historical owner)
#
#   response_format typed refusal (v0.6.3 Inc 1; ds4-on-spark#10)
#     rf_text      chat response_format {"type":"text"} -> 200 (accepted)
#     rf_obj       chat json_object -> 400 typed message + OpenAI envelope
#     rf_schema    chat json_schema, nested schema before type -> 400
#     rf_comp      completions json_object -> 400
#     rf_resp      responses text.format json_schema -> 400 names text.format
#     rf_anth      messages output_format json_schema -> 400 native envelope
#     rf_after     normal chat on the same boot -> 200 (no wedge)
#
#   chunked request bodies (v0.6.3 Inc 2)
#     ch_chat      chat via Transfer-Encoding: chunked -> 200, zero contract
#     ch_anth      messages chunked -> 200 output_tokens 0
#     ch_comp      completions chunked -> 200 text_completion
#     ch_resp      responses chunked -> 200 output_tokens 0
#     ch_rf        chunked body carrying json_object -> the Inc 1 typed 400
#                  (proves the DECODED body reaches the parser end-to-end)
#
#   think-dial observability (v0.6.3 Inc 3)
#     th_high      reasoning_effort high -> 200; srv.log carries think=high
#                  AND think=low (the dial rides every progress line);
#                  ds4_requests_think_total{mode="high"} >= 1;
#                  /v1/stats think_modes family present with positive
#                  high/low counts
#
#   stop honesty (Inc 2c)
#     anth_stop    /v1/messages with stop_sequences:["beta"] on an echo
#                  prompt -> 200 with "stop_reason":"stop_sequence" AND
#                  "stop_sequence":"beta" (the matched text; collapsing to
#                  end_turn told Anthropic clients the model chose to stop).
#
#   authoritative usage (Inc 2d)
#     usage_t1 /    turn-2 continuation built from the SERVED t1 answer;
#     usage_warm    the thinking t1's committed reasoning forces the engine
#                   to align the text-record proposal DOWN, and usage must
#                   report that verdict: cached_tokens > 0 AND equal to
#                   timings prefill_cached (the admit-time proposal is no
#                   longer what usage reports).
#
#   client cancel (Inc 2c)
#     cancel       a live stream whose client disconnects mid-decode (curl
#                  -m 4 on a 3000-token stream) must count
#                  requests_total{outcome="canceled"} -- never completed
#                  (the pre-2c behavior counted every mid-flight cancel as
#                  a completed request).
#
#   failure injection (Inc 2c)
#     strand       reboot with DS4_GATE_CONT_FAIL_AFTER_STEPS=8 (engine
#                  gate-teeth: strands every live bank mid-decode exactly as
#                  a real engine failure).  A live chat stream must get real
#                  deltas, then the NATIVE stream-error event carrying the
#                  forced-failure message -- and NO fabricated
#                  finish="length" / [DONE] (the pre-2c dishonesty, plan
#                  §2.1 / matrix defect 2).  Counters: the request counts
#                  FAILED, never completed; cont_batch_failures increments.
#
#   budget matrix (omitted is every other gate's default; negative above)
#     anth_zero    anthropic max_tokens:0 (prewarm contract) -> 200 +
#                  "output_tokens":0 (serial: NEED_PREFILL_ONLY)
#     resp_zero    responses max_output_tokens:0 -> 200, incomplete +
#                  output_tokens 0 + output [] (rides CONT since 3b; Inc 3c
#                  trims the engine's seed-floor token at every wire site)
#     oa_zero      chat max_tokens:0 -> 200, completion_tokens == 0, empty
#                  content, finish "length" (Inc 3c: pre-fix the cont lane
#                  leaked the seed token -- measured completion_tokens=1,
#                  content "Okay"; the old <= 1 tolerance was covering it)
#     oa_one       chat max_tokens:1 -> 200, completion_tokens == 1, finish
#                  "length"
#     oa_clamped   chat max_tokens:10000000 -> 200, finish "stop" (a huge
#                  budget must serve; EOS ends it long before the bound)
#
#   admission bounds (Inc 2e; two more boots with tiny bounds)
#     bytes_shed   serial-only boot, MAX_QUEUE_BYTES=5000: an ~8 KB body is
#                  shed PRE-PARSE -> 429 + Retry-After + the native
#                  rate_limit_error envelope + shed{queue_bytes}
#     age_shed     MAX_QUEUE_AGE_S=6: a request queued behind a long serial
#                  generation ages out -> 503 + Retry-After + native
#                  overloaded_error + shed{queue_age}; the head still 200s
#     depth_shed   MAX_QUEUE=3: a 6-way blast -> at least one 429 with
#                  Retry-After + shed{queue_depth}; the head still 200s
#     clients_shed cont boot, MAX_CLIENTS=2: two parked admission streams +
#                  a probe -> 503 + Retry-After + overloaded_error +
#                  shed{clients}
#     slow_reader  OUT_AGG_CAP=2048/EVICT_MIN=1024/CLIENT_SNDBUF=8192: an
#                  on-box client that stops reading its stream is EVICTED
#                  once the aggregate backlog passes the cap ->
#                  shed{slow_reader} + canceled (Inc 2c bucket), server
#                  healthy after.  The sndbuf pin is what makes the heap
#                  backlog reachable (loopback autotune = ~2.6 MiB/conn)
#
# Runs FROM the Mac over SSH like the other gates.  NOTE: the boot kills any
# running ds4-server on $R.  End state: ds4-server killed, box left free.
#
# Env overrides: R (${DS4_BOX33:?}) BINDIR (${DS4_BOX_TREE:?})
#                PORT (8000) TUNNEL_PORT (18000) CTX (16384)
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
set -uo pipefail

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/error_envelope_gate
OUT=${OUT:-/tmp/error_envelope_gate_$$}
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

# post <leg> <path> <json>  -> $OUT/<leg>.json, echoes HTTP code
post(){
  curl -s -m 180 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
# chpost <leg> <path> <json>  -> body sent with Transfer-Encoding: chunked
# (curl really chunks a stdin --data-binary body when the header is set)
chpost(){
  printf '%s' "$3" | curl -s -m 180 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -H 'Transfer-Encoding: chunked' --data-binary @-
}
# hpost <leg> <path> <json>  -> also captures response HEADERS in $OUT/<leg>.hdr
hpost(){
  curl -s -m 180 -o "$OUT/$1.json" -D "$OUT/$1.hdr" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
hdr_has(){ grep -qi "$2" "$OUT/$1.hdr" || fail "$1: missing header [$2] in $(tr -d '\r' < "$OUT/$1.hdr" | tr '\n' ' ')"; }
has(){ grep -q "$2" "$OUT/$1.json" || fail "$1: missing [$2] in $(head -c 300 "$OUT/$1.json")"; }
lacks(){ grep -q "$2" "$OUT/$1.json" && fail "$1: unexpected [$2] in $(head -c 300 "$OUT/$1.json")"; true; }
code_is(){ [ "$2" = "$3" ] || fail "$1: HTTP $2, want $3 ($(head -c 300 "$OUT/$1.json"))"; }
metric(){ curl -s -m 10 "$BASE/metrics" | grep -oE "^$1 [0-9]+" | awk '{print $2}'; }
lmetric(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$'; }

boot

# ---- envelope legs ---------------------------------------------------------
c=$(post anth_parse /v1/messages '{not json')
code_is anth_parse "$c" 400
has  anth_parse '"type":"error","error":{"type":"invalid_request_error"'
lacks anth_parse '"error":{"message":'
log "anth_parse PASS (native envelope on parse failure)"

c=$(post anth_neg /v1/messages '{"model":"m","max_tokens":-1,"messages":[{"role":"user","content":"hi"}]}')
code_is anth_neg "$c" 400
has anth_neg '"type":"error","error":{"type":"invalid_request_error"'
has anth_neg 'max_tokens must be >= 0'
log "anth_neg PASS"

c=$(post oa_neg /v1/chat/completions '{"max_tokens":-1,"messages":[{"role":"user","content":"hi"}]}')
code_is oa_neg "$c" 400
has oa_neg '"error":{"message":"max_tokens must be >= 0"'
log "oa_neg PASS"

c=$(post oa_neg2 /v1/chat/completions '{"max_completion_tokens":-2,"messages":[{"role":"user","content":"hi"}]}')
code_is oa_neg2 "$c" 400
has oa_neg2 'max_completion_tokens must be >= 0'
log "oa_neg2 PASS (field name echoed)"

c=$(post comp_neg /v1/completions '{"max_tokens":-1,"prompt":"hi"}')
code_is comp_neg "$c" 400
has comp_neg 'max_tokens must be >= 0'
log "comp_neg PASS"

c=$(post resp_neg /v1/responses '{"max_output_tokens":-3,"input":"hi"}')
code_is resp_neg "$c" 400
has resp_neg 'max_output_tokens must be >= 0'
log "resp_neg PASS (field name echoed)"

c=$(curl -s -m 20 -o "$OUT/not_found.json" -w '%{http_code}' "$BASE/nope")
code_is not_found "$c" 404
has not_found '"error":{"message":'
log "not_found PASS (historical-owner envelope)"

# ---- response_format typed refusal (v0.6.3 Inc 1; ds4-on-spark#10) ---------
c=$(post rf_text /v1/chat/completions '{"max_tokens":0,"temperature":0,"response_format":{"type":"text"},"messages":[{"role":"user","content":"hi"}]}')
code_is rf_text "$c" 200
log "rf_text PASS (type text accepted unchanged)"

c=$(post rf_obj /v1/chat/completions '{"response_format":{"type":"json_object"},"messages":[{"role":"user","content":"hi"}]}')
code_is rf_obj "$c" 400
has rf_obj "response_format type 'json_object' is not implemented"
has rf_obj '"error":{"message":'
log "rf_obj PASS (typed refusal, OpenAI envelope)"

c=$(post rf_schema /v1/chat/completions '{"response_format":{"json_schema":{"name":"x","schema":{"type":"object"}},"type":"json_schema"},"messages":[{"role":"user","content":"hi"}]}')
code_is rf_schema "$c" 400
has rf_schema "json_schema' is not implemented"
log "rf_schema PASS (nested schema before type still refused)"

c=$(post rf_comp /v1/completions '{"response_format":{"type":"json_object"},"prompt":"hi"}')
code_is rf_comp "$c" 400
has rf_comp "not implemented"
log "rf_comp PASS"

c=$(post rf_resp /v1/responses '{"text":{"format":{"type":"json_schema","name":"x"}},"input":"hi"}')
code_is rf_resp "$c" 400
has rf_resp "text.format type 'json_schema' is not implemented"
log "rf_resp PASS (nested text.format named)"

c=$(post rf_anth /v1/messages '{"model":"m","max_tokens":8,"output_format":{"type":"json_schema","schema":{}},"messages":[{"role":"user","content":"hi"}]}')
code_is rf_anth "$c" 400
has rf_anth '"type":"error","error":{"type":"invalid_request_error"'
has rf_anth "output_format type 'json_schema' is not implemented"
log "rf_anth PASS (native envelope)"

c=$(post rf_after /v1/chat/completions '{"max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"still serving?"}]}')
code_is rf_after "$c" 200
log "rf_after PASS (server healthy after refusals)"

# ---- chunked request bodies (v0.6.3 Inc 2) ---------------------------------
c=$(chpost ch_chat /v1/chat/completions '{"max_tokens":0,"temperature":0,"messages":[{"role":"user","content":"hi"}]}')
code_is ch_chat "$c" 200
has ch_chat '"completion_tokens":0'
log "ch_chat PASS (chunked body decoded, oa_zero contract held)"

c=$(chpost ch_anth /v1/messages '{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"prewarm via chunked"}]}')
code_is ch_anth "$c" 200
has ch_anth '"output_tokens":0'
log "ch_anth PASS"

c=$(chpost ch_comp /v1/completions '{"max_tokens":1,"temperature":0,"prompt":"hi"}')
code_is ch_comp "$c" 200
has ch_comp '"text_completion"'
log "ch_comp PASS"

c=$(chpost ch_resp /v1/responses '{"max_output_tokens":0,"input":"prewarm chunked"}')
code_is ch_resp "$c" 200
has ch_resp '"output_tokens":0'
log "ch_resp PASS"

c=$(chpost ch_rf /v1/chat/completions '{"response_format":{"type":"json_object"},"messages":[{"role":"user","content":"hi"}]}')
code_is ch_rf "$c" 400
has ch_rf "response_format type 'json_object' is not implemented"
log "ch_rf PASS (chunked body reaches the parser; typed refusal fires)"

# ---- think-dial observability (v0.6.3 Inc 3) -------------------------------
c=$(post th_high /v1/chat/completions '{"max_tokens":8,"temperature":0,"reasoning_effort":"high","messages":[{"role":"user","content":"hi"}]}')
code_is th_high "$c" 200
ssh "$R" "grep -q 'think=high' $SRV" || fail "th_high: no think=high on a serving line"
ssh "$R" "grep -q 'think=low' $SRV" || fail "th_high: think=low absent (dial should ride every progress line)"
tm=$(lmetric 'ds4_requests_think_total{mode="high"}')
[ "${tm:-0}" -ge 1 ] || fail "th_high: ds4_requests_think_total{mode=high} = ${tm:-unset}"
curl -s -m 10 -H 'Accept: application/json' "$BASE/v1/stats" > "$OUT/th_stats.json"
grep -q '"think_modes":{"none":' "$OUT/th_stats.json" || fail "th_stats: think_modes family missing from /v1/stats"
python3 -c "
import json,sys
d = json.load(open('$OUT/th_stats.json'))
tm = d['think_modes']
sys.exit(0 if tm['high'] >= 1 and tm['low'] >= 1 else 1)
" || fail "th_stats: high/low counts not positive in /v1/stats think_modes"
log "th_high PASS (think= on the serving line; counter family in /metrics + /v1/stats)"

# ---- stop honesty (Inc 2c): a matched stop is stop_sequence, not end_turn --
c=$(post anth_stop /v1/messages '{"model":"m","max_tokens":128,"stop_sequences":["beta"],"messages":[{"role":"user","content":"Repeat exactly: alpha beta gamma"}]}')
code_is anth_stop "$c" 200
has anth_stop '"stop_reason":"stop_sequence"'
has anth_stop '"stop_sequence":"beta"'
log "anth_stop PASS (matched stop surfaced natively)"

# ---- budget matrix ----------------------------------------------------------
c=$(post anth_zero /v1/messages '{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"prewarm this prompt"}]}')
code_is anth_zero "$c" 200
has anth_zero '"output_tokens":0'
log "anth_zero PASS (prewarm contract, zero decode)"

c=$(post resp_zero /v1/responses '{"max_output_tokens":0,"input":"prewarm this too"}')
code_is resp_zero "$c" 200
has resp_zero '"status":"incomplete"'
has resp_zero '"output_tokens":0'
has resp_zero '"output":\[\]'
log "resp_zero PASS (incomplete, zero output, empty output array on the cont lane)"

c=$(post oa_zero /v1/chat/completions '{"max_tokens":0,"temperature":0,"messages":[{"role":"user","content":"hi"}]}')
code_is oa_zero "$c" 200
has oa_zero '"completion_tokens":0'
has oa_zero '"content":""'
has oa_zero '"finish_reason":"length"'
log "oa_zero PASS (completion_tokens=0, empty content, finish length -- seed trimmed)"

c=$(post oa_one /v1/chat/completions '{"max_tokens":1,"temperature":0,"messages":[{"role":"user","content":"hi"}]}')
code_is oa_one "$c" 200
has oa_one '"completion_tokens":1'
has oa_one '"finish_reason":"length"'
log "oa_one PASS"

c=$(post oa_clamped /v1/chat/completions '{"max_tokens":10000000,"temperature":0,"messages":[{"role":"user","content":"Reply with exactly: envelope gate ok"}]}')
code_is oa_clamped "$c" 200
has oa_clamped '"finish_reason":"stop"'
log "oa_clamped PASS (huge budget served, EOS ended it)"

# ---- authoritative usage (Inc 2d): usage == the engine's cache verdict -----
# Turn-2 continuation built from the SERVED t1 answer (the a2a warm-gate
# recipe; a turn-1 repeat does NOT warm-match -- the record outruns the
# request and fork-always placement is off by design).  For a thinking t1
# the committed history holds reasoning the t2 re-render drops, so the
# engine ALIGNS the text-record proposal DOWN to the true shared token
# prefix -- exactly the proposal-vs-verdict window Inc 2d closes: usage
# must report the verdict (== timings prefill_cached) and still show
# engagement (> 0).
c=$(post usage_t1 /v1/chat/completions '{"max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one planet."}]}')
code_is usage_t1 "$c" 200
python3 - "$OUT/usage_t1.json" > "$OUT/usage_t2_req.json" <<'PY' || fail "usage_warm: t2 build failed"
import json, sys
t1 = json.load(open(sys.argv[1]))
ans = t1["choices"][0]["message"]["content"]
msgs = [{"role": "user", "content": "Name one planet."},
        {"role": "assistant", "content": ans},
        {"role": "user", "content": "Name another."}]
print(json.dumps({"max_tokens": 48, "temperature": 0, "messages": msgs}))
PY
c=$(curl -s -m 180 -o "$OUT/usage_warm.json" -w '%{http_code}' "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' -d @"$OUT/usage_t2_req.json")
code_is usage_warm "$c" 200
# Inc 3a frame law: usage lives in the CLIENT's token frame; the engine
# verdict can count a LARGER effective prompt on thinking-warm admits.  The
# structural invariant is: the client's uncached prompt portion equals what
# the engine computed fresh -- prompt_tokens - cached == timings
# prefill_tokens (and cache engagement stays visible: cached > 0).
uc=$(grep -oE '"cached_tokens":[0-9]+' "$OUT/usage_warm.json" | head -1 | cut -d: -f2)
pt=$(grep -oE '"prompt_tokens":[0-9]+' "$OUT/usage_warm.json" | head -1 | cut -d: -f2)
pf=$(grep -oE '"prefill_tokens":[0-9]+' "$OUT/usage_warm.json" | head -1 | cut -d: -f2)
[ -n "$uc" ] && [ -n "$pt" ] && [ -n "$pf" ] \
  || fail "usage_warm: usage/timings fields missing (cached=${uc:-?} prompt=${pt:-?} prefill=${pf:-?})"
[ $((pt - uc)) -eq "$pf" ] \
  || fail "usage_warm: uncached prompt $((pt - uc)) != engine computed $pf (cached=$uc prompt=$pt)"
[ "${uc:-0}" -gt 0 ] || fail "usage_warm: t2 continuation reported zero cache (no warm engagement)"
log "usage_warm PASS (client frame: prompt=$pt cached=$uc uncached==computed=$pf)"

# ---- client cancel (Inc 2c): a mid-stream disconnect counts CANCELED -------
can0=$(lmetric 'ds4_requests_total{outcome="canceled"}')
ccomp0=$(lmetric 'ds4_requests_total{outcome="completed"}')
curl -s -m 4 -o "$OUT/cancel.sse" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"stream":true,"temperature":0,"max_tokens":3000,"messages":[{"role":"user","content":"Write a very long story about the sea."}]}' || true
n=0
while :; do
  can1=$(lmetric 'ds4_requests_total{outcome="canceled"}')
  [ "${can1:-0}" -gt "${can0:-0}" ] && break
  n=$((n+1)); [ $n -ge 24 ] && fail "cancel: canceled counter never incremented (${can0:-?} -> ${can1:-?})"
  sleep 5
done
ccomp1=$(lmetric 'ds4_requests_total{outcome="completed"}')
[ "${ccomp1:-0}" -eq "${ccomp0:-0}" ] || fail "cancel: canceled stream counted COMPLETED ($ccomp0 -> $ccomp1)"
log "cancel PASS (mid-stream disconnect counted canceled, not completed)"

# ---- admission-callback cancellation (§5 Inc 2 gate item) -------------------
# The client dies DURING its admission prefill (a ~10k-token prompt takes
# tens of seconds cold; curl -m 2 leaves long before the first token).
# cont_alive's MSG_PEEK probe reaps the row between prefill chunks; the
# request must count CANCELED, and the server must keep serving.
python3 - > "$OUT/admit_cancel_req.json" <<'PY'
import json
body = "The quick brown fox jumps over the lazy dog. " * 900
print(json.dumps({"max_tokens": 32, "temperature": 0,
                  "messages": [{"role": "user", "content": body}]}))
PY
acan0=$(lmetric 'ds4_requests_total{outcome="canceled"}')
curl -s -m 2 -o /dev/null "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' -d @"$OUT/admit_cancel_req.json" || true
n=0
while :; do
  acan1=$(lmetric 'ds4_requests_total{outcome="canceled"}')
  [ "${acan1:-0}" -gt "${acan0:-0}" ] && break
  n=$((n+1)); [ $n -ge 24 ] && fail "admit_cancel: canceled never incremented (${acan0:-?} -> ${acan1:-?})"
  sleep 5
done
c=$(post admit_cancel_after /v1/chat/completions '{"max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"still alive?"}]}')
code_is admit_cancel_after "$c" 200
log "admit_cancel PASS (mid-admission disconnect counted canceled; server healthy after)"

# ---- failure injection: engine-stranded live stream (Inc 2c) ---------------
BOOT_ENV="DS4_GATE_CONT_FAIL_AFTER_STEPS=8" boot
comp0=$(lmetric 'ds4_requests_total{outcome="completed"}')
fail0=$(lmetric 'ds4_requests_total{outcome="failed"}')
c=$(curl -s -m 90 -o "$OUT/strand.sse" -w '%{http_code}' "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"stream":true,"temperature":0,"max_tokens":400,"messages":[{"role":"user","content":"Count slowly from one to fifty in words."}]}')
code_is strand "$c" 200   # headers + deltas went out before the failure
grep -qE '"delta":\{"(content|reasoning_content)":' "$OUT/strand.sse" \
  || fail "strand: no deltas before the failure"
grep -q 'event: error' "$OUT/strand.sse" || fail "strand: native stream-error event missing"
grep -q '"type":"server_error"' "$OUT/strand.sse" || fail "strand: error payload missing"
grep -q 'gate-forced continuous failure' "$OUT/strand.sse" \
  || fail "strand: forced-failure cause not surfaced"
grep -q '"finish_reason":"length"' "$OUT/strand.sse" \
  && fail "strand: fabricated length finish still present"
grep -q 'data: \[DONE\]' "$OUT/strand.sse" \
  && fail "strand: fabricated DONE after failure"
comp1=$(lmetric 'ds4_requests_total{outcome="completed"}')
fail1=$(lmetric 'ds4_requests_total{outcome="failed"}')
cbf=$(metric ds4_cont_batch_failures_total)
[ "${fail1:-0}" -gt "${fail0:-0}" ] || fail "strand: requests failed counter did not increment ($fail0 -> ${fail1:-?})"
[ "${comp1:-0}" -eq "${comp0:-0}" ] || fail "strand: stranded stream counted COMPLETED ($comp0 -> $comp1)"
[ -n "$cbf" ] && [ "$cbf" -ge 1 ] || fail "strand: cont_batch_failures=${cbf:-absent}"
log "strand PASS (deltas -> native error event, counted failed not completed, cont_batch_failures=$cbf)"

# ---- failure BEFORE headers (§5 Inc 2 gate item) ----------------------------
# Same forced-failure boot, BUFFERED request: the stranded row has nothing
# on the wire, so the sweep re-serves it on the single path -- the client
# gets a full 200 and the request counts COMPLETED (failed unchanged).
sfail0=$(lmetric 'ds4_requests_total{outcome="failed"}')
scomp0=$(lmetric 'ds4_requests_total{outcome="completed"}')
c=$(post strand_buffered /v1/chat/completions '{"temperature":0,"max_tokens":64,"messages":[{"role":"user","content":"Reply with exactly: strand buffered ok"}]}')
code_is strand_buffered "$c" 200
has strand_buffered '"finish_reason"'
sfail1=$(lmetric 'ds4_requests_total{outcome="failed"}')
scomp1=$(lmetric 'ds4_requests_total{outcome="completed"}')
[ "${scomp1:-0}" -gt "${scomp0:-0}" ] || fail "strand_buffered: not counted completed ($scomp0 -> $scomp1)"
[ "${sfail1:-0}" -eq "${sfail0:-0}" ] || fail "strand_buffered: counted FAILED ($sfail0 -> $sfail1) despite full serial re-serve"
log "strand_buffered PASS (before-headers failure re-served serially, counted completed)"

# ============================================================================
# Inc 2e: admission bounds + shed (429/503 + Retry-After, endpoint-native)
# ============================================================================

# ---- boot C: serial-only, tiny queue bounds --------------------------------
# CONTINUOUS=0 COALESCE=0 makes dispatch strictly serial FIFO, so the queue
# actually FILLS (the cont lane would admit everything instantly).
BOOT_ENV="DS4_SERVER_CONTINUOUS=0 DS4_SERVER_COALESCE=0 DS4_SERVER_MAX_QUEUE=3 DS4_SERVER_MAX_QUEUE_BYTES=5000 DS4_SERVER_MAX_QUEUE_AGE_S=6" boot
ssh "$R" "grep -q 'admission bounds: clients<=' $SRV" \
  || fail "bounds boot: admission-bounds boot line missing from srv.log"

# bytes_shed: an oversized body sheds PRE-PARSE (0 in flight + 8000 > 5000).
qb0=$(lmetric 'ds4_requests_shed_total{reason="queue_bytes"}')
python3 - > "$OUT/bytes_shed_req.json" <<'PY'
import json
print(json.dumps({"model": "m", "max_tokens": 16,
                  "messages": [{"role": "user", "content": "pad " * 2000}]}))
PY
c=$(curl -s -m 30 -o "$OUT/bytes_shed.json" -D "$OUT/bytes_shed.hdr" -w '%{http_code}' \
     "$BASE/v1/messages" -H 'Content-Type: application/json' -d @"$OUT/bytes_shed_req.json")
code_is bytes_shed "$c" 429
has bytes_shed '"type":"error","error":{"type":"rate_limit_error"'
lacks bytes_shed '"error":{"message":'
hdr_has bytes_shed 'Retry-After:'
qb1=$(lmetric 'ds4_requests_shed_total{reason="queue_bytes"}')
[ "${qb1:-0}" -gt "${qb0:-0}" ] || fail "bytes_shed: shed{queue_bytes} never incremented (${qb0:-?} -> ${qb1:-?})"
log "bytes_shed PASS (pre-parse 429 + Retry-After, native envelope, counter ${qb0:-0} -> $qb1)"

# age_shed: t2 queues behind a long serial generation, outwaits the 6s bound,
# and is shed 503 at dequeue -- while the head still completes normally.
qa0=$(lmetric 'ds4_requests_shed_total{reason="queue_age"}')
curl -s -m 180 -o "$OUT/age_head.json" -w '%{http_code}' "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"max_tokens":300,"temperature":0,"messages":[{"role":"user","content":"Write a paragraph about tides."}]}' \
     > "$OUT/age_head.code" &
AGE_HEAD_PID=$!
sleep 2
c=$(hpost age_shed /v1/messages '{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"quick question"}]}')
code_is age_shed "$c" 503
has age_shed '"type":"error","error":{"type":"overloaded_error"'
has age_shed 'over the 6s limit'
hdr_has age_shed 'Retry-After:'
qa1=$(lmetric 'ds4_requests_shed_total{reason="queue_age"}')
[ "${qa1:-0}" -gt "${qa0:-0}" ] || fail "age_shed: shed{queue_age} never incremented (${qa0:-?} -> ${qa1:-?})"
wait "$AGE_HEAD_PID" || true
[ "$(cat "$OUT/age_head.code")" = 200 ] || fail "age_shed: head request got $(cat "$OUT/age_head.code"), want 200"
log "age_shed PASS (queued 503 + Retry-After after the head's 200; counter ${qa0:-0} -> $qa1)"

# depth_shed: 6-way blast against MAX_QUEUE=3 -- late arrivals 429, the head
# still serves.  (Queued survivors may ALSO age out 503 under the 6s bound;
# only the depth counter and the 429/200 presence are asserted.)
qd0=$(lmetric 'ds4_requests_shed_total{reason="queue_depth"}')
BLAST_PIDS=""
for i in 1 2 3 4 5 6; do
  curl -s -m 180 -o "$OUT/depth_$i.json" -D "$OUT/depth_$i.hdr" -w '%{http_code}' \
       "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
       -d '{"max_tokens":150,"temperature":0,"messages":[{"role":"user","content":"Briefly define entropy."}]}' \
       > "$OUT/depth_$i.code" &
  BLAST_PIDS="$BLAST_PIDS $!"
  sleep 0.3
done
for p in $BLAST_PIDS; do wait "$p" || true; done
codes=$(cat "$OUT"/depth_?.code | tr '\n' ' ')
echo "$codes" | grep -q 429 || fail "depth_shed: no 429 in blast codes [$codes]"
echo "$codes" | grep -q 200 || fail "depth_shed: no 200 in blast codes [$codes]"
for i in 1 2 3 4 5 6; do
  if [ "$(cat "$OUT/depth_$i.code")" = 429 ]; then
    grep -qi 'Retry-After:' "$OUT/depth_$i.hdr" || fail "depth_shed: 429 #$i missing Retry-After"
    grep -q 'rate_limit_error' "$OUT/depth_$i.json" || fail "depth_shed: 429 #$i not the native envelope"
  fi
done
qd1=$(lmetric 'ds4_requests_shed_total{reason="queue_depth"}')
[ "${qd1:-0}" -gt "${qd0:-0}" ] || fail "depth_shed: shed{queue_depth} never incremented (${qd0:-?} -> ${qd1:-?})"
log "depth_shed PASS (blast codes [$codes]; counter ${qd0:-0} -> $qd1)"

# ---- boot D: cont, tiny client + slow-reader bounds ------------------------
# CLIENT_SNDBUF is pinned small: on GB10 loopback the kernel send buffer
# autotunes to ~2.6 MiB per connection (measured: ss skmem tb=2626560), so an
# unpinned socket swallows a gate-sized stalled stream whole and the drain
# never blocks -- the heap-side backlog machinery would be unreachable.
BOOT_ENV="DS4_SERVER_MAX_CLIENTS=2 DS4_SERVER_OUT_AGG_CAP=2048 DS4_SERVER_OUT_AGG_EVICT_MIN=1024 DS4_SERVER_CLIENT_SNDBUF=8192" boot

# clients_shed: two parked admission-heavy streams (long prompts prefill for
# tens of seconds and emit NOTHING, so they hold connections without touching
# the tiny backlog cap) + a probe = 3 > 2.
cs0=$(lmetric 'ds4_requests_shed_total{reason="clients"}')
python3 - > "$OUT/park_req.json" <<'PY'
import json
body = "The quick brown fox jumps over the lazy dog. " * 900
print(json.dumps({"stream": True, "max_tokens": 32, "temperature": 0,
                  "messages": [{"role": "user", "content": body}]}))
PY
curl -s -m 150 -o /dev/null "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' -d @"$OUT/park_req.json" &
PARK1=$!
curl -s -m 150 -o /dev/null "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' -d @"$OUT/park_req.json" &
PARK2=$!
sleep 3
c=$(hpost clients_shed /v1/messages '{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}')
code_is clients_shed "$c" 503
has clients_shed '"type":"error","error":{"type":"overloaded_error"'
hdr_has clients_shed 'Retry-After:'
cs1=$(lmetric 'ds4_requests_shed_total{reason="clients"}')
[ "${cs1:-0}" -gt "${cs0:-0}" ] || fail "clients_shed: shed{clients} never incremented (${cs0:-?} -> ${cs1:-?})"
kill "$PARK1" "$PARK2" 2>/dev/null || true
n=0
while :; do
  cc=$(metric ds4_clients_connected)
  [ -n "$cc" ] && [ "$cc" -le 1 ] && break
  n=$((n+1)); [ $n -ge 24 ] && fail "clients_shed: clients gauge stuck at ${cc:-?} after killing parked streams"
  sleep 5
done
log "clients_shed PASS (503 + Retry-After at 3 connections; counter ${cs0:-0} -> $cs1)"

# slow_reader: an ON-BOX client (no tunnel buffering) opens a stream, reads
# the headers, then stops reading.  Kernel buffers fill, client_main's drain
# stalls, the worker's job_emit backlog passes the 2048 aggregate cap, and
# the emitter is evicted: shed{slow_reader} + the Inc 2c canceled bucket.
sr0=$(lmetric 'ds4_requests_shed_total{reason="slow_reader"}')
scan0=$(lmetric 'ds4_requests_total{outcome="canceled"}')
ssh "$R" "cat > /tmp/stall_reader.py" <<PY
import json, socket, time
body = json.dumps({"stream": True, "max_tokens": 3000, "temperature": 0,
                   "messages": [{"role": "user", "content": "Tell a very long story about rivers."}]})
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
s.connect(("127.0.0.1", $PORT))
req = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body)
s.sendall(req.encode())
s.recv(256)          # response started
time.sleep(240)      # never read again; the server must evict us
PY
STALL_PID=$(ssh "$R" "cd /tmp && setsid nohup python3 /tmp/stall_reader.py > /tmp/stall_reader.log 2>&1 & echo \$!")
n=0
while :; do
  sr1=$(lmetric 'ds4_requests_shed_total{reason="slow_reader"}')
  [ "${sr1:-0}" -gt "${sr0:-0}" ] && break
  n=$((n+1)); [ $n -ge 30 ] && fail "slow_reader: shed{slow_reader} never incremented (${sr0:-?} -> ${sr1:-?})"
  sleep 5
done
n=0
while :; do
  scan1=$(lmetric 'ds4_requests_total{outcome="canceled"}')
  [ "${scan1:-0}" -gt "${scan0:-0}" ] && break
  n=$((n+1)); [ $n -ge 24 ] && fail "slow_reader: eviction never settled canceled (${scan0:-?} -> ${scan1:-?})"
  sleep 5
done
ssh "$R" "kill $STALL_PID 2>/dev/null; exit 0"
c=$(post slow_reader_after /v1/chat/completions '{"max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"still healthy?"}]}')
code_is slow_reader_after "$c" 200
log "slow_reader PASS (aggregate backlog eviction counted shed{slow_reader}=$sr1 + canceled; server healthy after)"

# ---- boot E: whole-prompt depth fence (v0.6.3 Inc 5) ------------------------
# Serial-only + DS4_METAL_PREFILL_CHUNK=0: every serial prefill is one
# whole-prompt forward.  Past 8192 rows that territory is unqualified, so
# the request must get a TYPED 503 naming the lever (not a crash, not a
# generic 500), the boot log must disclose the mode, and the escape env
# must genuinely lift the fence.
BOOT_ENV="DS4_SERVER_CONTINUOUS=0 DS4_SERVER_COALESCE=0 DS4_METAL_PREFILL_CHUNK=0" boot
ssh "$R" "grep -q 'serial prefill widened' $SRV" \
  || fail "fence boot: whole-prompt boot disclosure line missing from srv.log"

# ~8.5k words (> 8192 tokens, safely under the 16384 ctx parse gate).
python3 - > "$OUT/fence_req.json" <<'PY'
import json
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 850
print(json.dumps({"model": "m", "max_tokens": 8, "temperature": 0,
                  "messages": [{"role": "user", "content": words}]}))
PY
fr0=$(lmetric 'ds4_requests_rejected_total{lane="serial",reason="deep_policy"}')
c=$(curl -s -m 60 -o "$OUT/fence_refuse.json" -w '%{http_code}' "$BASE/v1/messages" \
     -H 'Content-Type: application/json' -d @"$OUT/fence_req.json")
code_is fence_refuse "$c" 503
has fence_refuse 'Whole-prompt prefill is fenced'
has fence_refuse 'DS4_PREFILL_NOFENCE'
ssh "$R" "grep -q 'whole-prompt depth fence: refusing serial request' $SRV" \
  || fail "fence_refuse: typed refusal line missing from srv.log"
fr1=$(lmetric 'ds4_requests_rejected_total{lane="serial",reason="deep_policy"}')
[ "${fr1:-0}" -gt "${fr0:-0}" ] || fail "fence_refuse: rejected{serial,deep_policy} never incremented (${fr0:-?} -> ${fr1:-?})"
# a short prompt still serves whole-prompt (width under the fence)
c=$(post fence_small /v1/chat/completions '{"max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"hi"}]}')
code_is fence_small "$c" 200
log "fence_refuse PASS (typed 503 names the lever; counter ${fr0:-0} -> $fr1; short prompt still 200)"

# escape hatch: the same deep request must get PAST the fence with
# DS4_PREFILL_NOFENCE=1.  What happens in unqualified territory is NOT
# asserted -- measured 08-20: this exact shape (a 12,755-row one-shot)
# fails TODAY as a clean typed 500 'cuda prefill failed', which is the
# fence's justification, not a defect.  The lever's contract: the fence
# genuinely lifts (no typed 503, no refusal line), whatever happens
# comes back as an ENVELOPE (200 or a typed 500 -- never a crash or a
# dropped connection), and the server serves the next request.
BOOT_ENV="DS4_SERVER_CONTINUOUS=0 DS4_SERVER_COALESCE=0 DS4_METAL_PREFILL_CHUNK=0 DS4_PREFILL_NOFENCE=1" boot
ssh "$R" "grep -q 'depth fence LIFTED' $SRV" \
  || fail "fence_lift boot: LIFTED disclosure line missing from srv.log"
c=$(curl -s -m 180 -o "$OUT/fence_lift.json" -w '%{http_code}' "$BASE/v1/messages" \
     -H 'Content-Type: application/json' -d @"$OUT/fence_req.json")
[ "$c" = 200 ] || [ "$c" = 500 ] \
  || fail "fence_lift: HTTP $c, want 200 or a typed 500 ($(head -c 300 "$OUT/fence_lift.json"))"
lacks fence_lift 'Whole-prompt prefill is fenced'
if [ "$c" = 200 ]; then has fence_lift '"stop_reason"'
else has fence_lift '"type":"error","error":{"type":"api_error"'; fi
ssh "$R" "grep -q 'whole-prompt depth fence: refusing' $SRV" \
  && fail "fence_lift: fence refused despite DS4_PREFILL_NOFENCE=1"
c=$(post fence_lift_after /v1/chat/completions '{"max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"still healthy?"}]}')
code_is fence_lift_after "$c" 200
log "fence_lift PASS (fence lifted: deep one-shot reached the engine, outcome stayed an envelope, server healthy after)"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT (server killed, $R left free)"
