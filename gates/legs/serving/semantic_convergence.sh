#!/bin/bash
# serving/semantic_convergence.sh — v0.5.6 API Inc 7 live gate: semantic convergence + hot-path
# observation (plan §5 Inc 7 line: same token tape through old/new semantic
# paths, MTP/DSpark on/off, no output-buffer or memory growth regression;
# the live ABBA + N=1/8/16 legs run separately via perf/batched_lane_abba.sh).
#
# What Inc 7 changed:
#   7a  ONE route-neutral semantic accumulator (sem_accum) feeds both the
#       serial loop and cont_on_token -- text, thinking, DSML tracker, stop
#       scan + hold-back, think-gated marker scan, matched stop, verdict.
#       The byte-level old/new equality is pinned at the UNIT layer (the
#       tape/oracle suites drive the accumulator through the real cont
#       entry points; cont temp-0 is not run-to-run deterministic live, so
#       units are the token-tape oracle).  These legs prove every semantic
#       shape still SERVES correctly live through the shared machine.
#   7b  Responses stream machine keeps item-text SPANS of the raw buffer
#       (no duplicate cumulative copies); reasoning_tokens is the
#       GENERATION-side count (accumulator / detok walk), the finalize
#       retokenize is gone.  Live oracle: delta-vs-done text equality +
#       nonzero reasoning_tokens on stream AND buffered rows.
#   7c  observation only -- ds4_cont_ontoken_* (host projection cost under
#       gen_mu) and ds4_batch_genmu_wait_* (/v1/batch fairness).  The legs
#       assert ENGAGEMENT + sanity ceilings and LOG the numbers; the
#       offload / bounded-epoch decisions are adjudicated in the receipt.
#
# Boot 1 legs (default env = spec ON):
#   sem_stream_n4    4 concurrent OpenAI streaming chats through the shared
#                    accumulator: all 200 + [DONE] + finish_reason; then
#                    spec_drafts > 0 (the ON side of the on/off matrix),
#                    cont_ontoken_tokens > 0, ns/token logged + < 1ms
#   resp_reason_stream  Responses stream w/ reasoning.summary: summary_text
#                    .done + output_text.done; DELTA-vs-DONE equality for
#                    BOTH item texts (the span-integrity live oracle);
#                    completed usage reasoning_tokens > 0
#   resp_reason_buffered  buffered Responses (plain row -> engine-buffered
#                    detok walk): reasoning_tokens > 0 in usage
#   rss_log          server RSS logged before/after the streams (receipt
#                    datum for the no-growth acceptance; ABBA is the gate)
#
# Boot 2 legs (--no-spec = MTP+DSpark OFF; the OFF side of the matrix):
#   nospec_stream    streaming chat 200 + [DONE]; spec_drafts stays 0
#   nospec_tools     STREAMING anthropic tool turn (the accumulator's DSML
#                    argmax override without the spec accept loops):
#                    input_json_delta + stop_reason tool_use
#
# Boot 3 legs (default env; /v1/batch fairness observation):
#   batch_idle       /v1/batch on an idle server: 200, waits +1, wait ~0
#                    (logged)
#   batch_vs_cont    /v1/batch posted MID-DECODE of a live cont stream:
#                    completes 200 (no deadlock), waits +1, wait_ms logged
#                    -- the starvation number the cut-list check reads
#
# PER-BOOT FUNDED WINDOW: boot 1 spends 6 cont admissions, boot 2 spends 3,
# boot 3 spends 2 -- all inside the ~8-10 per 16k boot budget.
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box left free.
# Env overrides: R (${DS4_BOX33:?}) BINDIR (${DS4_BOX_TREE:?})
#                PORT (8000) TUNNEL_PORT (18000) CTX (16384)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/inc7_gate
OUT=${OUT:-/tmp/inc7_gate_$$}
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

wait_mem(){
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
  ssh "$R" ": > $SRV; cd $BINDIR; env ${BOOT_ENV:-} setsid nohup ./ds4-server -c $CTX --port $PORT ${BOOT_ARGS:-} \
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
  curl -s -m 240 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
sse(){ # $1=name $2=path $3=body -> writes $OUT/$1.sse, echoes http code
  curl -s -m 240 --no-buffer -o "$OUT/$1.sse" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
has(){ grep -q "$2" "$OUT/$1.json" || fail "$1: missing [$2] in $(head -c 300 "$OUT/$1.json")"; }
shas(){ grep -q "$2" "$OUT/$1.sse" || fail "$1: missing [$2] in the stream"; }
code_is(){ [ "$2" = "$3" ] || fail "$1: HTTP $2, want $3 ($(head -c 300 "$OUT/$1".* 2>/dev/null))"; }
# METRICS-HELPER TRAP: extract the number BEFORE head -1.
m(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$' | head -1; }
srv_rss_mb(){ ssh "$R" "ps -o rss= -p \$(pgrep -x ds4-server | head -1) 2>/dev/null" | awk '{print int($1/1024)}'; }

TOOLS_ANTH='[{"name":"list_files","description":"List the files in a directory","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]'

# ==================== boot 1: spec ON semantics + projection cost ===========
boot

rss_boot=$(srv_rss_mb)
log "rss_log: post-boot RSS ${rss_boot:-?} MB"

# sem_stream_n4: four concurrent streams through the shared accumulator.
drafts0=$(m ds4_spec_drafts_total); tok0=$(m ds4_cont_ontoken_tokens_total)
ns0=$(m ds4_cont_ontoken_ns_total)
pids=()
for i in 1 2 3 4; do
  ( c=$(sse "sem_stream_$i" /v1/chat/completions '{"model":"m","max_tokens":160,"temperature":0,"stream":true,"messages":[{"role":"user","content":"Count from 1 to 20, one number per line, then say done. (row '"$i"')"}]}')
    echo "$c" > "$OUT/sem_stream_$i.code" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
for i in 1 2 3 4; do
  c=$(cat "$OUT/sem_stream_$i.code")
  code_is "sem_stream_$i" "$c" 200
  shas "sem_stream_$i" 'data: \[DONE\]'
  # finish may be stop OR length (default thinking can spend the budget);
  # the assert is a TERMINAL finish, not which one.
  grep -qE '"finish_reason":"(stop|length)"' "$OUT/sem_stream_$i.sse" || \
    fail "sem_stream_$i: no terminal finish_reason"
done
drafts1=$(m ds4_spec_drafts_total); tok1=$(m ds4_cont_ontoken_tokens_total)
ns1=$(m ds4_cont_ontoken_ns_total)
[ "${drafts1:-0}" -gt "${drafts0:-0}" ] || fail "sem_stream_n4: spec_drafts never moved (spec ON leg; ${drafts0:-?} -> ${drafts1:-?})"
dt=$(( ${tok1:-0} - ${tok0:-0} )); dn=$(( ${ns1:-0} - ${ns0:-0} ))
[ "$dt" -gt 0 ] || fail "sem_stream_n4: cont_ontoken_tokens never moved (projection metric dead)"
nspt=$(( dn / dt ))
[ "$nspt" -lt 1000000 ] || fail "sem_stream_n4: projection ${nspt} ns/token >= 1ms (sanity ceiling)"
log "sem_stream_n4 PASS (4 streams; projection ${nspt} ns/token over ${dt} tokens)"

# resp_reason_stream: the 7b span + reasoning-count live oracle.
c=$(sse resp_reason_stream /v1/responses '{"stream":true,"max_output_tokens":600,"temperature":0,"reasoning":{"summary":"auto"},"input":"A rectangle is twice as long as it is wide and its perimeter is 36; what is its area?"}')
code_is resp_reason_stream "$c" 200
shas resp_reason_stream '"type":"response.reasoning_summary_text.done"'
shas resp_reason_stream '"type":"response.output_text.done"'
shas resp_reason_stream '"type":"response.completed"'
grep -qE '"reasoning_tokens":[1-9]' "$OUT/resp_reason_stream.sse" || \
  fail "resp_reason_stream: reasoning_tokens not > 0 in completed usage"
# DELTA-vs-DONE equality for both item texts -- the span-integrity oracle:
# what streamed out must equal what the .done events render from the spans.
python3 - "$OUT/resp_reason_stream.sse" <<'PYEOF' || fail "resp_reason_stream: delta-vs-done text mismatch (span regression)"
import json, sys
deltas = {"reasoning": [], "text": []}
done = {}
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
        continue
    try:
        ev = json.loads(line[6:])
    except Exception:
        continue
    t = ev.get("type", "")
    if t == "response.reasoning_summary_text.delta":
        deltas["reasoning"].append(ev.get("delta", ""))
    elif t == "response.output_text.delta":
        deltas["text"].append(ev.get("delta", ""))
    elif t == "response.reasoning_summary_text.done":
        done["reasoning"] = ev.get("text", "")
    elif t == "response.output_text.done":
        done["text"] = ev.get("text", "")
for k in ("reasoning", "text"):
    if "".join(deltas[k]) != done.get(k, ""):
        print(f"{k}: {len(''.join(deltas[k]))} delta bytes vs {len(done.get(k,''))} done bytes")
        sys.exit(1)
sys.exit(0)
PYEOF
log "resp_reason_stream PASS (delta==done for both items; reasoning_tokens > 0)"

# resp_reason_buffered: plain engine-buffered row -> the detok-walk count.
c=$(post resp_reason_buffered /v1/responses '{"max_output_tokens":400,"temperature":0,"input":"What is 17 * 23? Answer with just the number."}')
code_is resp_reason_buffered "$c" 200
grep -qE '"reasoning_tokens":[1-9]' "$OUT/resp_reason_buffered.json" || \
  fail "resp_reason_buffered: reasoning_tokens not > 0 (detok-walk count dead)"
log "resp_reason_buffered PASS (buffered reasoning_tokens > 0)"

rss_end=$(srv_rss_mb)
log "rss_log: post-legs RSS ${rss_end:-?} MB (boot ${rss_boot:-?} MB)"

# ==================== boot 2: --no-spec (MTP+DSpark OFF) ====================
BOOT_ARGS="--no-spec" boot

drafts0=$(m ds4_spec_drafts_total)
c=$(sse nospec_stream /v1/chat/completions '{"model":"m","max_tokens":120,"temperature":0,"stream":true,"messages":[{"role":"user","content":"Name three planets, one per line."}]}')
code_is nospec_stream "$c" 200
shas nospec_stream 'data: \[DONE\]'
grep -qE '"finish_reason":"(stop|length)"' "$OUT/nospec_stream.sse" || \
  fail "nospec_stream: no terminal finish_reason"
drafts1=$(m ds4_spec_drafts_total)
[ "${drafts1:-0}" -eq "${drafts0:-0}" ] || fail "nospec_stream: spec_drafts moved under --no-spec (${drafts0:-?} -> ${drafts1:-?})"
log "nospec_stream PASS (spec OFF; drafts unmoved at ${drafts1:-0})"

c=$(sse nospec_tools /v1/messages '{"model":"m","max_tokens":1200,"temperature":0,"stream":true,"messages":[{"role":"user","content":"Use the list_files tool to list the files in /tmp. Call the tool."}],"tools":'"$TOOLS_ANTH"'}')
code_is nospec_tools "$c" 200
shas nospec_tools '"stop_reason":"tool_use"'
shas nospec_tools 'input_json_delta'
shas nospec_tools 'event: message_stop'
log "nospec_tools PASS (accumulator DSML override serves tools without spec)"

# ==================== boot 3: /v1/batch fairness observation ================
BOOT_ARGS="" boot

waits0=$(m ds4_batch_genmu_waits_total); wns0=$(m ds4_batch_genmu_wait_ns_total)
c=$(post batch_idle /v1/batch '{"prompts":["Say hi."],"max_tokens":24}')
code_is batch_idle "$c" 200
has batch_idle '"object":"batch"'
waits1=$(m ds4_batch_genmu_waits_total); wns1=$(m ds4_batch_genmu_wait_ns_total)
[ "${waits1:-0}" -gt "${waits0:-0}" ] || fail "batch_idle: waits counter never moved (metric dead)"
idle_ms=$(( ( ${wns1:-0} - ${wns0:-0} ) / 1000000 ))
log "batch_idle PASS (gen_mu wait ${idle_ms} ms on an idle box)"

# batch_vs_cont: the starvation number.  A live cont stream holds gen_mu for
# its whole epoch; the /v1/batch posted mid-decode waits for the drain.
( c=$(sse cont_occupier /v1/chat/completions '{"model":"m","max_tokens":400,"temperature":0,"stream":true,"messages":[{"role":"user","content":"Write a short story about a lighthouse keeper, about 300 words."}]}')
  echo "$c" > "$OUT/cont_occupier.code" ) &
OCC_PID=$!
sleep 2
t0=$(date +%s)
c=$(post batch_vs_cont /v1/batch '{"prompts":["Name one color."],"max_tokens":16}')
t1=$(date +%s)
code_is batch_vs_cont "$c" 200
has batch_vs_cont '"object":"batch"'
waits2=$(m ds4_batch_genmu_waits_total); wns2=$(m ds4_batch_genmu_wait_ns_total)
[ "${waits2:-0}" -gt "${waits1:-0}" ] || fail "batch_vs_cont: waits counter never moved"
busy_ms=$(( ( ${wns2:-0} - ${wns1:-0} ) / 1000000 ))
log "batch_vs_cont PASS (200 in $((t1-t0))s wall; gen_mu wait ${busy_ms} ms behind a live cont epoch -- the cut-list starvation datum)"
wait "$OCC_PID" 2>/dev/null
[ "$(cat "$OUT/cont_occupier.code")" = 200 ] || fail "cont_occupier stream failed"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT"
