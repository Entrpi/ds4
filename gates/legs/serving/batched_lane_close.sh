#!/bin/bash
# serving/batched_lane_close.sh — v0.5.6 API Inc 3 CLOSE: the plan §5 gate line's
# mixed-traffic + abort + capacity legs on the PROMOTED surfaces (§6.3
# parameterizes the existing failure gates across surfaces; the §7 perf
# acceptance runs separately in perf/batched_lane_abba.sh; the full battery runs
# after both).
#
# Boot 1 (zero-config + DS4_MEM_FLOOR_GB=2):
#   mixed_1111   ONE concurrent request per surface (chat STREAMING, legacy
#                completion, Anthropic, Responses buffered): every response
#                native, every surface's cont lane cell +1, serial unmoved,
#                and the boot's ledger closes (started == completed).
#   abort_anth   mid-decode client disconnect on a buffered Anthropic row
#                (budget 1500, curl -m 4): settles CANCELED (Inc 2c bucket),
#                server healthy after (probe 200).
#   abort_resp   same on Responses.
#   ledger       boot-delta identity: started == completed + canceled.
#
# Boot 2 (MAX_CLIENTS=2 + MAX_QUEUE_BYTES=5000 + DS4_MEM_FLOOR_GB=2):
#   cap_anth_429 an ~8 KB /v1/messages body sheds PRE-PARSE on the bytes
#                bound: 429 + Retry-After + NATIVE anthropic envelope
#                {"type":"error","error":{"type":"rate_limit_error"}} +
#                shed{queue_bytes}.  (The anthropic 503 overloaded_error
#                twin lives in error_envelope_gate's clients_shed leg.)
#   cap_resp_503 two parked admission streams + a /v1/responses probe = 3
#                connections > 2: 503 + Retry-After + the OpenAI-family
#                envelope ("type":"server_error") + shed{clients}.
#
# Runs FROM the Mac over SSH like the other gates.  End state: server
# killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/inc3_close_gate
OUT=${OUT:-/tmp/inc3_close_gate_$$}
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

hpost(){
  curl -s -m 180 -o "$OUT/$1.json" -D "$OUT/$1.hdr" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
has(){ grep -q "$2" "$OUT/$1.json" || fail "$1: missing [$2] in $(head -c 300 "$OUT/$1.json")"; }
code_is(){ [ "$2" = "$3" ] || fail "$1: HTTP $2, want $3 ($(head -c 300 "$OUT/$1.json"))"; }
hdr_has(){ grep -qi "$2" "$OUT/$1.hdr" || fail "$1: missing header [$2] in $(tr -d '\r' < "$OUT/$1.hdr" | tr '\n' ' ')"; }
# number extraction BEFORE head: the fixed-string grep also matches the
# "# TYPE <name> counter" comment for unlabeled metrics, and that line has
# no trailing number -- head-first returns empty and every delta reads 0.
m(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$' | head -1; }
lane(){ m "surface=\"$1\",lane=\"$2\""; }

# ==================== boot 1: mixed traffic + aborts =======================
BOOT_ENV="DS4_MEM_FLOOR_GB=2" boot

# mixed_1111: one request per surface, concurrent.  The chat row STREAMS
# (thinking default LOW rides the cont SSE machine); the other three are
# buffered.  400-token budgets so every lane EOSes (result-class law).
oc0=$(lane openai_chat continuous); cc0=$(lane openai_completion continuous)
ac0=$(lane anthropic_messages continuous); rc0=$(lane openai_responses continuous)
sl0=$(m ds4_requests_serial_total)
st0=$(m ds4_requests_started_total)
cp0=$(m 'ds4_requests_total{outcome="completed"}')
curl -s -N -m 180 -o "$OUT/mix_chat.sse" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"stream":true,"max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one planet."}]}' &
curl -s -m 180 -o "$OUT/mix_comp.json" "$BASE/v1/completions" \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"The capital of France is","max_tokens":48,"temperature":0}' &
curl -s -m 180 -o "$OUT/mix_anth.json" "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one ocean."}]}' &
curl -s -m 180 -o "$OUT/mix_resp.json" "$BASE/v1/responses" \
     -H 'Content-Type: application/json' \
     -d '{"max_output_tokens":400,"temperature":0,"input":"Name one river."}' &
wait
grep -q 'data: \[DONE\]' "$OUT/mix_chat.sse" || fail "mixed_1111: chat stream missing [DONE]"
grep -q '"finish_reason":"stop"' "$OUT/mix_chat.sse" || fail "mixed_1111: chat stream never finished stop"
grep -q '"object":"text_completion"' "$OUT/mix_comp.json" || fail "mixed_1111: completion shape missing"
grep -q '"type":"message"' "$OUT/mix_anth.json" || fail "mixed_1111: anthropic shape missing"
grep -q '"stop_reason"' "$OUT/mix_anth.json" || fail "mixed_1111: anthropic stop_reason missing"
grep -q '"object":"response"' "$OUT/mix_resp.json" || fail "mixed_1111: responses shape missing"
grep -q '"status":"completed"' "$OUT/mix_resp.json" || fail "mixed_1111: responses not completed"
oc1=$(lane openai_chat continuous); cc1=$(lane openai_completion continuous)
ac1=$(lane anthropic_messages continuous); rc1=$(lane openai_responses continuous)
sl1=$(m ds4_requests_serial_total)
[ "${oc1:-0}" -gt "${oc0:-0}" ] || fail "mixed_1111: chat lost the cont lane (${oc0:-?} -> ${oc1:-?})"
[ "${cc1:-0}" -gt "${cc0:-0}" ] || fail "mixed_1111: completion lost the cont lane (${cc0:-?} -> ${cc1:-?})"
[ "${ac1:-0}" -gt "${ac0:-0}" ] || fail "mixed_1111: anthropic lost the cont lane (${ac0:-?} -> ${ac1:-?})"
[ "${rc1:-0}" -gt "${rc0:-0}" ] || fail "mixed_1111: responses lost the cont lane (${rc0:-?} -> ${rc1:-?})"
[ "${sl1:-0}" -eq "${sl0:-0}" ] || fail "mixed_1111: a surface FELL BACK to serial (${sl0:-?} -> ${sl1:-?})"
log "mixed_1111 PASS (four surfaces concurrent, all native, all SERVED on cont)"

# abort_anth / abort_resp: mid-decode disconnect settles CANCELED (Inc 2c);
# the Inc 3 promotions must keep the disconnect-abort probe working for
# promoted buffered rows (cont_on_token probes BEFORE the plain-row early
# return, and the 3c zero-budget return sits after the probes too).
for surf in anth resp; do
  ca0=$(m 'ds4_requests_total{outcome="canceled"}')
  if [ "$surf" = anth ]; then
    curl -s -m 4 -o /dev/null "$BASE/v1/messages" -H 'Content-Type: application/json' \
         -d '{"model":"m","max_tokens":1500,"temperature":0,"messages":[{"role":"user","content":"Tell a very long story about mountains."}]}' || true
  else
    curl -s -m 4 -o /dev/null "$BASE/v1/responses" -H 'Content-Type: application/json' \
         -d '{"max_output_tokens":1500,"temperature":0,"input":"Tell a very long story about deserts."}' || true
  fi
  n=0
  while :; do
    ca1=$(m 'ds4_requests_total{outcome="canceled"}')
    [ "${ca1:-0}" -gt "${ca0:-0}" ] && break
    n=$((n+1)); [ $n -ge 12 ] && fail "abort_$surf: canceled never incremented (${ca0:-?} -> ${ca1:-?})"
    sleep 5
  done
  c=$(hpost "abort_${surf}_health" /v1/chat/completions '{"max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"hi"}]}')
  code_is "abort_${surf}_health" "$c" 200
  log "abort_$surf PASS (mid-decode disconnect settled CANCELED; server healthy after)"
done

# ledger: the boot's outcome identity (started == completed + canceled;
# nothing failed, shed, or refused in this boot).
st1=$(m ds4_requests_started_total)
cp1=$(m 'ds4_requests_total{outcome="completed"}')
ca=$(m 'ds4_requests_total{outcome="canceled"}')
fd=$(m 'ds4_requests_total{outcome="failed"}')
st_d=$(( ${st1:-0} - ${st0:-0} )); cp_d=$(( ${cp1:-0} - ${cp0:-0} ))
[ "$st_d" -eq $(( cp_d + ${ca:-0} )) ] && [ "${fd:-0}" -eq 0 ] \
  || fail "ledger: started_d=$st_d != completed_d=$cp_d + canceled=$ca (failed=$fd)"
log "ledger PASS (started == completed + canceled across the boot; failed 0)"

# ==================== boot 2: capacity on the promoted surfaces ============
BOOT_ENV="DS4_SERVER_MAX_CLIENTS=2 DS4_SERVER_MAX_QUEUE_BYTES=5000 DS4_MEM_FLOOR_GB=2" boot

qb0=$(m 'ds4_requests_shed_total{reason="queue_bytes"}')
python3 - > "$OUT/big_anth.json" <<'PY'
import json
body = "All the rivers of the northern watershed drain toward the delta. " * 130
print(json.dumps({"model": "m", "max_tokens": 16,
                  "messages": [{"role": "user", "content": body}]}))
PY
c=$(curl -s -m 60 -o "$OUT/cap_anth_429.json" -D "$OUT/cap_anth_429.hdr" -w '%{http_code}' \
     "$BASE/v1/messages" -H 'Content-Type: application/json' -d @"$OUT/big_anth.json")
code_is cap_anth_429 "$c" 429
has cap_anth_429 '"type":"error","error":{"type":"rate_limit_error"'
hdr_has cap_anth_429 'Retry-After:'
qb1=$(m 'ds4_requests_shed_total{reason="queue_bytes"}')
[ "${qb1:-0}" -gt "${qb0:-0}" ] || fail "cap_anth_429: shed{queue_bytes} never incremented (${qb0:-?} -> ${qb1:-?})"
log "cap_anth_429 PASS (pre-parse 429 + Retry-After, native anthropic envelope)"

cs0=$(m 'ds4_requests_shed_total{reason="clients"}')
# Parking on THIS boot must dodge the 5000-byte queue bound the 429 leg
# needs (the envelope gate's big-prompt parks would shed pre-parse here and
# hold nothing -- measured: the probe got a 200).  Small body, long DECODE:
# the stream holds its connection for minutes of generation instead.
python3 - > "$OUT/park_req.json" <<'PY'
import json
print(json.dumps({"stream": True, "max_tokens": 3000, "temperature": 0,
                  "messages": [{"role": "user", "content": "Tell a very long story about rivers."}]}))
PY
curl -s -m 150 -o /dev/null "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' -d @"$OUT/park_req.json" &
PARK1=$!
curl -s -m 150 -o /dev/null "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' -d @"$OUT/park_req.json" &
PARK2=$!
sleep 3
c=$(hpost cap_resp_503 /v1/responses '{"max_output_tokens":8,"input":"hi"}')
code_is cap_resp_503 "$c" 503
has cap_resp_503 '"error"'
has cap_resp_503 '"type":"server_error"'
hdr_has cap_resp_503 'Retry-After:'
cs1=$(m 'ds4_requests_shed_total{reason="clients"}')
[ "${cs1:-0}" -gt "${cs0:-0}" ] || fail "cap_resp_503: shed{clients} never incremented (${cs0:-?} -> ${cs1:-?})"
kill "$PARK1" "$PARK2" 2>/dev/null || true
log "cap_resp_503 PASS (503 + Retry-After at 3 connections, OpenAI-family envelope)"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT (server killed, $R left free)"
