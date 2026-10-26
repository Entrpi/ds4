#!/bin/bash
# serving/streaming_close.sh — v0.5.6 API Inc 4 CLOSE: the plan §5 gate line's
# remaining legs on the PROMOTED STREAMING surfaces.  serving/cont_streaming.sh
# owns automata/reassembly/deep-prefill/kill-switch; perf/batched_lane_abba.sh owns
# the §7 perf arm; the full battery runs after both.
#
# Boot 1 (zero-config + DS4_MEM_FLOOR_GB=2):
#   mixed_stream_1111  ONE concurrent STREAMING request per surface (chat,
#                      legacy completion, Anthropic, Responses): every
#                      terminal native, every cont lane cell +1, serial
#                      unmoved.
#   batch_survival     three concurrent Anthropic streams, the middle one's
#                      client killed mid-decode: victim CANCELED, the OTHER
#                      TWO complete valid -- one row's death never poisons
#                      the batch.
#   queue_cancel       a serial blocker holds gen_mu while a promoted stream
#                      queues behind it; the queued client dies BEFORE
#                      admission -> zombie-reaped CANCELED, its cont lane
#                      cell never moves (queue-phase cancellation).
#   ledger             started == completed + canceled.
#
# Boot 2 (DS4_SERVER_OUT_AGG_CAP=2048 + EVICT_MIN=1024 +
#         DS4_SERVER_CLIENT_SNDBUF=8192 + floor 2 -- the error_envelope
#         boot-D recipe, retargeted at the PROMOTED surface):
#   slow_reader        an ON-BOX stalled Anthropic stream reader (the ssh
#                      tunnel is itself an aggressive reader, so a stalled
#                      client BEHIND the tunnel never backs the server up --
#                      measured; the 2e leg learned the same about loopback
#                      sndbuf autotune): reads the headers then stops; the
#                      aggregate out-backlog cap evicts it ->
#                      shed{slow_reader} + CANCELED, server healthy
#                      (write-phase cancellation).
#
# Boot 3 (DS4_GATE_CONT_FAIL_AFTER_STEPS=4 + floor 2, the Inc 2c teeth):
#   strand_anth        engine strands a live Anthropic stream -> the native
#                      anthropic error EVENT on the wire, server survives.
#   strand_resp        same for Responses (data: {"type":"error",...} with
#                      a spliced sequence_number).
#
# Runs FROM the Mac over the ssh tunnel.  End state: server killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/inc4_close_gate
OUT=${OUT:-/tmp/inc4_close_gate_$$}
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

# number extraction BEFORE head (METRICS-HELPER TRAP).
m(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$' | head -1; }
lane(){ m "surface=\"$1\",lane=\"$2\""; }
sse_has(){ grep -q "$2" "$OUT/$1" || fail "$1: missing [$2]"; }
delta_is(){ [ $(( $3 - $2 )) -eq "$4" ] || fail "$1: delta $(( $3 - $2 )), want $4"; }
# Echoes the value once it exceeds the floor; returns 1 on timeout.  Callers
# MUST `|| fail` -- a fail() inside $(...) exits only the subshell (measured:
# a timed-out leg logged PASS with the failure text embedded in the value).
wait_counter(){ # $1=name $2=metric-expr $3=floor(exclusive) [$4=iterations]
  local n=0 v
  while :; do
    v=$(m "$2")
    [ -n "$v" ] && [ "$v" -gt "$3" ] && { echo "$v"; return 0; }
    n=$((n+1)); [ $n -ge "${4:-30}" ] && { echo "TIMEOUT:${v:-none}"; return 1; }
    sleep 2
  done
}

# ==================== boot 1: mixed streams + survival ======================
BOOT_ENV="DS4_MEM_FLOOR_GB=2" boot

# ---- mixed_stream_1111 ------------------------------------------------------
oc0=$(lane openai_chat continuous); cc0=$(lane openai_completion continuous)
ac0=$(lane anthropic_messages continuous); rc0=$(lane openai_responses continuous)
sl0=$(m ds4_requests_serial_total)
curl -s -N -m 240 -o "$OUT/mx_chat.sse" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"stream":true,"max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one planet."}]}' &
curl -s -N -m 240 -o "$OUT/mx_comp.sse" "$BASE/v1/completions" \
     -H 'Content-Type: application/json' \
     -d '{"stream":true,"prompt":"The capital of France is","max_tokens":48,"temperature":0}' &
curl -s -N -m 240 -o "$OUT/mx_anth.sse" "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","stream":true,"max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one ocean."}]}' &
curl -s -N -m 240 -o "$OUT/mx_resp.sse" "$BASE/v1/responses" \
     -H 'Content-Type: application/json' \
     -d '{"stream":true,"max_output_tokens":400,"temperature":0,"input":"Name one river."}' &
wait
sse_has mx_chat.sse 'data: \[DONE\]'
sse_has mx_chat.sse '"object":"chat.completion.chunk"'
sse_has mx_comp.sse 'data: \[DONE\]'
sse_has mx_comp.sse '"object":"text_completion"'
sse_has mx_anth.sse '"type":"message_stop"'
sse_has mx_resp.sse '"type":"response.completed"'
oc1=$(lane openai_chat continuous); cc1=$(lane openai_completion continuous)
ac1=$(lane anthropic_messages continuous); rc1=$(lane openai_responses continuous)
sl1=$(m ds4_requests_serial_total)
delta_is "mixed chat cont"  "$oc0" "$oc1" 1
delta_is "mixed comp cont"  "$cc0" "$cc1" 1
delta_is "mixed anth cont"  "$ac0" "$ac1" 1
delta_is "mixed resp cont"  "$rc0" "$rc1" 1
delta_is "mixed serial"     "$sl0" "$sl1" 0
log "mixed_stream_1111 PASS (four surfaces streamed one cont batch)"

# ---- batch_survival ---------------------------------------------------------
ca0=$(m 'ds4_requests_total{outcome="canceled"}')
curl -s -N -m 240 -o "$OUT/sv_a.sse" "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","stream":true,"max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name two colors."}]}' &
curl -s -N -m 3 -o "$OUT/sv_victim.sse" "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","stream":true,"max_tokens":1500,"temperature":0,"messages":[{"role":"user","content":"Write a long essay about rivers."}]}' &
curl -s -N -m 240 -o "$OUT/sv_b.sse" "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","stream":true,"max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name two trees."}]}' &
wait
sse_has sv_a.sse '"type":"message_stop"'
sse_has sv_b.sse '"type":"message_stop"'
ca1=$(wait_counter batch_survival 'ds4_requests_total{outcome="canceled"}' "$ca0") || \
  fail "batch_survival: canceled never moved ($ca1)"
log "batch_survival PASS (victim canceled ${ca0}->${ca1}, siblings completed)"

# ---- queue_cancel: killed while QUEUED behind a serial blocker --------------
# The blocker is a buffered return_token_ids chat request: TOKEN_IDS off the
# streaming-chat shape routes SERIAL.  Engagement proof is the serial counter
# (token_ids renders only on the STREAMING echo path -- buffered responses
# never carry the field).
ac0=$(lane anthropic_messages continuous)
sl0=$(m ds4_requests_serial_total)
ca0=$ca1
curl -s -m 240 -o "$OUT/qc_blocker.json" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"max_tokens":200,"temperature":0,"return_token_ids":true,"thinking":false,"messages":[{"role":"user","content":"Count upward from one to forty as words."}]}' &
BLOCKER=$!
sleep 1
curl -s -N -m 1 -o /dev/null "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","stream":true,"max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one bird."}]}' \
     2>/dev/null
wait $BLOCKER
grep -q '"finish_reason"' "$OUT/qc_blocker.json" || fail "queue_cancel: blocker request failed"
sl1=$(m ds4_requests_serial_total)
delta_is "queue_cancel blocker serial" "$sl0" "$sl1" 1
ca1=$(wait_counter queue_cancel 'ds4_requests_total{outcome="canceled"}' "$ca0") || \
  fail "queue_cancel: canceled never moved ($ca1)"
ac1=$(lane anthropic_messages continuous)
delta_is "queue_cancel anth cont (victim never admitted)" "$ac0" "$ac1" 0
log "queue_cancel PASS (canceled ${ca0}->${ca1}, blocker serial, cont cell unmoved)"

# ---- boot 1 ledger -----------------------------------------------------------
st=$(m ds4_requests_started_total); cp=$(m 'ds4_requests_total{outcome="completed"}')
ca=$(m 'ds4_requests_total{outcome="canceled"}')
[ "$st" -eq $(( cp + ca )) ] || fail "ledger: started=$st != completed=$cp + canceled=$ca"
log "ledger PASS (started=$st completed=$cp canceled=$ca)"

# ==================== boot 2: slow-reader write-phase cap ===================
BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_SERVER_OUT_AGG_CAP=2048 DS4_SERVER_OUT_AGG_EVICT_MIN=1024 DS4_SERVER_CLIENT_SNDBUF=8192" boot
sr0=$(m 'reason="slow_reader"')
ca0=$(m 'ds4_requests_total{outcome="canceled"}')
ssh "$R" "cat > /tmp/stall_anth_reader.py" <<PY
import json, socket, time
body = json.dumps({"model": "m", "stream": True, "max_tokens": 3000, "temperature": 0,
                   "messages": [{"role": "user", "content": "Tell a very long story about rivers."}]})
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
s.connect(("127.0.0.1", $PORT))
req = "POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body)
s.sendall(req.encode())
s.recv(256)          # response started (admission-time headers)
time.sleep(240)      # never read again; the server must evict us
PY
STALL_PID=$(ssh "$R" "cd /tmp && setsid nohup python3 /tmp/stall_anth_reader.py > /tmp/stall_anth_reader.log 2>&1 & echo \$!")
sr1=$(wait_counter slow_reader 'reason="slow_reader"' "${sr0:-0}" 60) || \
  fail "slow_reader: shed counter never moved ($sr1)"
ca1=$(wait_counter slow_reader_cancel 'ds4_requests_total{outcome="canceled"}' "$ca0" 60) || \
  fail "slow_reader: canceled never moved ($ca1)"
ssh "$R" "kill $STALL_PID 2>/dev/null; exit 0"
hc=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/v1/models")
[ "$hc" = "200" ] || fail "slow_reader: health probe HTTP $hc"
curl -s -m 240 -o "$OUT/sr_recover.json" "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one color."}]}'
grep -q '"stop_reason"' "$OUT/sr_recover.json" || fail "slow_reader: recovery request failed"
log "slow_reader PASS (shed ${sr0:-0}->${sr1}, canceled ${ca0}->${ca1}, recovered)"

# ==================== boot 3: engine strand -> native stream errors =========
BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_GATE_CONT_FAIL_AFTER_STEPS=4" boot
curl -s -N -m 60 -o "$OUT/strand_anth.sse" "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","stream":true,"max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one planet."}]}' \
     || true
sse_has strand_anth.sse 'event: error'
sse_has strand_anth.sse '"type":"error","error":{"type":"api_error"'
hc=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/v1/models")
[ "$hc" = "200" ] || fail "strand_anth: health probe HTTP $hc"
log "strand_anth PASS (native anthropic error event, server alive)"

curl -s -N -m 60 -o "$OUT/strand_resp.sse" "$BASE/v1/responses" \
     -H 'Content-Type: application/json' \
     -d '{"stream":true,"max_output_tokens":400,"temperature":0,"input":"Name one planet."}' \
     || true
sse_has strand_resp.sse '"type":"error"'
sse_has strand_resp.sse '"code":"server_error"'
sse_has strand_resp.sse '"sequence_number"'
hc=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/v1/models")
[ "$hc" = "200" ] || fail "strand_resp: health probe HTTP $hc"
log "strand_resp PASS (native data: error event with sequence_number)"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT"
