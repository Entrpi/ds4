#!/bin/bash
# gates/legs/serving/abort_paths.sh — v0.5.2 inc3: dead-client abort gate.
#
# Field (forum 378855 post 6, emX0r: "memory never released after
# interruption").  Measured pre-fix on .33 (-c 65536): a killed STREAMING
# client stopped within one step (send-fail path, healthy); a killed
# NON-STREAMING client kept decoding ~21 tok/s toward its 4000-token budget
# indefinitely; a client killed MID-PREFILL had its ~45k prompt prefilled to
# completion (~80 s) and then STARTED its decode.  The fix probes the job's
# socket (zombie-reap MSG_PEEK, never consumes) at every abort point:
# admission-prefill chunks (engine alive callback), each cont decode token,
# each serial decode step.  DS4_SERVER_DISCONNECT_ABORT=0 restores v0.5.1.
#
#   leg abort_on   (default boot)
#     S1 streaming abort  -> ds4_tokens_decoded_total freezes within 10 s
#     S2 non-stream abort -> freezes within 10 s (the fixed blind spot)
#     S3 prefill abort    -> 'pending admission aborted' line; no decode
#                            from that row; prefill stops
#     S4 health           -> a normal completion still serves afterwards
#                            (banks reusable), requests_inflight settles 0
#   leg abort_off  (DS4_SERVER_DISCONNECT_ABORT=0)
#     S2 shape            -> decode KEEPS growing 15 s after the kill
#                            (escape works; reproduces the pre-fix waste)
#
# Runs FROM the Mac over SSH. End state: ds4-server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
CTX=${CTX:-65536}
SRV=/tmp/abort_gate_srv.log
OUT=${OUT:-/tmp/abort_paths_gate_$$}
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; ssh "$R" "pkill -x ds4-server" 2>/dev/null; exit 1; }
metric(){ curl -s -m 5 "http://127.0.0.1:$TUNNEL/metrics" | grep -oE "^$1 [0-9]+" | awk '{print $2}'; }

wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge "$1" ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable never reached ${1}G"; sleep 5
  done }

boot(){ # $1=extra env
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0"
  wait_mem 100
  ssh "$R" ": > $SRV; cd $BINDIR; env ${1:-} setsid nohup ./ds4-server -c $CTX --port $PORT \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ds4-server >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
  curl -s -m 5 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1 || {
    ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true; sleep 2
    curl -s -m 10 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null || fail "tunnel"
  }
}

fire_and_kill(){ # $1=tag $2=json-body-or-@file $3=kill-after-sec $4=streaming(1/0)
  local flags=""; [ "$4" = 1 ] && flags="-N"
  curl -s $flags -m 600 "http://127.0.0.1:$TUNNEL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$2" > "$OUT/$1.out" 2>&1 &
  local pid=$!
  sleep "$3"
  kill -9 $pid 2>/dev/null
  wait $pid 2>/dev/null
  log "$1: client killed after ${3}s"
}

# freeze check: decoded_total must be stable across 2 consecutive 5s samples
# within $2 seconds of now.
assert_freeze(){ # $1=tag $2=deadline_s
  local t=0 prev cur
  prev=$(metric ds4_tokens_decoded_total)
  while [ $t -lt "$2" ]; do
    sleep 5; t=$((t+5))
    cur=$(metric ds4_tokens_decoded_total)
    log "$1 +${t}s decoded=$cur (+$((cur-prev)))"
    if [ "$cur" = "$prev" ]; then
      sleep 5
      local cur2=$(metric ds4_tokens_decoded_total)
      [ "$cur2" = "$cur" ] && { log "$1 FROZEN at $cur"; return 0; }
      cur=$cur2
    fi
    prev=$cur
  done
  fail "$1: decode did not freeze within ${2}s of the kill"
}

STORY='{"messages":[{"role":"user","content":"Write a 3000-word story about the sea. Do not stop."}],"max_tokens":4000%s}'

log "=== leg abort_on: default boot ==="
boot ""
log "-- S1: streaming decode abort"
fire_and_kill s1 "$(printf "$STORY" ',"stream":true')" 12 1
assert_freeze S1 15
log "-- S2: NON-streaming decode abort (the fixed blind spot)"
fire_and_kill s2 "$(printf "$STORY" '')" 12 0
assert_freeze S2 15
log "-- S3: mid-prefill abort"
python3 - "$OUT/s3_body.json" <<'PY'
import json, sys
words = ("golf hotel india juliet kilo lima mike november oscar papa " * 4500).strip()
open(sys.argv[1],"w").write(json.dumps({"messages":[{"role":"user","content":
  words + "\nSummarize the above in one word."}],"max_tokens":2000,"stream":True}))
PY
d0=$(metric ds4_tokens_decoded_total)
fire_and_kill s3 "@$OUT/s3_body.json" 8 1
n=0
until ssh "$R" "grep -q 'pending admission aborted' $SRV" 2>/dev/null; do
  sleep 5; n=$((n+5)); [ $n -ge 40 ] && fail "S3: no 'pending admission aborted' line within 40s"
done
log "S3: abort line at +${n}s: $(ssh "$R" "grep -a 'pending admission aborted' $SRV | tail -1")"
sleep 10
d1=$(metric ds4_tokens_decoded_total)
[ $((d1 - d0)) -le 2 ] || fail "S3: decode ran after a prefill abort (decoded +$((d1-d0)))"
log "-- S4: health after aborts"
c=$(curl -s -m 120 -o "$OUT/s4.out" -w '%{http_code}' "http://127.0.0.1:$TUNNEL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}')
[ "$c" = 200 ] || fail "S4 http=$c"
grep -q '"content"' "$OUT/s4.out" || fail "S4 malformed"
sleep 3
infl=$(metric ds4_requests_inflight)
[ "${infl:-1}" = 0 ] || fail "S4: requests_inflight=$infl (leaked jobs)"
log "leg abort_on PASS"

log "=== leg abort_off: DS4_SERVER_DISCONNECT_ABORT=0 (escape) ==="
boot "DS4_SERVER_DISCONNECT_ABORT=0"
fire_and_kill off_s2 "$(printf "$STORY" '')" 12 0
p0=$(metric ds4_tokens_decoded_total); sleep 15
p1=$(metric ds4_tokens_decoded_total)
[ $((p1 - p0)) -ge 50 ] || fail "abort_off: decode stopped (grew only $((p1-p0))) — escape broken or shape changed"
log "abort_off: decode kept flowing (+$((p1-p0)) in 15s) = pre-fix behavior reproduced"
ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0"
log "ALL LEGS PASS — artifacts in $OUT ($R left free)"
