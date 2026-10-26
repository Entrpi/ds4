#!/bin/bash
# gates/legs/memory/serial_reap.sh — MT-3 (v0.6.1 memory truth): serial
# idle reaper.
#
# A committed serial session graph used to live until replace or shutdown:
# one serial detour permanently converted ~5 GiB on bank-holding boots (the
# leg-1 cliff's residue).  With DS4_SERIAL_IDLE_REAP_S (default 120; the
# gate pins 15) the worker replaces the idle session with a fresh pending
# boot-ctx one — from the dequeue tick when quiet, and from cont_admit when
# the worker lives inside the cont loop (exactly when the cont lane wants
# the memory).  MT-2's lease release + census scope make the reap fully
# observable: lease -> 0/0, session_tensors -> ~0, MemAvailable recovers.
#
# Legs:
#   R:  serial turn commits the graph (lease/census assert), idle 35 s ->
#       reap fired (log + counter), lease 0/0, session_tensors back under
#       64 MiB, MemAvailable recovered >= 2500 MiB of the graph's cost.
#   R2: the lane revives — a second serial turn re-allocs (HTTP 200 +
#       commit #2), proving the reap leaves a servable path.
#   B:  busy-cont reap — a long cont stream keeps the worker inside the
#       cont loop past the window; the reap fires FROM cont_admit while
#       the stream is still running.
#   O:  kill switch — reboot with window 0, serial turn, wait: zero reaps.
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
CTX=${CTX:-16384}
SRV=/tmp/serial_reap_gate_srv.log
OUT=${OUT:-/tmp/serial_reap_gate_$$}
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null; exit 1; }
srv_count(){ local c; c=$(ssh "$R" "grep -c \"$1\" $SRV 2>/dev/null || true" 2>/dev/null | tail -1); echo "${c:-0}"; }

wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable never reached 100G"; sleep 5
  done }

boot(){ # $1 = extra env
  log "boot: killing old $BIN on $R"
  ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
  wait_mem
  ssh "$R" ": > $SRV; cd $BINDIR; DS4_SERVER_COALESCE_MAX=4 $1 setsid nohup ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
  log "boot up"
}

metrics(){ ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/metrics"; }
census_mib(){ metrics | grep "class=\"$1\",state=\"allocated\"" | awk '{s+=$2} END {printf "%d\n", s/1048576}'; }
lease_mib(){ metrics | grep "ds4_memory_lease_bytes{consumer=\"serial_session\",field=\"$1\"}" | awk '{printf "%d\n", $2/1048576}'; }
metric1(){ metrics | grep "^$1 " | awk '{print $2}'; }
memavail_mib(){ ssh "$R" "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo"; }
serial_turn(){ # $1 = out tag; a return_token_ids non-streaming turn = serial hold
  ssh "$R" "curl -s -m 120 -o /tmp/srg_$1.json -w '%{http_code}' \
    http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Name three lighthouses in one sentence.\"}],\"max_tokens\":48,\"return_token_ids\":true}'"; }

# ---------- Leg R: quiet-server reap ----------
boot "DS4_SERIAL_IDLE_REAP_S=15"
# MT-5 reclassified the persistent BATCH ctx's graph tensors into
# SESSION_TENSORS (a no-serial boot shows ~4.5 GiB in the class), so the
# reap oracle is the DELTA back to this boot baseline -- never absolute
# zero.  ST0 = the batch share.
ST0=$(census_mib session_tensors); ST0=${ST0:-0}
HTTP=$(serial_turn r1)
[ "$HTTP" = "200" ] || fail "leg R: serial turn HTTP $HTTP"
LI=$(lease_mib intent); LR=$(lease_mib resident); ST=$(census_mib session_tensors)
[ -n "$LI" ] && [ "$LI" -ge 500 ] && [ "$LI" = "$LR" ] || fail "leg R: committed lease ${LI:-?}/${LR:-?} MiB (expected equal, >= 500)"
[ "$ST" -ge $((ST0 + 500)) ] || fail "leg R: session_tensors $ST MiB after commit (boot baseline $ST0)"
M1=$(memavail_mib)
log "leg R: committed (lease ${LI} MiB, session_tensors ${ST} MiB, MemAvailable ${M1} MiB); idling 35s past the 15s window"
sleep 35
RP=$(metric1 ds4_serial_idle_reaps_total)
[ "${RP:-0}" -ge 1 ] || fail "leg R: no reap within 35s (counter ${RP:-?}; log x$(srv_count 'serial idle reap'))"
[ "$(srv_count 'serial idle reap')" -ge 1 ] || fail "leg R: counter without the reap log line"
LI2=$(lease_mib intent); ST2=$(census_mib session_tensors)
[ "${LI2:-1}" = "0" ] || fail "leg R: lease intent $LI2 MiB after reap (phantom)"
[ "${ST2:-99999}" -le $((ST0 + 64)) ] || fail "leg R: session_tensors $ST2 MiB after reap (boot baseline $ST0; serial graph not freed)"
M2=$(memavail_mib)
[ $((M2 - M1)) -ge 2500 ] || fail "leg R: MemAvailable recovered only $((M2-M1)) MiB (expected >= 2500 of the ~${LI} MiB graph)"
log "R PASS (reap x${RP}; lease -> 0; session_tensors ${ST} -> ${ST2} MiB; MemAvailable +$((M2-M1)) MiB)"

# ---------- Leg R2: the lane revives ----------
LZ0=$(srv_count 'session graph allocated lazily')
HTTP=$(serial_turn r2)
[ "$HTTP" = "200" ] || fail "leg R2: post-reap serial turn HTTP $HTTP"
[ "$(srv_count 'session graph allocated lazily')" -gt "$LZ0" ] || fail "leg R2: no re-alloc after the reap"
LI3=$(lease_mib intent)
[ -n "$LI3" ] && [ "$LI3" -ge 500 ] || fail "leg R2: revived lease ${LI3:-?} MiB"
log "R2 PASS (re-alloc + committed lease ${LI3} MiB)"

# ---------- Leg B: reap fires FROM cont_admit under cont load ----------
RP1=$(metric1 ds4_serial_idle_reaps_total)
ssh "$R" "setsid nohup curl -s -m 300 -o /tmp/srg_cont.txt http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Count upward from 1, one number per line, without stopping and without any commentary.\"}],\"max_tokens\":1500,\"stream\":true}' \
  > /dev/null 2>&1 < /dev/null & exit 0" 2>/dev/null
RP2=$RP1
for i in $(seq 1 12); do
  sleep 5
  RP2=$(metric1 ds4_serial_idle_reaps_total)
  [ -n "$RP2" ] && [ "$RP2" -gt "${RP1:-0}" ] && break
done
[ -n "$RP2" ] && [ "$RP2" -gt "${RP1:-0}" ] || fail "leg B: no reap within 60s of cont load (counter ${RP2:-?})"
ssh "$R" "pgrep -x curl >/dev/null" || log "leg B note: cont stream already finished when the reap was observed"
log "B PASS (reap fired under cont load: $RP1 -> $RP2)"
ssh "$R" "pkill -x curl" 2>/dev/null
sleep 3

# ---------- Leg O: kill switch ----------
boot "DS4_SERIAL_IDLE_REAP_S=0"
HTTP=$(serial_turn o1)
[ "$HTTP" = "200" ] || fail "leg O: serial turn HTTP $HTTP"
sleep 30
[ "$(metric1 ds4_serial_idle_reaps_total)" = "0" ] || fail "leg O: reap fired with the window at 0"
[ "$(srv_count 'serial idle reap')" = "0" ] || fail "leg O: reap log line with the window at 0"
LI4=$(lease_mib intent)
[ -n "$LI4" ] && [ "$LI4" -ge 500 ] || fail "leg O: committed lease ${LI4:-?} MiB should persist with reap off"
log "O PASS (window 0: graph persists, lease ${LI4} MiB)"

# ---------- Leg E: eager-graph mode declines (teb fast-leg death repro) ----
# DS4_SESSION_LAZY_GRAPH=0 makes every session create materialize the full
# graph.  Pre-fix, the reaper freed ~6 GiB, re-materialized it in the same
# call, and re-fired every dequeue tick (1438 cycles in the field repro)
# until a recreate lost the fit race and die()d the server.  The fix: the
# due-check declines in eager mode.  Assert: window elapsed, ZERO reaps,
# server alive, serial still serves, lease intact.
boot "DS4_SERIAL_IDLE_REAP_S=15 DS4_SESSION_LAZY_GRAPH=0"
HTTP=$(serial_turn e1)
[ "$HTTP" = "200" ] || fail "leg E: serial turn HTTP $HTTP"
LE0=$(lease_mib intent)
[ -n "$LE0" ] && [ "$LE0" -ge 500 ] || fail "leg E: committed lease ${LE0:-?} MiB"
sleep 40
ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" || fail "leg E: server DIED in eager mode (the teb death shape)"
[ "$(metric1 ds4_serial_idle_reaps_total)" = "0" ] || fail "leg E: reap fired in eager-graph mode"
[ "$(srv_count 'serial idle reap')" = "0" ] || fail "leg E: reap log line in eager-graph mode"
LE1=$(lease_mib intent)
[ "$LE1" = "$LE0" ] || fail "leg E: lease moved ${LE0} -> ${LE1} MiB with no reap allowed"
HTTP=$(serial_turn e2)
[ "$HTTP" = "200" ] || fail "leg E: post-window serial turn HTTP $HTTP"
log "E PASS (eager mode: zero reaps past the window, server alive, lease ${LE1} MiB intact)"

ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null
log "ALL LEGS PASS — artifacts in $OUT ($BIN killed, $R left free)"
