#!/bin/bash
# gates/legs/memory/pool_truth.sh — MT-6 (v0.6.1 memory truth): pool truth.
#
# What changed and what this gate pins:
#   V9: DS4_MEMC_GRAPH_EXEC is finally WRITTEN -- every cudaGraphInstantiate
#       notes an empirical driver-free delta (per-slot recorded, released
#       verbatim at destroy), so the /metrics "graph_exec" row reports the
#       captured-exec pool instead of a permanent zero.  An idle sweep
#       (hits==0 since last window) rides ds4_gpu_own_trim at the pressure
#       ladder + serial reaper (DS4_GRAPH_EXEC_TRIM=0 disables), and
#       ds4_session_free drops the SERIAL layer pool outright -- before
#       this, every reap leaked the whole pool's execs (stale baked
#       pointers, never replayed, never destroyed) until an unrelated
#       scratch resize swept them.
#   V10: q8-f16 cache budget + managed-KV routing read the ONE typed
#       observation (gap-3/C11+C12 closed); the out_a cublas+f16-cache
#       fallback is default OFF (outa_own replaced it; the classic tiers
#       serve when outa_own declines).  DS4_CUDA_CUBLAS_ATTENTION_OUTPUT_A=1
#       re-opts.
#
# Legs:
#   G:  default boot + serial turn + cont turn -> graph_exec census > 0
#       (the class reports), outa_own engage line present, weight_derived
#       census == 0 (no f16 cache build in ship config).
#   R:  reap (15 s window) -> "dropped N serial layer exec(s)" log +
#       graph_exec live census FALLS vs its pre-reap reading.
#   O:  DS4_GRAPH_EXEC_TRIM=0 boot -> reap still drops the layer pool
#       (correctness is not gated) but no "graph-exec idle trim" line.
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
SRV=/tmp/mt6_pool_gate_srv.log
OUT=${OUT:-/tmp/mt6_pool_gate_$$}
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
  log "boot: killing old $BIN on $R (env='$1')"
  ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
  wait_mem
  ssh "$R" ": > $SRV; cd $BINDIR; $1 setsid nohup ./$BIN -c 16384 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
  log "boot up"
}

metrics(){ ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/metrics"; }
census_mib(){ metrics | grep "class=\"$1\",state=\"allocated\"" | awk '{s+=$2} END {printf "%d\n", s/1048576}'; }
serial_turn(){ ssh "$R" "curl -s -m 120 -o /tmp/m6_$1.json -w '%{http_code}' \
    http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to ten in words.\"}],\"max_tokens\":64,\"return_token_ids\":true}'"; }
cont_turn(){ ssh "$R" "curl -s -m 120 -o /tmp/m6_$1.json -w '%{http_code}' \
    http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Name four rivers, one per line.\"}],\"max_tokens\":64}'"; }

run_traffic(){ # serial + cont turns so both layer and cont pools capture
  HTTP=$(serial_turn s); [ "$HTTP" = "200" ] || fail "$1: serial turn HTTP $HTTP"
  HTTP=$(cont_turn c);   [ "$HTTP" = "200" ] || fail "$1: cont turn HTTP $HTTP"
}

# ---------- Leg G: the class reports ----------
boot "DS4_SERIAL_IDLE_REAP_S=15"
run_traffic "leg G"
GE0=$(census_mib graph_exec); WD0=$(census_mib weight_derived)
[ -n "$GE0" ] && [ "$GE0" -ge 1 ] || fail "leg G: graph_exec census ${GE0:-?} MiB (class still silent)"
[ "${WD0:-0}" = "0" ] || fail "leg G: weight_derived $WD0 MiB in ship config (fallback cache built?)"
[ "$(srv_count 'attention out_a fused own kernel engage')" -ge 1 ] \
  || fail "leg G: outa_own engage line missing (ship path changed)"
log "G PASS (graph_exec=$GE0 MiB live; weight_derived=0; outa_own engaged)"

# ---------- Leg R: session-free drop + census release ----------
log "leg R: idling 35s past the 15s reap window"
sleep 35
[ "$(srv_count 'serial idle reap')" -ge 1 ] || fail "leg R: reap never fired"
[ "$(srv_count 'dropped .* serial layer exec')" -ge 1 ] \
  || fail "leg R: no layer-pool drop at session free (the leak is back)"
GE1=$(census_mib graph_exec)
[ -n "$GE1" ] && [ "$GE1" -lt "$GE0" ] \
  || fail "leg R: graph_exec census ${GE1:-?} did not fall from $GE0 MiB after the drop"
log "R PASS (layer pool dropped with the session; graph_exec $GE0 -> $GE1 MiB)"

# ---------- Leg O: sweep kill switch; drop is not gated ----------
boot "DS4_SERIAL_IDLE_REAP_S=15 DS4_GRAPH_EXEC_TRIM=0"
run_traffic "leg O"
sleep 35
[ "$(srv_count 'serial idle reap')" -ge 1 ] || fail "leg O: reap never fired"
[ "$(srv_count 'dropped .* serial layer exec')" -ge 1 ] \
  || fail "leg O: layer-pool drop must run regardless of the sweep switch"
[ "$(srv_count 'graph-exec idle trim')" = "0" ] \
  || fail "leg O: idle sweep ran with DS4_GRAPH_EXEC_TRIM=0"
log "O PASS (drop unconditional; sweep respects the kill switch)"

ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null
log "ALL LEGS PASS — artifacts in $OUT ($BIN killed, $R left free)"
