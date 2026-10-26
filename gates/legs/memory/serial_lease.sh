#!/bin/bash
# gates/legs/memory/serial_lease.sh — MT-2 (v0.6.1 memory truth): honest
# serial session-graph lease + census scope.
#
# Before MT-2, the serial lane published intent==resident==estimate while
# its graph was still lazy-PENDING (the D0b-3 "commit-lead window"), so the
# multi-GiB build window contributed ZERO unfunded debt to the other lanes'
# governed quotes; and ds4_session_alloc_graph ran the burst with NO census
# scope, so every non-KV tensor of the graph (~3.98 GiB pc-scratch group at
# pc 4096) landed in ENGINE_OTHER — the leg-1 "engine_other 5.03 GiB"
# mystery, violating the documented ENGINE_OTHER live == 0 invariant.
#
# MT-2 moves the lease lifecycle into ds4_session_alloc_graph itself
# ((est,0) before the burst, (est,est) at commit, (0,0) on failure —
# covering all 7 ensure entry points) and brackets the body in
# SESSION_TENSORS.  The window ARITHMETIC (a bank claim charges the whole
# open window, zero after commit) is pinned by the
# test_mem_gov_serial_window_lease unit; this gate asserts the LIVE truths:
#
#   S:  a buffered tool turn (serial-held route) commits the lazy graph ->
#       lease shows intent == resident > 500 MiB (committed, honest);
#       census: session_tensors grew >= 500 MiB and engine_other grew
#       < 64 MiB (the scope fix — pre-MT-2 this delta was the whole non-KV
#       graph); the "session graph allocated lazily" line printed; zero
#       governor/census faults.
#   C:  a cont row admits clean AFTER the commit (the committed lease does
#       not double-charge: resident bytes are already in raw free).
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
CTX=${CTX:-16384}
SRV=/tmp/serial_lease_gate_srv.log
OUT=${OUT:-/tmp/serial_lease_gate_$$}
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

metrics(){ ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/metrics"; }
# Sum a census class's allocated bytes across domains -> MiB (integer).
census_mib(){ # $1 = class name
  metrics | grep "class=\"$1\",state=\"allocated\"" | \
    awk '{s+=$2} END {printf "%d\n", s/1048576}'; }
lease_mib(){ # $1 = field (intent|resident)
  metrics | grep "ds4_memory_lease_bytes{consumer=\"serial_session\",field=\"$1\"}" | \
    awk '{printf "%d\n", $2/1048576}'; }
metric1(){ metrics | grep "^$1 " | awk '{print $2}'; }

log "boot: killing old $BIN on $R"
ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
wait_mem
ssh "$R" ": > $SRV; cd $BINDIR; DS4_SERVER_COALESCE_MAX=4 setsid nohup ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
  sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
done
log "boot up"

# ---------- baseline ----------
# The boot PREWARM session commits a small graph through the same lease
# lifecycle and is freed before the listener -- the phantom-release assert:
# a freed session must leave ZERO intent (live-caught 08-17: without the
# ds4_session_free release the prewarm left 746 MiB of stuck intent).
EO0=$(census_mib engine_other)
ST0=$(census_mib session_tensors)
LI0=$(lease_mib intent)
LZ0=$(srv_count 'session graph allocated lazily')
log "baseline: engine_other=${EO0} MiB session_tensors=${ST0} MiB serial_intent=${LI0} MiB lazy_commits=${LZ0}"
[ "${LI0:-0}" = "0" ] || fail "baseline: freed prewarm session left a phantom serial lease (${LI0} MiB intent)"

# ---------- Leg S: token-ids projection -> serial hold -> lazy graph commit ----------
# Non-streaming return_token_ids keeps the serial route (the token-ids
# projection has no cont equivalent off-streaming); its first root API
# ensures the pending boot session graph = THE burst.
HTTP=$(ssh "$R" "curl -s -m 120 -o /tmp/slg_ser_resp.json -w '%{http_code}' \
  http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Name three lighthouses in one sentence.\"}],\"max_tokens\":48,\"return_token_ids\":true}'")
[ "$HTTP" = "200" ] || fail "leg S: serial turn HTTP $HTTP: $(ssh "$R" "head -c 300 /tmp/slg_ser_resp.json")"
LZ1=$(srv_count 'session graph allocated lazily')
[ "$LZ1" -gt "$LZ0" ] || fail "leg S: the lazy graph never committed (the request did not ride serial?)"

LI1=$(lease_mib intent)
LR1=$(lease_mib resident)
[ -n "$LI1" ] && [ "$LI1" -ge 500 ] || fail "leg S: serial lease intent ${LI1:-?} MiB < 500 (no honest lease at commit)"
[ "$LI1" = "$LR1" ] || fail "leg S: committed lease intent $LI1 != resident $LR1 (the window never closed)"

EO1=$(census_mib engine_other)
ST1=$(census_mib session_tensors)
[ $((ST1 - ST0)) -ge 500 ] || fail "leg S: session_tensors grew only $((ST1-ST0)) MiB (graph not scoped into its class)"
[ $((EO1 - EO0)) -lt 64 ] || fail "leg S: engine_other grew $((EO1-EO0)) MiB (the scope fix regressed: graph bytes are untagged again)"

GF=$(metric1 ds4_memory_governor_faults_total)
CF=$(metric1 ds4_memory_census_faults_total)
[ "${GF:-0}" = "0" ] || fail "leg S: governor faults $GF"
[ "${CF:-0}" = "0" ] || fail "leg S: census faults $CF"
log "S PASS (commit lease ${LI1}/${LR1} MiB; session_tensors +$((ST1-ST0)) MiB; engine_other +$((EO1-EO0)) MiB; 0 faults)"

# ---------- Leg C: cont admits clean beside the committed lease ----------
REJ0=$(metric1 ds4_cont_admit_rejects_total)
ssh "$R" "curl -s -m 25 -o /tmp/slg_cont.json http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Tell me a short story about a lighthouse keeper.\"}]}'" 2>/dev/null
REJ1=$(metric1 ds4_cont_admit_rejects_total)
[ "$REJ1" = "$REJ0" ] || fail "leg C: cont reject beside the committed serial lease ($REJ0 -> $REJ1): double-charge"
ADM=$(metrics | grep '^ds4_admits_total' | awk '{s+=$2} END {print s+0}')
[ "${ADM:-0}" -ge 1 ] || fail "leg C: no cont admission recorded"
log "C PASS (cont admitted clean beside the committed lease, rejects unchanged)"

ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null
log "ALL LEGS PASS — artifacts in $OUT ($BIN killed, $R left free)"
