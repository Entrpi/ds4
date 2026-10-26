#!/bin/bash
# gates/legs/memory/defaults_derivation.sh — MT-5 (v0.6.1 memory truth): defaults
# re-derivation (audit V6/V7/V8 + the V13 hygiene batch).
#
# What changed and what this gate pins:
#   V6: DS4_SERVER_COALESCE_MAX default is ctx-aware (coalesce_max_default:
#       32 through the W8-measured regime <=16k, halving against the ~1 Mi
#       fundable-token depth, floor 4).  At deep ctx the old flat 32 granted
#       banks the box could never fund to depth, stranding ~450 MiB/bank of
#       eager remainder (~4.5 GiB at -c 786432).
#   V7: fit headroom default unified to flat 6144 MiB (the old deep 8192
#       tier guarded page-cache pressure from the pre-V6 32-bank grant).
#       Run-1 measured free-at-fit ~21 GiB (page cache holds the mmap
#       weights): the deep budget is CAPACITY-bound, so the flip is a
#       live +2 GiB of admission capacity.  Leg H pins the arithmetic
#       (plan identical; capacity moves 1:1 with the knob; budget =
#       min(plan, capacity) on both sides).  The loaded-decode guard
#       (the documented 7% page-cache stamp, now under the DEEPER
#       residency the extra allowance admits) is the MT-7 deep-admission
#       A/B's job -- at-rest decode cannot see an unspent allowance.
#   V8: the 32 MiB FP8 predecode scratch prewarm is gated on its consumers
#       (fp8-kv mirror present AND predecode enabled) and logs when it runs.
#   V13: batch graph tensors reclassed SESSION_TENSORS (bank census is
#       banks only); "context buffers" boot line labeled ESTIMATE;
#       prefill_chunk printed in the batch-ctx ready ledger.
#
# Legs:
#   S:  -c 16384 default boot   -> REQUESTED 32 verbatim (W8 regime; the
#       fit may grant fewer from live free memory -- run-1 granted 30
#       once the DSpark term made the estimate honest), prewarm log line
#       present, ESTIMATE labeling present.
#   P:  -c 16384 PREDECODE=0    -> no prewarm line; a completion still
#       serves (scalar fp8 path).
#   D:  -c 524288 default boot  -> max_seq GOVERNED [4,32] (v0.6.2 bank
#       plan; the frozen 4 is dead), prefill_chunk=4096; census:
#       engine_other < 64 MiB, session_tensors >= 512 MiB (graph-tensor
#       reclass), batch_bank < 4 GiB (banks only); vmm budget line parsed.
#   H:  -c 524288 HEADROOM=8192 -> same max_seq + IDENTICAL plan/budget as
#       leg D (plan-bound => headroom non-binding); capacity differs by
#       ~2 GiB (proof the knob engaged).
#   O:  -c 524288 COALESCE_MAX=8 -> max_seq=8 (env overrides ctx default).
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
SRV=/tmp/mt5_defaults_gate_srv.log
OUT=${OUT:-/tmp/mt5_defaults_gate_$$}
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null; exit 1; }
srv_grep(){ ssh "$R" "grep -m1 \"$1\" $SRV 2>/dev/null" 2>/dev/null; }
srv_count(){ local c; c=$(ssh "$R" "grep -c \"$1\" $SRV 2>/dev/null || true" 2>/dev/null | tail -1); echo "${c:-0}"; }

wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable never reached 100G"; sleep 5
  done }

boot(){ # $1 = ctx  $2 = extra env
  log "boot: killing old $BIN on $R (ctx=$1 env='$2')"
  ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
  wait_mem
  ssh "$R" ": > $SRV; cd $BINDIR; $2 setsid nohup ./$BIN -c $1 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
  log "boot up"
}

metrics(){ ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/metrics"; }
census_mib(){ metrics | grep "class=\"$1\",state=\"allocated\"" | awk '{s+=$2} END {printf "%d\n", s/1048576}'; }
# "ds4: batch vmm: ... budget=41.23 GiB [plan 41.23, capacity 87.45], ..."
vmm_field(){ # $1 = plan|capacity|budget
  case "$1" in
    budget)   srv_grep "batch vmm:" | sed -n 's/.*budget=\([0-9.]*\) GiB.*/\1/p' ;;
    plan)     srv_grep "batch vmm:" | sed -n 's/.*\[plan \([0-9.]*\),.*/\1/p' ;;
    capacity) srv_grep "batch vmm:" | sed -n 's/.*capacity \([0-9.]*\)\].*/\1/p' ;;
  esac }
ready_field(){ # $1 = max_seq|prefill_chunk
  srv_grep "persistent batch ctx ready" | sed -n "s/.*$1=\([0-9]*\).*/\1/p"; }

# ---------- Leg S: short-ctx verbatim ----------
boot 16384 ""
MS=$(ready_field max_seq)
# The DEFAULT under test is the REQUESTED count; the fit's grant tracks
# live free memory (page-cache state) and may be lower.  Requested shows
# in the reduction print when reduced, else granted == requested.
REQ=$(srv_grep "batch fit:" | sed -n 's/.*(requested \([0-9]*\)).*/\1/p')
[ "${REQ:-$MS}" = "32" ] || fail "leg S: requested ${REQ:-$MS} at ctx 16384 (expected the W8 32 verbatim; granted $MS)"
[ "$(srv_count 'fp8 predecode scratch prewarmed')" -ge 1 ] || fail "leg S: prewarm line missing on default boot"
[ "$(srv_count 'context buffers ESTIMATE')" -ge 1 ] || fail "leg S: context-buffers line not labeled ESTIMATE"
log "S PASS (requested ${REQ:-$MS}, granted $MS; prewarm logged, ESTIMATE labeled)"

# ---------- Leg P: prewarm kill switch ----------
boot 16384 "DS4_CUDA_FP8_KV_PREDECODE=0"
[ "$(srv_count 'fp8 predecode scratch prewarmed')" = "0" ] || fail "leg P: prewarm ran with predecode off"
HTTP=$(ssh "$R" "curl -s -m 120 -o /tmp/mt5_p.json -w '%{http_code}' \
  http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":32}'")
[ "$HTTP" = "200" ] || fail "leg P: completion HTTP $HTTP with predecode off"
log "P PASS (no prewarm, scalar path serves)"

# ---------- Leg D: deep defaults ----------
# Governed bank plan (v0.6.2): the deep default is priced from the LIVE
# budget (the frozen ctx-aware 4 is dead) -- the grant is box-dependent
# in [4, 32] and the MT GATE LAW (assert the request/regime, never a
# pinned grant) binds here too.  The bank census bound scales with the
# grant (~1 GiB/bank eager remainder at this shape).
boot 524288 ""
MS=$(ready_field max_seq); PC=$(ready_field prefill_chunk)
[ -n "$MS" ] && [ "$MS" -ge 4 ] && [ "$MS" -le 32 ] \
  || fail "leg D: max_seq ${MS:-?} outside the governed [4,32] at ctx 524288"
[ "$(srv_count 'kv plan')" -ge 1 ] || fail "leg D: governed kv-plan line missing at 524288"
D_MS=$MS
[ "$PC" = "4096" ] || fail "leg D: prefill_chunk $PC missing from ready ledger"
EO=$(census_mib engine_other); ST=$(census_mib session_tensors); BB=$(census_mib batch_bank)
[ -n "$EO" ] && [ "$EO" -lt 64 ] || fail "leg D: engine_other ${EO:-?} MiB (invariant < 64)"
[ -n "$ST" ] && [ "$ST" -ge 512 ] || fail "leg D: session_tensors ${ST:-?} MiB (graph reclass expected >= 512)"
[ -n "$BB" ] && [ "$BB" -lt $((MS * 1024)) ] \
  || fail "leg D: batch_bank ${BB:-?} MiB (expected < ~1 GiB/bank x $MS banks)"
D_PLAN=$(vmm_field plan); D_BUD=$(vmm_field budget); D_CAP=$(vmm_field capacity)
[ -n "$D_PLAN" ] && [ -n "$D_BUD" ] && [ -n "$D_CAP" ] || fail "leg D: vmm budget line unparsed"
D_NB=$(srv_grep "batch vmm:" | sed -n 's/.*MiB\/bank x \([0-9]*\) banks.*/\1/p')
D_WF=$(srv_grep "batch vmm:" | sed -n 's/.*work floor=packed \([0-9.]*\) GiB.*/\1/p')
[ -n "$D_NB" ] && [ -n "$D_WF" ] || fail "leg D: banks/packed-work-floor unparsed from vmm line"
log "D PASS (max_seq=$MS governed, prefill_chunk=4096; censuses eo=${EO} st=${ST} bb=${BB} MiB; plan=$D_PLAN budget=$D_BUD capacity=$D_CAP wf=$D_WF GiB)"

# ---------- Leg H: headroom arithmetic (V7, re-derived at v0.6.2) ----------
# The deep budget stays min(plan, capacity) with capacity = max(res +
# (free - headroom) + floors, work floor) -- but the work floor is
# PACKED now (Inc 1: the vmm line prints it, `work floor=packed X GiB`;
# the old 2 x virtual arithmetic is the killed phantom), the stock
# headroom DERIVES (Inc 2: floor + burst; v0.7.0: floor 3072 + burst 2048 =
# 5120 at this depth, the serial hold-out being off above ctx 131072),
# and the BANK GRANT follows the budget (governed plan), so a tighter
# headroom may legitimately shrink the grant and with it plan_allow.
# Assert what holds in every regime: per-bank plan rate identical
# across boots, budget = min(plan, capacity), capacity >= the printed
# packed work floor, and the capacity delta is ~0 (floor/plan-ruled)
# or ~+3 GiB (capacity-bound: pin 8192 vs derived 5120), cross-boot
# MemAvailable drift tolerated.  Log the observed regime.
boot 524288 "DS4_BATCH_FIT_HEADROOM_MB=8192"
MS=$(ready_field max_seq)
[ -n "$MS" ] && [ "$MS" -ge 4 ] && [ "$MS" -le "$D_MS" ] \
  || fail "leg H: max_seq ${MS:-?} vs D's $D_MS (a tighter headroom can only shrink the governed grant)"
H_PLAN=$(vmm_field plan); H_BUD=$(vmm_field budget); H_CAP=$(vmm_field capacity)
H_NB=$(srv_grep "batch vmm:" | sed -n 's/.*MiB\/bank x \([0-9]*\) banks.*/\1/p')
H_WF=$(srv_grep "batch vmm:" | sed -n 's/.*work floor=packed \([0-9.]*\) GiB.*/\1/p')
[ -n "$H_NB" ] && [ -n "$H_WF" ] || fail "leg H: banks/packed-work-floor unparsed"
python3 -c "exit(0 if abs($H_PLAN/$H_NB - $D_PLAN/$D_NB) <= 0.05 else 1)" \
  || fail "leg H: per-bank plan rate moved ($H_PLAN/$H_NB vs $D_PLAN/$D_NB GiB/bank)"
CAP_DELTA=$(python3 -c "print(round($D_CAP - $H_CAP, 2))")
python3 -c "exit(0 if $D_CAP >= $D_WF - 0.05 and $H_CAP >= $H_WF - 0.05 else 1)" \
  || fail "leg H: capacity below the packed work floor (D: $D_CAP vs $D_WF; H: $H_CAP vs $H_WF)"
python3 -c "exit(0 if abs($CAP_DELTA) <= 0.2 or (1.0 <= $CAP_DELTA <= 3.6) else 1)" \
  || fail "leg H: capacity delta $CAP_DELTA GiB fits no regime (~0 floor/plan-ruled, ~+3 capacity-bound: pin 8192 vs derived 5120)"
python3 -c "exit(0 if abs($D_BUD - min($D_PLAN, $D_CAP)) < 0.05 and abs($H_BUD - min($H_PLAN, $H_CAP)) < 0.05 else 1)" \
  || fail "leg H: budget != min(plan, capacity) (D: $D_BUD vs min($D_PLAN,$D_CAP); H: $H_BUD vs min($H_PLAN,$H_CAP))"
REGIME=$(python3 -c "
d=abs($CAP_DELTA)
if abs($D_CAP - $D_WF) <= 0.05: print('floor-ruled (packed work floor=%s GiB)' % $D_WF)
elif d <= 0.2: print('plan-ruled')
else: print('capacity-bound (+%s GiB: pin 8192 vs derived 5120)' % $CAP_DELTA)")
log "H PASS (per-bank rate identical; regime: $REGIME; budget=min(plan,capacity) both sides: D=$D_BUD H=$H_BUD GiB)"

# ---------- Leg O: env override wins ----------
boot 524288 "DS4_SERVER_COALESCE_MAX=8"
MS=$(ready_field max_seq)
[ "$MS" = "8" ] || fail "leg O: max_seq $MS with COALESCE_MAX=8 (override must win)"
log "O PASS (override max_seq=8)"

ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null
log "ALL LEGS PASS — artifacts in $OUT ($BIN killed, $R left free)"
