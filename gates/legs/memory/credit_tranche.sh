#!/bin/bash
# gates/legs/memory/credit_tranche.sh — MT-1b (v0.6.1 memory truth): tranche
# decode credits + mid-flight extension (DS4_CONT_ADMIT_TRANCHE).
#
# The phantom this kills: every omitted-max_tokens request used to hold a
# LIFETIME credit for the server's full default decode budget (393216 tok)
# from admission until row end -- ~1.6 GiB of never-decoded future charged
# per agent-style admission (the leg-1 codex deep-bank refusal anatomy).
# With the tranche, admission credits min(prompt + min(budget, T), seq_cap)
# and the decode loop extends the credit tranche-by-tranche under the SAME
# funding verdict (ds4_batch_cont_fund_check).  An extension refusal pins
# the row to its funded boundary: it finishes there with finish=length --
# strictly better than the legacy shape, which refused the whole request.
#
# Legs (ctx 16384 x 4 banks; the omitted-budget normalized mn ~= 16.3K, so
# a pinned tranche bites without a deep boot; unions are MEASURED via
# DS4_ADMIT_DEBUG -- mres+mneed is the timing-independent lifetime union,
# the memory/admission_credit.sh law):
#   CAL:  two default-budget boots (tranche=2048 / tranche=0) x 4 concurrent
#         omitted-budget rows -> record union(4) each side; assert the
#         tranche saves >= 40 MiB (the pessimism being deleted) and that
#         the admit-debug discloses decode_budget=2048 on the tranche side.
#   T/K:  budget pinned to the midpoint of the two measured unions.
#         T (tranche=2048): all four rows admit, zero rejects.
#         K (tranche=0, the kill switch): the legacy over-credit REPRODUCES
#         the refusal ('rejected on comp-cache budget') at the same budget
#         -- the leg-1 pessimism class, at gate scale.
#   X:    extension walk (tranche=512, default budget): a long row crosses
#         its first boundary -> ds4_cont_credit_extension_granted_total
#         advances; no refusal line.
#   XR:   extension refusal (tranche=4096, budget pinned to the measured
#         single-row admission union + 4 MiB): the first boundary's
#         next-tranche pages cannot fit -> refusal line + refused counter,
#         and THE REQUEST STILL COMPLETES with finish_reason "length"
#         (the funded-boundary invariant made client-visible); a follow-up
#         row admits clean after row end (credit released).
#
# External live-memory loss stays memory/floor.sh's leg; this gate's boots
# assert the floor line NEVER fires (budget verdicts only).
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
CTX=${CTX:-16384}
SRV=/tmp/credit_tranche_gate_srv.log
OUT=${OUT:-/tmp/credit_tranche_gate_$$}
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

boot(){ # $1 = extra env (may be empty)
  log "boot: killing old $BIN on $R"
  ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
  wait_mem
  ssh "$R" ": > $SRV; cd $BINDIR; DS4_SERVER_COALESCE_MAX=4 DS4_ADMIT_DEBUG=1 $1 setsid nohup ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
  log "boot up"
}

push_row(){ # $1 = tag  $2 = max-time  $3 = prompt  (omitted max_tokens: the default-budget shape)
  ssh "$R" "setsid nohup curl -s -m $2 -o /tmp/ctg_$1.json http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"$3\"}]}' \
    > /dev/null 2>&1 < /dev/null & exit 0" 2>/dev/null
}

metric(){ ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/metrics" | grep "^$1 " | awk '{print $2}'; }
admits_now(){ ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/metrics" | \
  grep '^ds4_admits_total' | awk '{s+=$2} END {print s+0}'; }

# The Nth 'admit debug' line's mres+mneed = union(N): the timing-independent
# lifetime promise (resident + beyond-resident projection == the page union,
# invariant to how much prefill has faulted -- admission_credit_gate law).
union_at(){ # $1 = which admit-debug line (1-based)
  ssh "$R" "grep 'admit debug' $SRV | sed -n '$1p'" | \
    python3 -c "import re,sys
l = sys.stdin.read()
mn = re.search(r'mneed=([0-9.]+) MiB', l); mr = re.search(r'mres=([0-9.]+) MiB', l)
print(int(float(mn.group(1)) + float(mr.group(1))) if mn and mr else '')"
}

cal_four(){ # $1 = leg name; pushes 4 staggered omitted-budget rows, echoes union(4)
  local a0 a1
  a0=$(admits_now)
  push_row "${1}1" 35 "Tell me a very long and winding story about a lighthouse keeper."
  sleep 2
  push_row "${1}2" 35 "Tell me a very long and winding story about a desert caravan."
  sleep 2
  push_row "${1}3" 35 "Tell me a very long and winding story about a glacier pilot."
  sleep 2
  push_row "${1}4" 35 "Tell me a very long and winding story about a deep sea diver."
  sleep 8
  a1=$(admits_now)
  [ $((a1 - a0)) -eq 4 ] || fail "cal $1: only $((a1-a0))/4 rows admitted"
  union_at 4
}

# ---------- CAL-T: tranche unions at the default budget ----------
boot "DS4_CONT_ADMIT_TRANCHE=2048"
RT0=$(metric ds4_cont_admit_rejects_total)
UT=$(cal_four t)
[ -n "$UT" ] || fail "cal-T: no union from the 4th admit-debug line"
RT1=$(metric ds4_cont_admit_rejects_total)
[ "$RT1" = "$RT0" ] || fail "cal-T: rejects at the default budget"
DB=$(srv_count 'decode_budget=2048')
[ "$DB" -ge 4 ] || fail "cal-T: admit debug does not disclose the tranche (decode_budget=2048 x$DB)"
log "CAL-T union(4 rows @ tranche 2048) = $UT MiB (disclosure x$DB)"
sleep 22   # curls die; reap releases the credits

# ---------- CAL-K: legacy unions at the default budget ----------
boot "DS4_CONT_ADMIT_TRANCHE=0"
UK=$(cal_four k)
[ -n "$UK" ] || fail "cal-K: no union from the 4th admit-debug line"
log "CAL-K union(4 rows @ legacy full credit) = $UK MiB"
SAVED=$((UK - UT))
[ "$SAVED" -ge 40 ] || fail "tranche saves only $SAVED MiB (UT=$UT UK=$UK; expected >= 40)"
PIN=$(( (UT + UK) / 2 ))
log "pessimism deleted per 4-row shape: $SAVED MiB; pinning budget $PIN MiB"
sleep 22

# ---------- Leg T: tranche admits the width the pin refuses legacy ----------
boot "DS4_CONT_ADMIT_TRANCHE=2048 DS4_BATCH_VMM_BUDGET_MB=$PIN"
FLR0=$(srv_count 'rejected on memory floor')
REJ0=$(srv_count 'rejected on comp-cache budget')
AT0=$(admits_now)
push_row T1 35 "Tell me a very long and winding story about a mountain guide."
sleep 2
push_row T2 35 "Tell me a very long and winding story about a river ferryman."
sleep 2
push_row T3 35 "Tell me a very long and winding story about a night watchman."
sleep 2
push_row T4 35 "Tell me a very long and winding story about a signal operator."
sleep 8
AT1=$(admits_now)
REJ1=$(srv_count 'rejected on comp-cache budget')
[ $((AT1 - AT0)) -eq 4 ] || fail "leg T: $((AT1-AT0))/4 admitted at the pinned budget (tranche should fit all four)"
[ "$REJ1" = "$REJ0" ] || fail "leg T: comp-cache reject with the tranche on ($REJ0 -> $REJ1)"
log "T PASS (4/4 admitted at the $PIN MiB pin, zero rejects)"
sleep 22

# ---------- Leg K: the kill switch reproduces the legacy refusal ----------
boot "DS4_CONT_ADMIT_TRANCHE=0 DS4_BATCH_VMM_BUDGET_MB=$PIN"
REJ2=$(srv_count 'rejected on comp-cache budget')
SER0=$(srv_count 'prompt start')
AK0=$(admits_now)
push_row K1 35 "Tell me a very long and winding story about a canal engineer."
sleep 2
push_row K2 35 "Tell me a very long and winding story about a bridge painter."
sleep 2
push_row K3 35 "Tell me a very long and winding story about a harbor master."
sleep 2
push_row K4 35 "Tell me a very long and winding story about a train conductor."
sleep 8
AK1=$(admits_now)
REJ3=$(srv_count 'rejected on comp-cache budget')
# MT-7: arrivals = admits + budget rejects (each push lands as exactly one).
# The admission band (+~2% on legacy's phantom-inflated unions) moved the
# refusal boundary at the midpoint pin from after-3 to after-2 rows, so the
# oracle asserts the SHAPE -- pushes all arrived, some admitted, some
# refused -- never an exact admit count at a mid-band pin.
ARR=$((AK1 - AK0 + REJ3 - REJ2))
[ $ARR -ge 4 ] || fail "leg K: only $ARR pushes arrived as verdicts (admits+rejects; push failure, not a verdict)"
[ $((AK1 - AK0)) -ge 2 ] || fail "leg K: only $((AK1-AK0)) admits before refusal (pin implausibly tight)"
[ "$REJ3" -gt "$REJ2" ] || fail "leg K: legacy full credits did NOT refuse at the pin (tranche A/B broken)"
SER1=$SER0
for i in $(seq 1 16); do
  SER1=$(srv_count 'prompt start')
  [ "$SER1" -gt "$SER0" ] && break
  sleep 5
done
[ "$SER1" -gt "$SER0" ] || fail "leg K: rejected row never reached the serial fallback (waited 80s)"
log "K PASS (legacy refusal reproduced: reject +$((REJ3-REJ2)), serial fallback +$((SER1-SER0)))"
[ "$(srv_count 'rejected on memory floor')" = "0" ] || fail "leg K: floor line fired in a budget-only gate"
sleep 15

# ---------- Leg X: the extension walk (grants, no refusal) ----------
boot "DS4_CONT_ADMIT_TRANCHE=512"
G0=$(metric ds4_cont_credit_extension_granted_total)
push_row X1 120 "Count upward from 1, one number per line, without stopping and without any commentary."
G1=$G0
for i in $(seq 1 20); do
  G1=$(metric ds4_cont_credit_extension_granted_total)
  [ -n "$G1" ] && [ "$G1" -gt "${G0:-0}" ] && break
  sleep 5
done
[ -n "$G1" ] && [ "$G1" -gt "${G0:-0}" ] || fail "leg X: no extension granted within 100s (tranche 512)"
XRF=$(srv_count 'credit-extension refused')
[ "$XRF" = "0" ] || fail "leg X: refusal at the default budget ($XRF)"
log "X PASS (extensions granted: $G0 -> $G1, zero refusals)"
ssh "$R" "pkill -x curl" 2>/dev/null
sleep 15

# ---------- Leg XR: extension refusal = finish_reason length ----------
# Pin: the single-row admission union + 4 MiB -- admission fits, the first
# boundary's next-tranche pages (~16 MiB expected at 4096 tok across the
# layer slabs) do not.  An unlucky page phase grants once; cumulative
# growth guarantees refusal by the following boundary -- poll generously.
# MT-7: both XR boots pin the admission band to 1024 (physics-exact) -- the
# pin derives from the QUOTE while resident tops out at the unbanded page
# extent, so a banded quote inflates the pin ~2 MiB above what the verdict
# ever reaches and starves the refusal margin at page granularity.  This
# leg tests the extension-refusal machinery; the band has its own units +
# deep-gate asserts.
boot "DS4_CONT_ADMIT_TRANCHE=4096 DS4_CONT_ADMIT_BAND_X1024=1024"
push_row P1 20 "Count upward from 1, one number per line, without stopping and without any commentary."
sleep 8
U1=$(union_at 1)
[ -n "$U1" ] || fail "leg XR cal: no single-row union"
ssh "$R" "pkill -x curl" 2>/dev/null
XPIN=$((U1 + 4))
log "leg XR: single-row union $U1 MiB -> pinning budget $XPIN MiB (tranche 4096)"
boot "DS4_CONT_ADMIT_TRANCHE=4096 DS4_CONT_ADMIT_BAND_X1024=1024 DS4_BATCH_VMM_BUDGET_MB=$XPIN"
RF0=$(metric ds4_cont_credit_extension_refused_total)
ssh "$R" "rm -f /tmp/ctg_XR.json"
push_row XR 600 "Count upward from 1, one number per line, without stopping and without any commentary."
RF1=$RF0
for i in $(seq 1 50); do
  RF1=$(metric ds4_cont_credit_extension_refused_total)
  [ -n "$RF1" ] && [ "$RF1" -gt "${RF0:-0}" ] && break
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "leg XR: server died mid-decode"
  sleep 10
done
[ -n "$RF1" ] && [ "$RF1" -gt "${RF0:-0}" ] || fail "leg XR: no extension refusal within 500s at the $XPIN MiB pin"
[ "$(srv_count 'credit-extension refused')" -ge 1 ] || fail "leg XR: refused counter without the refusal log line"
# The row must now COMPLETE at its funded boundary -- poll the response.
FIN=""
for i in $(seq 1 24); do
  FIN=$(ssh "$R" "python3 -c \"import json;print(json.load(open('/tmp/ctg_XR.json'))['choices'][0]['finish_reason'])\" 2>/dev/null")
  [ -n "$FIN" ] && break
  sleep 5
done
[ "$FIN" = "length" ] || fail "leg XR: finish_reason '$FIN' != 'length' (row must finish AT the funded boundary)"
# Credit released at row end: a follow-up admits with no new reject.
REJ6=$(srv_count 'rejected on comp-cache budget')
push_row XR2 20 "Tell me a very long and winding story about a tram driver."
sleep 8
REJ7=$(srv_count 'rejected on comp-cache budget')
[ "$REJ7" = "$REJ6" ] || fail "leg XR: follow-up rejected after row end ($REJ6 -> $REJ7): credit not released"
log "XR PASS (refused +$((RF1-RF0)), finish_reason=length at the funded boundary, follow-up admitted)"

FLR1=$(srv_count 'rejected on memory floor')
[ "$FLR1" = "${FLR0:-0}" ] || fail "memory-floor line fired in a budget-only gate"

ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null
log "ALL LEGS PASS — artifacts in $OUT ($BIN killed, $R left free)"
