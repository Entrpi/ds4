#!/bin/bash
# gates/legs/memory/admission_credit.sh — Inc 0b: lifetime page-union
# admission credits (plan review 2.2).
#
# Every continuous admission now holds a CREDIT for its full normalized
# target min(prompt + decode budget, seq_cap) from install until the row
# ends; the admission verdicts charge the page UNION of resident bytes and
# every live credit.  Before this, only pending PROMPT targets were charged
# (`outstanding`), so a bank whose prefill landed lost its decode-growth
# commitment and later rows could promise the same headroom.
#
# Legs (all boots pin the plan: DS4_SERVER_COALESCE_MAX=4):
#   B width:   DEFAULT budget, 4 concurrent omitted-budget rows -> all four
#              admit, zero rejects (the plan budget = banks x full extent
#              covers four full-bank credits; publishes the default-384K
#              achieved width the plan requires).
#   A promise: budget pinned to 1.14x virtual/bank -> three omitted-budget
#              rows admit, the FOURTH is REJECTED AT ADMIT ('rejected on
#              comp-cache budget') and completes via the serial fallback
#              ('prompt start').  Sizing is MEASURED, not derived: at this
#              pinned shape (ctx 16384 x 4 banks, page 2 MiB, virtual
#              280 MiB/bank) the DS4_ADMIT_DEBUG verdict totals -- which
#              are timing-independent lifetime promises (resident +
#              projected credits) -- are union(1..4) = 171.3 / 213.3 /
#              297.3 / 339.3 MiB.  Neighbor banks share VMM edge pages
#              (per-layer strides ~1.4 pages), so unions sit FAR below
#              k x virtual/bank; the pin 1.14 x 280 = 319 MiB lands
#              between union(3) and union(4) with ~20 MiB margin each way.
#   C release: after every leg-A row is dead (dead-client reap), a new row
#              admits with NO new reject -- credits released.
#   C2 prefill-abort: a deep prompt killed mid-prefill releases its credit
#              (interrupted checkpoint path); a follow-up row admits clean.
#
# External live-memory loss stays memory/floor.sh's leg; this gate's boots
# assert the floor line NEVER fires (budget rejects only).
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
CTX=${CTX:-16384}
SRV=/tmp/admission_credit_gate_srv.log
OUT=${OUT:-/tmp/admission_credit_gate_$$}
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
  ssh "$R" ": > $SRV; cd $BINDIR; DS4_SERVER_COALESCE_MAX=4 $1 setsid nohup ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
  log "boot up"
}

# One omitted-budget chat request pushed in the background ON THE BOX
# (curl dies at its --max-time; the dead-client reap then ends the row).
# setsid+nohup+detached fds is MANDATORY: a plain `ssh host "curl ... &"`
# lets sshd reap the curl at session teardown before it even connects --
# the same hang/loss class as an undetached remote server boot.
push_row(){ # $1 = tag  $2 = max-time  $3 = prompt
  ssh "$R" "setsid nohup curl -s -m $2 -o /tmp/acg_$1.json http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"$3\"}]}' \
    > /dev/null 2>&1 < /dev/null & exit 0" 2>/dev/null
}

metric(){ ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/metrics" | grep "^$1 " | awk '{print $2}'; }
admits_now(){ ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/metrics" | \
  grep '^ds4_admits_total' | awk '{s+=$2} END {print s+0}'; }

# ---------- Leg B: default budget, achieved width ----------
boot ""
BOOTLINE=$(ssh "$R" "grep 'batch vmm' $SRV | head -1")
log "boot: $BOOTLINE"
VPB=$(echo "$BOOTLINE" | grep -oE 'virtual [0-9.]+ MiB/bank' | grep -oE '[0-9.]+')
[ -n "$VPB" ] || fail "no virtual MiB/bank on the boot vmm line"
A0=$(admits_now); R0=$(metric ds4_cont_admit_rejects_total)
push_row b1 25 "Tell me a very long and winding story about a lighthouse keeper."
push_row b2 25 "Tell me a very long and winding story about a desert caravan."
push_row b3 25 "Tell me a very long and winding story about a glacier pilot."
push_row b4 25 "Tell me a very long and winding story about a deep sea diver."
sleep 12
A1=$(admits_now); R1=$(metric ds4_cont_admit_rejects_total)
WIDTH=$((A1 - A0))
[ "$WIDTH" -eq 4 ] || fail "leg B width $WIDTH != 4 (default budget must cover four full-bank credits)"
[ "$R1" = "$R0" ] || fail "leg B rejects advanced ($R0 -> $R1) at the default budget"
log "B PASS (achieved width 4/4 at the default budget, zero rejects)"
sleep 20   # curls hit max-time; dead-client reap ends the rows

# ---------- Leg A: pinned budget between union(3) and union(4) ----------
BUDGET_MB=$(python3 -c "print(int(1.14 * $VPB))")
log "leg A: pinning DS4_BATCH_VMM_BUDGET_MB=$BUDGET_MB (1.14 x $VPB MiB/bank: above union(3)=297, below union(4)=339)"
boot "DS4_BATCH_VMM_BUDGET_MB=$BUDGET_MB"
REJ0=$(srv_count 'rejected on comp-cache budget')
SER0=$(srv_count 'prompt start')
FLR0=$(srv_count 'rejected on memory floor')
AA0=$(admits_now)
push_row a1 35 "Tell me a very long and winding story about a mountain guide."
sleep 2
push_row a2 35 "Tell me a very long and winding story about a river ferryman."
sleep 2
push_row a3 35 "Tell me a very long and winding story about a night watchman."
sleep 2
push_row a4 35 "Tell me a very long and winding story about a signal operator."
sleep 8
AA1=$(admits_now)
[ $((AA1 - AA0)) -ge 3 ] || fail "leg A: only $((AA1-AA0)) rows arrived/admitted (push failure, not a verdict)"
REJ1=$(srv_count 'rejected on comp-cache budget')
[ "$REJ1" -gt "$REJ0" ] || fail "leg A: no comp-cache budget reject (credits not charged?)"
# The rejected job is parked until the cont epoch drains (the admitted rows
# run to their curl timeouts), THEN reruns on the serial path -- poll for it.
SER1=$SER0
for i in $(seq 1 16); do
  SER1=$(srv_count 'prompt start')
  [ "$SER1" -gt "$SER0" ] && break
  sleep 5
done
[ "$SER1" -gt "$SER0" ] || fail "leg A: rejected row never reached the serial fallback (waited 80s)"
ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" || fail "leg A: server died"
log "A PASS (reject +$((REJ1-REJ0)) at admit, serial fallback +$((SER1-SER0)))"
sleep 15   # let the fallback row's curl die and the reap land

# ---------- Leg C: credits released after row death ----------
REJ2=$(srv_count 'rejected on comp-cache budget')
C0=$(ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/metrics" | \
  grep 'surface="openai_chat",lane="continuous"' | awk '{print $2}')
push_row c1 20 "Tell me a very long and winding story about a canal engineer."
sleep 8
REJ3=$(srv_count 'rejected on comp-cache budget')
C1=$(ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/metrics" | \
  grep 'surface="openai_chat",lane="continuous"' | awk '{print $2}')
[ "$REJ3" = "$REJ2" ] || fail "leg C: reject after every credit was released ($REJ2 -> $REJ3)"
[ -n "$C0" ] && [ -n "$C1" ] && [ "$C1" -gt "$C0" ] || fail "leg C: row did not ride cont ($C0 -> $C1)"
log "C PASS (released credits restored headroom, cont engagement $C0 -> $C1)"
sleep 22

# ---------- Leg C2: mid-prefill abort releases the credit ----------
# ~12K tokens: must FIT one bank (seq_cap = CTX) so the abort exercises the
# CONT interrupted-prefill path, not a serial fallback -- 700 entries was
# ~17K > 16384 and silently routed serial (engagement law: prove the lane).
python3 - > "$OUT/deep_prompt.txt" <<'PY'
import random
random.seed(42)
towns = ["Arles","Bergen","Cadiz","Dover","Erfurt","Faro","Ghent","Hobart"]
s = []
for i in range(480):
    s.append(f"{towns[i % 8]} logbook entry {i}: the harbor master recorded "
             f"wind from the {'north' if i % 2 else 'south'}west, two vessels "
             f"cleared customs, and the evening ferry left {3 + i % 4} minutes late.")
print(" ".join(s))
PY
python3 - "$OUT/deep_prompt.txt" > "$OUT/deep_req.json" <<'PY'
import json, sys
print(json.dumps({"messages": [{"role": "user", "content": open(sys.argv[1]).read()
                                + " Summarize every logbook entry above in detail."}]}))
PY
scp -q "$OUT/deep_req.json" "$R:/tmp/acg_deep_req.json"
ABT0=$(srv_count 'cont pending admission aborted')
ssh "$R" "curl -s -m 4 -o /dev/null http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/acg_deep_req.json" 2>/dev/null
sleep 6   # curl died at 4s (mid-prefill); reap + checkpoint land
ABT1=$(srv_count 'cont pending admission aborted')
[ "$ABT1" -gt "$ABT0" ] || fail "leg C2: the CONT mid-prefill abort never engaged ($ABT0 -> $ABT1)"
REJ4=$(srv_count 'rejected on comp-cache budget')
push_row c2 20 "Tell me a very long and winding story about a bridge painter."
sleep 8
REJ5=$(srv_count 'rejected on comp-cache budget')
[ "$REJ5" = "$REJ4" ] || fail "leg C2: reject after a mid-prefill abort ($REJ4 -> $REJ5)"
log "C2 PASS (cont mid-prefill abort +$((ABT1-ABT0)) released its credit; no reject on the follow-up)"

FLR1=$(srv_count 'rejected on memory floor')
[ "$FLR1" = "$FLR0" ] || fail "memory-floor line fired in a budget-only gate ($FLR0 -> $FLR1)"

ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null
log "ALL LEGS PASS — artifacts in $OUT ($BIN killed, $R left free)"
