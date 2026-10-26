#!/bin/bash
# battery: real_budgets tier=2b stop=0 server=33 driver=mac trigger="admission/census/ledger/serial-lane changed" est=35 seq=10
# verdict: V062 GATE: (ALL LEGS PASS|FAILED)
# catches: the boot bank plan and its disclosures end to end: budget = min(plan, capacity), the derived headroom on a stripped floor, kill switches, packed-vs-virtual floor A/B at 786432, strict reconcile serve, two-tenant coexist + named trim victim (driven FROM the workstation: it syncs and rebuilds the box)
# cmd: ssh $RB_SERVER "rm -rf $RB_LOCK_DIR"; R=$RB_SERVER OUT="$RB_OUT/v062" bash gates/legs/memory/real_budgets.sh; rc=$?; ssh $RB_SERVER "mkdir -p $RB_LOCK_DIR && echo \"$RB_NONCE\" > $RB_LOCK_DIR/nonce"; exit $rc
# gates/legs/memory/real_budgets.sh — the v0.6.2 "real budgets" arc gate
# (plan: local/docs/v062/plan-v062-real-budgets.md, Inc 0-3).
#
# What shipped and what this gate pins:
#   Inc 0  mem reconcile line: ledger + named one-time charges vs the raw
#          free-memory delta since boot settle; idle-tick prints, on-demand
#          /v1/stats + /metrics, DS4_MEM_RECONCILE_STRICT gate token, the
#          first-admit warmup one-shot, serial estimate rows re-based to
#          the alloc's measured census delta.
#   Inc 1  the anti-thrash budget floor re-denominated PACKED (2 x
#          rate_phys x seq_cap x band + max_seq x floor_pb) -- the 4.6x
#          virtual phantom killed; work-floor token + floored-line
#          disclosures; DS4_BATCH_VMM_FLOOR_PACKED=0 restores virtual.
#   Inc 2  fit headroom derives (live floor + DS4_BATCH_FIT_BURST_MB 2048;
#          shipped-default parity: v0.6.2 4096+2048 = the old static 6144,
#          v0.7.0 3072+2048 = 5120 above ctx 131072 (below it a 2048 serial
#          hold-out joins the sum: gates/legs/memory/soak.sh sees that shape));
#          DS4_BATCH_FIT_HEADROOM_MB pins, _DERIVED=0 restores static.
#   Inc 3  trim victims: invalid > oldest activity > shortest history
#          (DS4_BATCH_TRIM_VICTIM=hist restores), per-victim named log
#          lines, boot-line order disclosure.
#
# Legs (each proves ENGAGEMENT, never just absence of failure):
#   U   box unit suite (./ds4_test --server) on the fresh build.
#   H1  -c 262144 stock: the four boot disclosures on one boot --
#       headroom parity line (floor 3.00 + boot-burst 2.00 = 5.00),
#       work floor=packed token, trim victim order=recency blend,
#       mem reconcile baseline armed; budget == min(plan, capacity)
#       recheck from the printed inputs.
#   H2  -c 262144 DS4_MEM_FLOOR_GB=1: derivation moves with the floor
#       (1.00 + 2.00 = 3.00 -- the exact line is THE oracle) and printed
#       capacity gains ~3 GiB vs H1.  BRACKET LAW: the cross-boot delta
#       is corroboration only -- MemAvailable drifts ~1 GiB between
#       boots (runs 2/3 measured +2.31/+1.89), so the band is LOOSE
#       ([1.2, 4.8]) and the measured value is recorded as the receipt.
#   H3  -c 32768 DERIVED=0 + TRIM_VICTIM=hist: NO derived headroom line
#       (static kill switch) AND trim victim order=hist disclosed.
#   H4  -c 32768 HEADROOM_MB=8192: NO derived line (explicit pin wins).
#   P0  -c 786432 stock: work floor=packed at depth, kv plan present,
#       budget==min recheck; the printed capacity/floors derive the
#       box's free-at-sample for P1/P2's adaptive headroom pin (the
#       probe-config law: the leg must REACH the floored regime, not
#       measure the idle default).
#   P1  -c 786432 headroom pinned so sparable ~4.5 GiB: capacity is
#       floored -- "budget floored to two packed banks" line, budget ==
#       the printed packed work floor (the c3 guarantee, packed terms).
#   P2  P1 + DS4_BATCH_VMM_FLOOR_PACKED=0: "floored to two virtual
#       banks", virtual/packed floor ratio >= 2.5 -- THE phantom-kill
#       receipt at the dcl_run2 shape.
#   R   -c 262144 DS4_MEM_RECONCILE_STRICT=1: serve one cont (temp-0
#       buffered) + one serial (temp>0; lazy graph alloc fires the
#       estimate-reconcile line with lease basis = measured); after the
#       idle tick: first-admit warmup captured, mem reconcile line
#       printed, NO STRICT line (post-capture residual under tol),
#       /v1/stats reconcile supported+captured, /metrics residual gauge;
#       graceful TERM prints "mem reconcile final:".
#   C3  -c 65536 COALESCE_MAX=4 BUDGET_MB=768 (deterministic-reject
#       lever): tenants A,B coexist with ZERO trims -- two packed
#       working sets never thrash at a budget that holds two but not
#       three; tenant C forces the trim: per-victim named line +
#       summary line + C serves + a warm record survives (ONE trim
#       funds C).  Sizing is EMPIRICALLY BRACKETED (runs 5+6): 3300
#       driver sentences tokenize to 61.7k tokens (18.7/sentence);
#       at 512 the second tenant already forced a trim (union(2) >
#       512) and one 210 MiB victim funded it (union(2) <= 670); at
#       960 the third admitted trim-free (union(3) <= 960, so
#       union(3) > 802 given union(2) > 512).  768 sits inside the
#       window on every bound: union(2) <= 670 < 768 (two sets never
#       thrash), 768 < 802 < union(3) (the third always forces the
#       trim), post-trim union == union(2) <= 768 (one victim
#       suffices).  All three tenants identical -- a larger C would
#       cross the ctx ceiling and draw the typed 400 for an oversized
#       PROMPT, never an admission.  DS4_ADMIT_DEBUG=1 puts the exact
#       projections in the receipt.  Chain with P1: the floored
#       budget IS two packed sets + floors, and two packed sets are
#       proven thrash-free.
#
# Runs FROM the Mac over SSH.  Takes /tmp/ds4_box_lock on the box (the
# two-session law) and releases it at exit.  End state: server killed,
# box lock freed.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

# LAUNCH LAW (ssh-daemonize): plain ssh + setsid --fork + a local alarm;
# never `& echo $!`, never ServerAlive opts on a launching call.
SSH(){ perl -e 'alarm 120; exec @ARGV or die' ssh "$@"; }
SSH_LONG(){ perl -e 'alarm 2700; exec @ARGV or die' ssh "$@"; }

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
SRV=/tmp/v062_gate_srv.log
OUT=${OUT:-/tmp/v062_gate_$$}
TS=$(date +%s); NONCE="v062_${TS}_$$"
mkdir -p "$OUT"
DRV="$OUT/driver.log"

log(){ echo "[$(date +%H:%M:%S)] V062: $*" | tee -a "$DRV"; }

LOCKDIR=/tmp/v062_gate.lockdir
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "another v062_budgets_gate instance is running ($LOCKDIR, pid $(cat "$LOCKDIR/pid" 2>/dev/null || echo '?'))"; exit 3
fi
echo $$ > "$LOCKDIR/pid"

BOX_LOCKED=0
full_exit(){
  rc=$?
  if [ "$BOX_LOCKED" = 1 ]; then
    SSH "$R" "pkill -x ${BIN:0:15} 2>/dev/null; exit 0" 2>/dev/null
    SSH "$R" "rm -rf /tmp/ds4_box_lock; exit 0" 2>/dev/null
    log "cleanup: server down, box lock freed; logs in $OUT"
  else
    log "cleanup: box untouched (lock was never ours); logs in $OUT"
  fi
  rm -rf "$LOCKDIR"
  exit $rc
}
trap full_exit EXIT INT TERM
fail(){ log "FAIL: $*"; SSH "$R" "tail -8 $SRV 2>/dev/null; exit 0" 2>/dev/null | tee -a "$DRV"; log "V062 GATE: FAILED"; exit 1; }

# ---- box ownership (two-session law) ----
OWN=$(SSH "$R" "cat /tmp/ds4_box_lock/nonce 2>/dev/null; exit 0" 2>/dev/null)
[ -n "$OWN" ] && fail "box lock held by another session ($OWN) -- refusing"
SSH "$R" "mkdir /tmp/ds4_box_lock && echo '$NONCE' > /tmp/ds4_box_lock/nonce" \
  || fail "could not take the box lock (raced another session?)"
BOX_LOCKED=1
GOT=$(SSH "$R" "cat /tmp/ds4_box_lock/nonce 2>/dev/null; exit 0" 2>/dev/null)
[ "$GOT" = "$NONCE" ] || fail "box lock nonce mismatch after take ($GOT)"
log "box lock taken ($NONCE)"

wait_mem(){ local n=0 got=0
  while :; do
    got=$(SSH "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable ${got:-?}G never reached 100G"; sleep 5
  done }

boot(){ # $1 = ctx  $2 = extra env (may be empty)
  log "boot: ctx=$1 env='${2:-}'"
  SSH "$R" "pkill -x ${BIN:0:15} 2>/dev/null; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
  wait_mem
  SSH "$R" ": > $SRV; cd $BINDIR && ${2:-} setsid --fork nohup ./$BIN -c $1 --port $PORT > $SRV 2>&1 < /dev/null; exit 0"
  local n=0
  until SSH "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    SSH "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(SSH "$R" "tail -2 $SRV; exit 0" 2>/dev/null | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout (ctx=$1)"
  done
  log "boot up (ctx=$1)"
}

srv_grep(){ SSH "$R" "grep -m1 \"$1\" $SRV 2>/dev/null; exit 0" 2>/dev/null; }
srv_count(){ local c; c=$(SSH "$R" "grep -c \"$1\" $SRV 2>/dev/null; exit 0" 2>/dev/null | tail -1); echo "${c:-0}"; }
# NOTE: anchored to the VALUE line -- a bare substring grep matches the
# '# TYPE' header first and parses nothing (run-7 lesson).
metric(){ SSH "$R" "curl -s -m 10 http://127.0.0.1:$PORT/metrics | grep -E '^$1[ {]' | head -1 | grep -oE '[0-9.-]+\$'; exit 0" 2>/dev/null; }
save_ledger(){ # $1 = tag
  SSH "$R" "grep -E 'batch fit|batch vmm|persistent batch ctx ready|kv plan|mem reconcile' $SRV 2>/dev/null; exit 0" 2>/dev/null > "$OUT/ledger_$1.txt"
  cat "$OUT/ledger_$1.txt" | tee -a "$DRV"
}
faults_zero(){ # $1 = leg tag
  local cf gf
  cf=$(metric ds4_memory_census_faults_total); gf=$(metric ds4_memory_governor_faults_total)
  [ "${cf:-1}" = "0" ] || fail "$1: census faults ${cf:-unparsed}"
  [ "${gf:-1}" = "0" ] || fail "$1: governor faults ${gf:-unparsed}"
  [ "$(srv_count 'illegal')" = "0" ] || fail "$1: illegal-access lines"
  [ "$(srv_count 'continuous batch failed')" = "0" ] || fail "$1: continuous batch failures"
}
# vmm-line field parsers.  The boot line prints `budget=X GiB [plan Y,
# capacity Z]` -- plan/capacity carry NO unit suffix (run-1 lesson).
vmm_line(){ srv_grep "batch vmm: comp/index slabs demand-mapped"; }
vmm_budget(){ echo "$1" | sed -n 's/.*budget=\([0-9.]*\) GiB.*/\1/p'; }
vmm_plan(){ echo "$1" | sed -n 's/.*\[plan \([0-9.]*\),.*/\1/p'; }
vmm_capacity(){ echo "$1" | sed -n 's/.*capacity \([0-9.]*\)\].*/\1/p'; }

# ---------- build + Leg U ----------
log "sync: git ls-files -> $R:$BINDIR (rsync --files-from, no --delete)"
git ls-files > "$OUT/files.txt" || fail "git ls-files failed (run from the repo root)"
rsync -a --files-from="$OUT/files.txt" . "$R:$BINDIR/" || fail "rsync failed"
log "build: make -j6 cuda-spark + ds4_test on $R"
SSH_LONG "$R" "cd $BINDIR && make -j6 cuda-spark ds4_test" > "$OUT/build.log" 2>&1 \
  || fail "box build failed: $(tail -3 "$OUT/build.log" | tr '\n' ' ')"
SASS=$(SSH "$R" "/usr/local/cuda/bin/cuobjdump --list-elf $BINDIR/$BIN 2>/dev/null | head -3; exit 0" 2>/dev/null)
case "$SASS" in *sm_121*) : ;; *) fail "sm_121 not found (empty = could not verify)";; esac
SSH_LONG "$R" "cd $BINDIR && ./ds4_test --server" > "$OUT/units.log" 2>&1
grep -q "ds4 tests: ok" "$OUT/units.log" || fail "leg U: box unit suite failed: $(tail -3 "$OUT/units.log" | tr '\n' ' ')"
log "U PASS (build sm_121 + box unit suite ok)"

# The single-shot buffered-chat driver (temp 0 = cont lane; a nonzero
# TEMP routes the request onto the serial lane -- R leg's graph alloc).
put_driver(){
  SSH "$R" "cat > /tmp/v062_driver.py" <<'EOF'
import http.client, json, sys, time
PORT, SENT, MAXTOK = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
NONCE, TEMP = sys.argv[4], float(sys.argv[5])
p = ('Summarize the flood log. ' +
     ' '.join('the tidal basin at station %s-%d filled ahead of the model forecast'
              % (NONCE[-6:], i) for i in range(SENT)))
t0 = time.time()
c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=1800)
c.request('POST', '/v1/chat/completions',
          json.dumps({'messages':[{'role':'user','content':p}],
                      'max_tokens': MAXTOK, 'temperature': TEMP}),
          {'Content-Type':'application/json'})
r = c.getresponse(); data = r.read()
if r.status != 200:
    print('%s FAIL http=%d body=%s' % (NONCE, r.status, data[:160]), flush=True); sys.exit(1)
j = json.loads(data)
content = j['choices'][0]['message']['content'] or ''
if not content:
    print('%s FAIL empty-content' % NONCE, flush=True); sys.exit(1)
u = j.get('usage', {}) or {}
print('%s OK tokens=%s wall_ms=%.0f' %
      (NONCE, u.get('completion_tokens'), (time.time()-t0)*1000.0), flush=True)
EOF
}
drive_one(){ # $1 = tag  $2 = sentences  $3 = max_tokens  $4 = temperature
  local legnonce="${NONCE}_$1"
  log "$1: driving 1 request (sent=$2 max_tokens=$3 temp=$4, nonce $legnonce)"
  SSH_LONG "$R" "cd /tmp && timeout 2400 python3 v062_driver.py $PORT $2 $3 $legnonce $4" \
      > "$OUT/drive_$1.log" 2>&1
  cat "$OUT/drive_$1.log" | tee -a "$DRV"
  grep -q "$legnonce OK" "$OUT/drive_$1.log" || fail "$1: request did not serve"
}

# ---------- Leg H1: shipped-default disclosures + arithmetic ----------
boot 262144 ""
save_ledger H1
HR=$(srv_grep "batch fit headroom: floor")
echo "$HR" | grep -q "floor 3.00 GiB + boot-burst 2.00 GiB = 5.00 GiB (derived" \
  || fail "leg H1: derived-headroom parity line missing/wrong (v0.7.0: floor 3 + burst 2, no serial term above ctx 131072): $HR"
H1_HR=$(echo "$HR" | grep -oE '= [0-9.]+ GiB' | grep -oE '[0-9.]+')
V1=$(vmm_line)
echo "$V1" | grep -q "work floor=packed" || fail "leg H1: work floor=packed token missing: $V1"
srv_grep "trim victim order=recency blend" | grep -q . || fail "leg H1: victim-order disclosure missing"
srv_grep "mem reconcile: baseline armed" | grep -q . || fail "leg H1: reconcile baseline not armed"
BUD=$(vmm_budget "$V1"); PLAN=$(vmm_plan "$V1")
CAP=$(vmm_capacity "$V1")
[ -n "$BUD" ] && [ -n "$PLAN" ] && [ -n "$CAP" ] || fail "leg H1: vmm line unparsable"
python3 -c "
b,p,c=float('$BUD'),float('$PLAN'),float('$CAP')
import sys; sys.exit(0 if abs(b-min(p,c))<=0.02 else 1)" \
  || fail "leg H1: budget $BUD != min(plan $PLAN, capacity $CAP)"
H1_CAP=$CAP
log "H1 PASS (parity 5.00 derived; packed floor token; victim order + baseline disclosed; budget=min($PLAN,$CAP))"

# ---------- Leg H2: stripped-floor derivation ----------
boot 262144 "DS4_MEM_FLOOR_GB=1"
save_ledger H2
HR=$(srv_grep "batch fit headroom: floor")
echo "$HR" | grep -q "floor 1.00 GiB + boot-burst 2.00 GiB = 3.00 GiB (derived" \
  || fail "leg H2: stripped derivation line missing/wrong: $HR"
H2_HR=$(echo "$HR" | grep -oE '= [0-9.]+ GiB' | grep -oE '[0-9.]+')
V2=$(vmm_line); CAP2=$(vmm_capacity "$V2")
[ -n "$CAP2" ] || fail "leg H2: capacity unparsable"
# The corroboration band is centred on the derivation's OWN prediction (H1 headroom - H2 headroom: 3.00 GiB on
# the 4 GiB floor, 2.00 GiB since the v0.7.0 3 GiB default) with the same +-1.8 GiB looseness for cross-boot
# MemAvailable drift (measured -0.9 to -1.1 GiB on the second boot: +2.14 vs 3 on 09-04, +0.88 vs 2 on 09-05).
EXP=$(python3 -c "print(float('$H1_HR')-float('$H2_HR'))")
python3 -c "
d=float('$CAP2')-float('$H1_CAP'); e=float('$EXP')
import sys; sys.exit(0 if e-1.8<=d<=e+1.8 else 1)" \
  || fail "leg H2: capacity delta $(python3 -c "print(float('$CAP2')-float('$H1_CAP'))") GiB outside the loose [$(python3 -c "print(float('$EXP')-1.8)"),$(python3 -c "print(float('$EXP')+1.8)")] band around the derived $EXP GiB"
log "H2 PASS (floor 1 -> headroom $H2_HR; capacity +$(python3 -c "print(f'{float(\"$CAP2\")-float(\"$H1_CAP\"):.2f}')") GiB vs H1, derivation predicts $EXP)"

# ---------- Leg H3: static kill switch + hist victim order ----------
boot 32768 "DS4_BATCH_FIT_HEADROOM_DERIVED=0 DS4_BATCH_TRIM_VICTIM=hist"
save_ledger H3
[ "$(srv_count 'batch fit headroom: floor')" = "0" ] \
  || fail "leg H3: derived line printed under DERIVED=0"
srv_grep "trim victim order=hist" | grep -q . || fail "leg H3: hist victim order not disclosed"
log "H3 PASS (static headroom silent; hist order disclosed)"

# ---------- Leg H4: explicit pin wins ----------
boot 32768 "DS4_BATCH_FIT_HEADROOM_MB=8192"
save_ledger H4
[ "$(srv_count 'batch fit headroom: floor')" = "0" ] \
  || fail "leg H4: derived line printed under an explicit HEADROOM_MB pin"
log "H4 PASS (pin wins, derivation silent)"

# ---------- Leg P0: deep stock + adaptive pin derivation ----------
boot 786432 ""
save_ledger P0
V0=$(vmm_line)
echo "$V0" | grep -q "work floor=packed" || fail "leg P0: work floor=packed missing at depth: $V0"
[ "$(srv_count 'kv plan')" -ge 1 ] || fail "leg P0: kv-plan line missing at 786432"
BUD=$(vmm_budget "$V0"); PLAN=$(vmm_plan "$V0"); CAP=$(vmm_capacity "$V0")
python3 -c "
b,p,c=float('$BUD'),float('$PLAN'),float('$CAP')
import sys; sys.exit(0 if abs(b-min(p,c))<=0.02 else 1)" \
  || fail "leg P0: budget $BUD != min(plan $PLAN, capacity $CAP)"
FPB=$(echo "$V0" | sed -n 's/.*floor \([0-9.]*\) MiB\/bank.*/\1/p')
NB=$(echo "$V0" | sed -n 's/.*MiB\/bank x \([0-9]*\) banks.*/\1/p')
[ -n "$FPB" ] && [ -n "$NB" ] || fail "leg P0: floor/banks unparsable from vmm line"
# free-at-sample ~= capacity - floors + headroom(6 GiB); pin headroom so
# sparable ~4.5 GiB: enough for the eager fit, below the packed floor.
PIN_MB=$(python3 -c "
cap=float('$CAP'); floors=float('$FPB')*int('$NB')/1024.0
vfree=cap-floors+6.0
print(max(int((vfree-4.5)*1024), 1024))")
log "P0 PASS (budget=min ok; packed floor at depth; floors=${FPB}MiBx${NB}; adaptive pin ${PIN_MB} MiB)"

# ---------- Leg P1: the floored regime, packed ----------
boot 786432 "DS4_BATCH_FIT_HEADROOM_MB=$PIN_MB"
save_ledger P1
FL=$(srv_grep "budget floored to two packed banks")
[ -n "$FL" ] || fail "leg P1: packed floored-line missing (pin ${PIN_MB} MiB did not reach the floor regime -- probe-config law)"
VP=$(vmm_line)
WFP=$(echo "$VP" | sed -n 's/.*work floor=packed \([0-9.]*\) GiB.*/\1/p')
BUDP=$(vmm_budget "$VP")
[ -n "$WFP" ] && [ -n "$BUDP" ] || fail "leg P1: packed floor/budget unparsable: $VP"
python3 -c "
import sys; sys.exit(0 if abs(float('$BUDP')-float('$WFP'))<=0.05 else 1)" \
  || fail "leg P1: floored budget $BUDP != packed work floor $WFP"
log "P1 PASS (floored: budget $BUDP == packed work floor $WFP GiB)"

# ---------- Leg P2: the same regime, virtual (the phantom) ----------
boot 786432 "DS4_BATCH_FIT_HEADROOM_MB=$PIN_MB DS4_BATCH_VMM_FLOOR_PACKED=0"
save_ledger P2
FL=$(srv_grep "budget floored to two virtual banks")
[ -n "$FL" ] || fail "leg P2: virtual floored-line missing under FLOOR_PACKED=0"
VV=$(vmm_line)
WFV=$(echo "$VV" | sed -n 's/.*work floor=virtual \([0-9.]*\) GiB.*/\1/p')
[ -n "$WFV" ] || fail "leg P2: virtual work floor unparsable: $VV"
python3 -c "
import sys; r=float('$WFV')/max(float('$WFP'),0.01)
sys.exit(0 if r>=2.5 else 1)" \
  || fail "leg P2: virtual floor $WFV vs packed $WFP -- ratio < 2.5, phantom not demonstrated"
log "P2 PASS (virtual floor $WFV GiB vs packed $WFP GiB: the phantom, on receipt)"

# ---------- Leg R: reconcile line end to end (STRICT) ----------
boot 262144 "DS4_MEM_RECONCILE_STRICT=1"
save_ledger Rboot
srv_grep "mem reconcile: baseline armed" | grep -q . || fail "leg R: baseline not armed"
put_driver
drive_one Rcont 60 48 0        # cont lane (buffered temp-0)
drive_one Rserial 40 32 0.7    # serial lane -> lazy graph alloc
EST=$(srv_grep "serial graph estimate reconcile: est=")
[ -n "$EST" ] || fail "leg R: serial estimate-reconcile line missing"
echo "$EST" | grep -q "lease basis = measured" || fail "leg R: lease basis not measured: $EST"
log "R: estimate row: $EST"
sleep 35                        # >= 3 idle ticks: warmup capture + print
[ "$(srv_count 'mem reconcile: first-admit warmup')" -ge 1 ] \
  || fail "leg R: first-admit warmup was not captured at idle"
[ "$(srv_count 'mem reconcile: drop')" -ge 1 ] \
  || fail "leg R: no reconcile line printed at idle"
[ "$(srv_count 'mem reconcile STRICT')" = "0" ] \
  || fail "leg R: STRICT residual over tolerance: $(srv_grep 'mem reconcile STRICT')"
STATS=$(SSH "$R" "curl -s -m 10 -H 'Accept: application/json' http://127.0.0.1:$PORT/v1/stats; exit 0" 2>/dev/null)
echo "$STATS" | grep -q '"reconcile":{"supported":true' || fail "leg R: stats reconcile object missing"
echo "$STATS" | grep -q '"warmup":"captured"' || fail "leg R: stats warmup not captured"
RESID=$(metric ds4_mem_reconcile_residual_bytes)
[ -n "$RESID" ] || fail "leg R: /metrics residual gauge missing"
faults_zero "leg R"
# graceful stop -> the forced final line
SSH "$R" "pkill -x ${BIN:0:15} 2>/dev/null; exit 0"
N=0
while SSH "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null; do
  sleep 3; N=$((N+3)); [ $N -ge 120 ] && fail "leg R: server did not stop in 120 s"
done
[ "$(srv_count 'mem reconcile final:')" -ge 1 ] \
  || fail "leg R: final reconcile line missing after graceful stop"
save_ledger Rfinal
log "R PASS (estimate measured; warmup captured; STRICT clean; residual gauge $RESID; final line printed)"

# ---------- Leg C3: two packed working sets never thrash + named victims ----------
boot 65536 "DS4_SERVER_COALESCE_MAX=4 DS4_BATCH_VMM_BUDGET_MB=768 DS4_ADMIT_DEBUG=1"
save_ledger C3
put_driver
drive_one C3A 3300 48 0        # tenant A ~61.7k tokens (measured)
drive_one C3B 3300 48 0        # tenant B ~61.7k tokens
[ "$(srv_count 'trim victim bank=')" = "0" ] \
  || fail "leg C3: trim fired with only two working sets resident (thrash)"
drive_one C3C 3300 48 0        # tenant C: the third set forces the trim
[ "$(srv_count 'trim victim bank=')" -ge 1 ] \
  || fail "leg C3: no named victim line after the forcing tenant"
[ "$(srv_count 'trimmed [0-9]* bank')" -ge 1 ] \
  || fail "leg C3: trim summary line missing"
WREC=$(metric ds4_warm_records)
[ -n "$WREC" ] && [ "$WREC" -ge 1 ] || fail "leg C3: no warm record survived the trim (${WREC:-unparsed})"
faults_zero "leg C3"
VLINE=$(srv_grep "trim victim bank=")
log "C3 PASS (A+B coexist trim-free at 2-set budget; victim named: $VLINE; warm_records=$WREC)"

log "V062 GATE: ALL LEGS PASS (receipts in $OUT)"
