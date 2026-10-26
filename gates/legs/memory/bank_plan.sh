#!/bin/bash
# gates/legs/memory/bank_plan.sh — governed cont bank plan (v0.6.2): boot
# bank count sized from the live budget instead of the frozen MT-5
# fundable-token ladder (DS4_COALESCE_FUNDABLE_TOKENS).
#
# What changed and what this gate pins:
#   - coalesce_max_boot requests the regime ceiling (32) wherever a live
#     memory answer exists; create_fit's kv plan prices the ladder's own
#     criterion (banks x ctx <= fundable tokens) from the ACTUAL boot
#     budget and the shape-derived banded packed rate (the same truth
#     admissions are charged, MT-7).  KV-only divisor; eager fit still
#     the separate bound; floor 4; <=16384 the W8 32 stands verbatim.
#   - DS4_SERVER_COALESCE_MAX still overrides AND disarms the kv plan
#     (the operator's number rules).  DS4_BATCH_FIT_KV=0 restores the
#     MT-5 ladder + eager-only fit end to end.
#   - the worker's static gather bound reads the persistent ctx's actual
#     bank count (agreement by construction; not directly observable
#     here beyond healthy static serves).
#
# Legs:
#   U:  box-side unit suite (./ds4_test --server) on the fresh build.
#   S:  -c 16384 stock    -> requested 32 verbatim, NO kv-plan line
#       (regime guard); granted recorded (fit may grant fewer).
#   M:  -c 262144 stock (the CUDA DEFAULT since 5985bad) -> kv-plan
#       ledger line present (at the measured 4142 B/tok rate the bound
#       binds well below the eager fit here); granted == min(plan, 32);
#       4 <= granted <= 32; no descent lines; arithmetic recheck from
#       the printed inputs (off-by-one tolerated as WARN: the budget
#       prints at 2dp GiB).
#   A:  (same boot as M) admission battery: N = granted concurrent
#       buffered chats, nonce-stamped driver, every stream HTTP 200 +
#       nonempty content; zero cont admit rejects, zero census/governor
#       faults, no crash lines.
#   O:  -c 524288 COALESCE_MAX=6 -> granted 6 (explicit override beats
#       the plan's deep floor 4: the disarm works).
#   K:  -c 262144 FIT_KV=0 -> granted 4 (ladder fallback end to end).
#   D:  -c 524288 stock   -> granted >= 4; kv-plan line present; one
#       completion serves; faults zero.
#   T:  decode stamp at -c 262144 (the CUDA DEFAULT since 5985bad --
#       the tier where the plan raises the default boot from the
#       ladder's 4 to what the live budget funds): stock boot vs
#       FIT_KV=0 boot, SAME width N=min(G_stock,G_ladder) on both
#       (apples to apples per-stream), then N=G_stock on the stock boot
#       when it grants more (the parallelism line).  A STAMP: asserts
#       clean completion + zero faults and RECORDS window tok/s +
#       per-stream ms/tok + tok/step; the throughput adjudication lives
#       in the receipt (fresh boots both sides per the observability
#       law).
#
# Runs FROM the Mac over SSH.  Takes /tmp/ds4_box_lock on the box (the
# two-session law) and releases it at exit.  End state: ds4-server
# killed by name, box lock freed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

# LAUNCH LAW (ssh-daemonize, measured on .33): plain ssh + setsid --fork
# + a local alarm; never `& echo $!`, never ServerAlive options on a
# launching call.  SSH = short control calls; SSH_LONG = blocking calls
# that legitimately run long (build, drivers) -- still alarmed so a
# wedge fails loud instead of hanging the gate.
SSH(){ perl -e 'alarm 120; exec @ARGV or die' ssh "$@"; }
SSH_LONG(){ perl -e 'alarm 2700; exec @ARGV or die' ssh "$@"; }

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
SRV=/tmp/bankplan_gate_srv.log
OUT=${OUT:-/tmp/bankplan_gate_$$}
TS=$(date +%s); NONCE="bankplan_${TS}_$$"
mkdir -p "$OUT"
DRV="$OUT/driver.log"

log(){ echo "[$(date +%H:%M:%S)] BANKPLAN: $*" | tee -a "$DRV"; }

# Mac-side single-instance guard (taken before any box interaction).
LOCKDIR=/tmp/bankplan_gate.lockdir
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "another bankplan_gate instance is running ($LOCKDIR, pid $(cat "$LOCKDIR/pid" 2>/dev/null || echo '?'))"; exit 3
fi
echo $$ > "$LOCKDIR/pid"

BOX_LOCKED=0
full_exit(){
  rc=$?
  # Touch the box ONLY when this instance owns it (two-session law: a
  # refused ownership check must leave the other session's server alone).
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
fail(){ log "FAIL: $*"; SSH "$R" "tail -8 $SRV 2>/dev/null; exit 0" 2>/dev/null | tee -a "$DRV"; log "BANKPLAN GATE: FAILED"; exit 1; }

# ---- box ownership (two-session law): check FIRST, act in a SECOND call ----
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
ready_field(){ srv_grep "persistent batch ctx ready" | sed -n "s/.*$1=\([0-9]*\).*/\1/p"; }
metric(){ SSH "$R" "curl -s -m 10 http://127.0.0.1:$PORT/metrics | grep -E '^$1[ {]' | head -1 | grep -oE '[0-9.]+\$'; exit 0" 2>/dev/null; }
metric_sum(){ SSH "$R" "curl -s -m 10 http://127.0.0.1:$PORT/metrics | awk '/^$1[{ ]/{s+=\$NF} END{printf \"%d\\n\", s}'; exit 0" 2>/dev/null; }
save_ledger(){ # $1 = tag: preserve the decisive boot lines
  SSH "$R" "grep -E 'batch fit|batch vmm|persistent batch ctx ready|kv plan' $SRV 2>/dev/null; exit 0" 2>/dev/null > "$OUT/ledger_$1.txt"
  cat "$OUT/ledger_$1.txt" | tee -a "$DRV"
}
faults_zero(){ # $1 = leg tag
  local cf gf rej
  cf=$(metric ds4_memory_census_faults_total); gf=$(metric ds4_memory_governor_faults_total)
  rej=$(metric_sum ds4_cont_admit_rejects_total)
  [ "${cf:-1}" = "0" ] || fail "$1: census faults $cf"
  [ "${gf:-1}" = "0" ] || fail "$1: governor faults $gf"
  [ "${rej:-1}" = "0" ] || fail "$1: cont admit rejects $rej"
  [ "$(srv_count 'illegal')" = "0" ] || fail "$1: illegal-access lines in server log"
  [ "$(srv_count 'cuBLAS')" = "0" ] || fail "$1: cuBLAS error lines"
  [ "$(srv_count 'continuous batch failed')" = "0" ] || fail "$1: continuous batch failures"
}

# The concurrent buffered-chat driver (box-side; nonce-stamped lines,
# strict per-stream verdicts -- the 08-04 collision postmortem shape).
put_driver(){
  SSH "$R" "cat > /tmp/bankplan_driver.py" <<'EOF'
import http.client, json, sys, threading, time
PORT, NS, SENT, MAXTOK = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
NONCE = sys.argv[5]
def prompt(sid):
    return ('Summarize the flood log. ' +
            ' '.join('the tidal basin at station %d-%s-%d filled ahead of the model forecast'
                     % (sid, NONCE[-6:], i) for i in range(SENT)))
ok = [False]*NS; toks = [0]*NS; ms = [0.0]*NS
def run(sid):
    t0 = time.time()
    try:
        c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=1800)
        body = json.dumps({'messages':[{'role':'user','content':prompt(sid)}],
                           'max_tokens': MAXTOK, 'temperature': 0})
        c.request('POST', '/v1/chat/completions', body,
                  {'Content-Type':'application/json'})
        r = c.getresponse(); data = r.read()
        if r.status != 200:
            print('%s stream=%d FAIL http=%d body=%s' % (NONCE, sid, r.status, data[:160]), flush=True); return
        j = json.loads(data)
        content = j['choices'][0]['message']['content'] or ''
        u = j.get('usage', {}) or {}
        toks[sid] = int(u.get('completion_tokens') or 0)
        ms[sid] = (time.time()-t0)*1000.0
        if len(content) == 0:
            print('%s stream=%d FAIL empty-content' % (NONCE, sid), flush=True); return
        ok[sid] = True
        print('%s stream=%d OK tokens=%d wall_ms=%.0f' % (NONCE, sid, toks[sid], ms[sid]), flush=True)
    except Exception as e:
        print('%s stream=%d FAIL exc=%r' % (NONCE, sid, e), flush=True)
t0 = time.time()
th = [threading.Thread(target=run, args=(i,)) for i in range(NS)]
for t in th: t.start()
for t in th: t.join()
wall = time.time()-t0
n_ok = sum(ok); tot = sum(toks)
print('%s VERDICT ok=%d/%d total_tokens=%d window_s=%.1f window_tok_s=%.1f' %
      (NONCE, n_ok, NS, tot, wall, tot/wall if wall > 0 else 0.0), flush=True)
sys.exit(0 if n_ok == NS else 1)
EOF
}

drive(){ # $1 = leg tag  $2 = streams  $3 = sentences  $4 = max_tokens
  local legnonce="${NONCE}_$1"
  log "$1: driving $2 streams (sent=$3 max_tokens=$4, nonce $legnonce)"
  SSH_LONG "$R" "cd /tmp && timeout 2400 python3 bankplan_driver.py $PORT $2 $3 $4 $legnonce" \
      > "$OUT/drive_$1.log" 2>&1
  local rc=$?
  cat "$OUT/drive_$1.log" | tee -a "$DRV"
  grep -q "$legnonce VERDICT ok=$2/$2" "$OUT/drive_$1.log" || fail "$1: driver verdict incomplete (rc=$rc)"
  local n_ok
  n_ok=$(grep -c "$legnonce stream=[0-9]* OK" "$OUT/drive_$1.log")
  [ "$n_ok" = "$2" ] || fail "$1: only $n_ok/$2 streams OK under this nonce"
}

# ---------- build + Leg U ----------
log "sync: git ls-files -> $R:$BINDIR (rsync --files-from, no --delete)"
git ls-files > "$OUT/files.txt" || fail "git ls-files failed (run from the repo root)"
rsync -a --files-from="$OUT/files.txt" . "$R:$BINDIR/" || fail "rsync failed"
log "build: make -j6 cuda-spark on $R"
SSH_LONG "$R" "cd $BINDIR && make -j6 cuda-spark" > "$OUT/build.log" 2>&1 \
  || fail "box build failed: $(tail -3 "$OUT/build.log" | tr '\n' ' ')"
SASS=$(SSH "$R" "/usr/local/cuda/bin/cuobjdump --list-elf $BINDIR/$BIN 2>/dev/null | head -3; exit 0" 2>/dev/null)
echo "$SASS" | tee -a "$DRV"
case "$SASS" in *sm_121*) : ;; *) fail "sm_121 not found in cuobjdump output (empty = could not verify)";; esac
log "build ok (sm_121 verified)"
SSH_LONG "$R" "cd $BINDIR && make -j6 ds4_test >/dev/null 2>&1; ./ds4_test --server" > "$OUT/units.log" 2>&1
grep -q "ds4 tests: ok" "$OUT/units.log" || fail "leg U: box unit suite failed: $(tail -3 "$OUT/units.log" | tr '\n' ' ')"
log "U PASS (box unit suite ok)"
put_driver

# ---------- Leg S: measured regime verbatim ----------
boot 16384 ""
save_ledger S
MS=$(ready_field max_seq)
REQ=$(srv_grep "batch fit: free=" | sed -n 's/.*(requested \([0-9]*\)).*/\1/p')
[ "${REQ:-$MS}" = "32" ] || fail "leg S: requested ${REQ:-$MS} at ctx 16384 (expected 32 verbatim; granted $MS)"
[ "$(srv_count 'kv plan')" = "0" ] || fail "leg S: kv-plan line printed inside the measured regime"
log "S PASS (requested ${REQ:-$MS}, granted $MS, no kv plan <= 16384)"

# ---------- Legs M + A: mid-deep plan + admission battery ----------
boot 262144 ""
save_ledger M
KV_LINE=$(srv_grep "kv plan")
[ -n "$KV_LINE" ] || fail "leg M: kv-plan ledger line missing at ctx 262144"
PLAN=$(echo "$KV_LINE" | sed -n 's/.*-> max_seq \([0-9]*\).*/\1/p')
MS=$(ready_field max_seq)
[ -n "$PLAN" ] && [ -n "$MS" ] || fail "leg M: could not parse plan ($PLAN) / granted ($MS)"
# The ledger prints the UNCLAMPED plan; the grant is min(plan, requested 32)
# (at the measured 4142 B/tok packed rate the mid-ctx plan can exceed 32).
PLAN_EFF=$PLAN; [ "$PLAN_EFF" -gt 32 ] && PLAN_EFF=32
[ "$MS" = "$PLAN_EFF" ] || fail "leg M: granted $MS != min(kv plan $PLAN, 32) (unexplained gap)"
[ "$MS" -ge 4 ] && [ "$MS" -le 32 ] || fail "leg M: granted $MS outside [4,32]"
[ "$(srv_count 'retrying at')" = "0" ] || fail "leg M: physical descent engaged (plan overshot free memory)"
# Arithmetic recheck from printed inputs (WARN on off-by-one: budget is 2dp GiB).
BUD_GIB=$(echo "$KV_LINE" | sed -n 's/.*budget=\([0-9.]*\) GiB.*/\1/p')
SEQC=$(echo "$KV_LINE" | sed -n 's/.*(\([0-9]*\) tok x.*/\1/p')
BPT=$(echo "$KV_LINE" | sed -n 's/.*tok x \([0-9]*\) B\/tok.*/\1/p')
BAND=$(echo "$KV_LINE" | sed -n 's/.*band \([0-9]*\)\/1024.*/\1/p')
EAGER=$(echo "$KV_LINE" | sed -n 's/.*(eager fit \([0-9]*\)).*/\1/p')
if [ -n "$BUD_GIB" ] && [ -n "$SEQC" ] && [ -n "$BPT" ] && [ -n "$BAND" ] && [ -n "$EAGER" ]; then
  RECHECK=$(python3 -c "
bud=int(float('$BUD_GIB')*(1<<30)); kv=($SEQC*$BPT*$BAND+1023)//1024
n=bud//kv; n=max(n,4); n=min(n,$EAGER); print(n)")
  if [ "$RECHECK" != "$PLAN" ]; then
    D=$((RECHECK - PLAN)); [ "${D#-}" -le 1 ] \
      && log "M WARN: recheck $RECHECK vs plan $PLAN (2dp budget rounding)" \
      || fail "leg M: plan arithmetic recheck $RECHECK != printed $PLAN"
  fi
fi
log "M PASS (ctx 262144: plan=$PLAN granted=$MS eager=$EAGER budget=${BUD_GIB}GiB bpt=${BPT}B/t band=$BAND)"
drive A "$MS" 260 64      # ~4k-token prompts x granted-many streams
faults_zero "leg A"
log "A PASS ($MS concurrent admissions clean; zero rejects, zero faults)"

# ---------- Leg O: explicit override beats the plan ----------
boot 524288 "DS4_SERVER_COALESCE_MAX=6"
save_ledger O
MS=$(ready_field max_seq)
[ "$MS" = "6" ] || fail "leg O: granted $MS at ctx 524288 with COALESCE_MAX=6 (override must beat the plan floor)"
log "O PASS (explicit 6 granted at 524288)"

# ---------- Leg K: full ladder fallback ----------
boot 262144 "DS4_BATCH_FIT_KV=0"
save_ledger K
MS=$(ready_field max_seq)
[ "$MS" = "4" ] || fail "leg K: granted $MS at ctx 262144 with FIT_KV=0 (expected ladder 4)"
[ "$(srv_count 'kv plan')" = "0" ] || fail "leg K: kv-plan line printed with FIT_KV=0"
log "K PASS (ladder fallback end to end)"

# ---------- Leg D: deep stock ----------
boot 524288 ""
save_ledger D
MS=$(ready_field max_seq)
[ -n "$MS" ] && [ "$MS" -ge 4 ] || fail "leg D: granted ${MS:-?} at ctx 524288 (floor 4)"
[ "$(srv_count 'kv plan')" -ge 1 ] || fail "leg D: kv-plan line missing at 524288"
drive Dsmoke 1 120 48
faults_zero "leg D"
log "D PASS (ctx 524288: granted $MS; smoke serve clean)"

# ---------- Leg T: decode stamp at the SHIPPING DEFAULT ctx ----------
# 262144 is the CUDA default since 5985bad: the plan raises the default
# boot from the ladder's 4 to whatever the live budget funds (~7-14 at
# the measured 4142 B/tok packed rate), so THIS is the tier the V6
# decode-throughput question must be stamped at.
boot 262144 ""
save_ledger T1
G1=$(ready_field max_seq)
[ -n "$G1" ] && [ "$G1" -ge 4 ] || fail "leg T: stock granted ${G1:-?} at 262144"
boot 262144 "DS4_BATCH_FIT_KV=0"
save_ledger T2
G2=$(ready_field max_seq)
[ -n "$G2" ] || fail "leg T: ladder-side granted unparsable"
NB=$G2; [ "$G1" -lt "$G2" ] && NB=$G1
log "T: stock grants $G1, ladder grants $G2; base width $NB"
# tokens-per-step rides every stamp line (never compare ms/tok without
# tok/step -- spec accept health must match across the sides).
decode_stamp(){ # $1 = tag: drive already ran; print tok/step delta
  local tk sp
  tk=$(metric ds4_tokens_decoded_total); sp=$(metric ds4_decode_steps_total)
  echo "tokens=$tk steps=$sp tok_per_step=$(python3 -c "print(f'{$tk/max($sp,1):.2f}')")"
}
# ladder side first (we are already booted on it), base width
drive Tladder "$NB" 700 256
faults_zero "leg T (ladder)"
LADDER_LINE="$(grep "VERDICT" "$OUT/drive_Tladder.log" | tail -1) $(decode_stamp Tladder)"
# stock side, fresh boot (observability law), base width then full width
boot 262144 ""
drive Tstock "$NB" 700 256
faults_zero "leg T (stock)"
STOCK_LINE="$(grep "VERDICT" "$OUT/drive_Tstock.log" | tail -1) $(decode_stamp Tstock)"
STOCK_FULL_LINE=""
if [ "$G1" -gt "$NB" ]; then
  drive Tstockfull "$G1" 700 256
  faults_zero "leg T (stock full)"
  STOCK_FULL_LINE=$(grep "VERDICT" "$OUT/drive_Tstockfull.log" | tail -1)
fi
log "T STAMP ladder(N=$NB):  $LADDER_LINE"
log "T STAMP stock(N=$NB):   $STOCK_LINE"
[ -n "$STOCK_FULL_LINE" ] && log "T STAMP stock(N=$G1): $STOCK_FULL_LINE"
log "T PASS (stamp recorded; adjudication in the receipt)"

log "BANKPLAN GATE: ALL LEGS PASS (receipts in $OUT)"
