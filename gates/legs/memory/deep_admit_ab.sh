#!/bin/bash
# battery: deep_admit_ab tier=2 stop=0 server=33 driver=15 trigger="a batch-fit headroom term changed" est=60 seq=20
# verdict: ^\[[0-9:]+\] (ALL LEGS PASS|FAIL:)
# catches: deep-shape admission UNDER enforcement (500k in flight + 2 x 245k admits on truth credit vs tranche=0 at 1.3x budget); the only leg where a changed headroom term shows as a refused admit at depth
# cmd: R=$RB_SERVER BINDIR=$RB_TREE OUT="$RB_OUT/ab" EXTRA_ARGS="-m $RB_VBASE --vision $RB_ENC" bash gates/legs/memory/deep_admit_ab.sh
# gates/legs/memory/deep_admit_ab.sh — MT-7 (v0.6.1 memory truth): the
# deep-admission A/B — the release-battery leg the v0.6.0 battery lacked
# (deep shapes were never tested under admission enforcement; leg-1's field
# failure lived exactly there).
#
# Leg A (tranche credit ON, the default; unpinned budget):
#   the charter concurrent shape — a 500k admit IN FLIGHT while 2x245k
#   admit beside it — must fully serve: all three rc=0, zero cont admission
#   rejects.  DS4_ADMIT_DEBUG=1 discloses every funding quote; the peak
#   union (max mres+mneed) is recorded as U.  Carries the V7 loaded-decode
#   stamp: decode ms/tok at rest (fresh boot) vs under ~1M resident bank
#   tokens; ratio asserted <= 1.30 (tripwire) and LOGGED as the release
#   stamp (the at-rest number alone proves nothing about depth).
# Leg B (tranche=0 kill switch; budget PINNED to 1.2 x U):
#   the SAME shape under the SAME funded ceiling, legacy lifetime credit.
#   The phantom decode budget (~393k tokens/agent-style row) inflates the
#   union ~1.8x, busting the pin the truth credit fit with room: the 500k
#   still admits, at least one 245k is REFUSED (rejects counter grows).
#   This is the pessimism MT-1 deleted, reproduced on demand.
#
# Runs FROM the Mac over SSH + tunnel.  Release-battery cadence (~60 min).
# End state: server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL:-18001}
CTX=${CTX:-786432}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${OUT:-/tmp/deep_admit_ab_$$}
EXTRA_ARGS=${EXTRA_ARGS:-}   # extra ds4-server flags spliced after -c (e.g. "-m VISION.gguf --vision ENC.gguf": the v0.7.0 vision-reserve trigger)
SRV=/tmp/deep_admit_ab_srv.log
URL="http://127.0.0.1:$TUNNEL/v1/chat/completions"
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
wd(){ perl -e 'alarm shift @ARGV; exec @ARGV or die' "$@"; }
cleanup(){
  ssh "$R" "pkill -x ${BIN:0:15}; exit 0" 2>/dev/null
  pkill -f "[s]sh -f -N -L $TUNNEL:" 2>/dev/null
}
fail(){ log "FAIL: $*"; scp -q "$R:$SRV" "$OUT/srv_fail.log" 2>/dev/null; cleanup; exit 1; }
metric(){ curl -s -m 5 "http://127.0.0.1:$TUNNEL/metrics" | grep "^$1" | awk '{print $2}' | head -1; }

boot(){ # $1 = env prefix string
  log "boot: fresh $BIN -c $CTX${EXTRA_ARGS:+ $EXTRA_ARGS} (env='$1')"
  ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
  ssh "$R" 'prev=$(awk "/MemAvailable/{print \$2}" /proc/meminfo); i=0
    while [ $i -lt 24 ]; do sleep 5
      cur=$(awk "/MemAvailable/{print \$2}" /proc/meminfo)
      d=$((cur - prev)); [ $d -lt 0 ] && d=$((-d))
      [ $d -lt 512000 ] && exit 0
      prev=$cur; i=$((i+1)); done; exit 0'
  ssh "$R" ": > $SRV; cd $BINDIR; env DS4_SERVER_DEFAULT_TEMP=0 DS4_ADMIT_DEBUG=1 $1 \
      setsid nohup ./$BIN -c $CTX $EXTRA_ARGS --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || \
      fail "BOOT-DIED: $(ssh "$R" "tail -3 $SRV" 2>/dev/null | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 1500 ] && fail "boot timeout"
  done
  pkill -f "[s]sh -f -N -L $TUNNEL:" 2>/dev/null; sleep 1
  ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" || fail "tunnel"
  sleep 2
  log "boot up"
}

peak_union_mib(){ # max mres+mneed over every admit-debug quote, integer MiB
  ssh "$R" "grep 'admit debug' $SRV" | python3 -c "
import re, sys
best = 0.0
for l in sys.stdin:
    mn = re.search(r'mneed=([0-9.]+) MiB', l); mr = re.search(r'mres=([0-9.]+) MiB', l)
    if mn and mr: best = max(best, float(mn.group(1)) + float(mr.group(1)))
print(int(best))"; }

decode_stamp(){ # $1 = tag -> echoes total wall seconds of a fixed 128-token
                # probe (temp-0 thinking rides a non-content SSE channel, so
                # delta-based tok/s reads 0; the 128-token cap is
                # deterministic, so wall time IS the comparative decode
                # stamp -- same tiny prompt both sides)
  wd 300 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/PR.json" "$URL" "$1" > "$OUT/$1.out" 2>&1 || \
    fail "$1 decode probe transport"
  sed -n 's/.*total=\([0-9.]*\)s.*/\1/p' "$OUT/$1.out" | head -1; }

# ---------- prompts ----------------------------------------------------------
# AGENT SHAPE (run-1 lesson): the deep rows must OMIT max_tokens -- the
# phantom decode budget (~393k tokens charged per request that never sets
# one; codex and most agents never do) exists ONLY on that shape.  With a
# max_tokens cap the legacy credit is prompt+cap and leg B refuses nothing
# (run-1: unions 2.1/3.1/4.1 GiB under any sane pin).  Uncapped, tranche
# charges prompt+32768 while legacy charges prompt+393216(clamped): THE A/B.
# Generation still ends at EOS (needle answers, temp 0) -- wd guards remain.
log "build prompts (500k + 2x245k agent-shape + decode probe)"
python3 "$REPO/gates/lib/needle_prompt.py" 500000 "$OUT/A.json" 0.5 "7413-DELTA-ONYX"  deepseek-v4-flash 0 || fail "prompt A"
python3 "$REPO/gates/lib/needle_prompt.py" 245000 "$OUT/B.json" 0.5 "5527-IRON-MESA"   deepseek-v4-flash 1 || fail "prompt B"
python3 "$REPO/gates/lib/needle_prompt.py" 245000 "$OUT/C.json" 0.5 "9082-CEDAR-VAULT" deepseek-v4-flash 2 || fail "prompt C"
python3 - "$OUT/A.json" "$OUT/B.json" "$OUT/C.json" <<'EOF' || fail "agent-shape strip"
import json, sys
for p in sys.argv[1:]:
    b = json.load(open(p)); b.pop("max_tokens", None); json.dump(b, open(p, "w"))
EOF
cat > "$OUT/PR.json" <<'EOF'
{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Count from one to two hundred as English words, comma separated, no other text."}],"max_tokens":128}
EOF

run_shape(){ # pushes 500k BG, then 2x245k BG mid-prefill; sets RC_A RC_B RC_C
  wd 3000 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/A.json" "$URL" "${1}A" > "$OUT/${1}A.out" 2>&1 &
  local PA=$!
  sleep 90    # A is mid-prefill (chunked interleave admits beside it)
  wd 1800 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/B.json" "$URL" "${1}B" > "$OUT/${1}B.out" 2>&1 &
  local PB=$!
  sleep 5
  wd 1800 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/C.json" "$URL" "${1}C" > "$OUT/${1}C.out" 2>&1 &
  local PC=$!
  wait $PA; RC_A=$?
  wait $PB; RC_B=$?
  wait $PC; RC_C=$?
  tail -2 "$OUT/${1}A.out"; tail -2 "$OUT/${1}B.out"; tail -2 "$OUT/${1}C.out"
}

# ---------- Leg A: tranche ON, unpinned --------------------------------------
boot ""
AT_REST=$(decode_stamp a0)
[ -n "$AT_REST" ] || fail "leg A: no at-rest decode stamp"
python3 -c "import sys; sys.exit(0 if float('$AT_REST') > 0.5 else 1)" || \
  fail "leg A: at-rest stamp ${AT_REST}s implausible"
log "leg A: at-rest decode probe ${AT_REST}s / 128 tok"
RJ0=$(metric ds4_cont_admit_rejects_total); RJ0=${RJ0:-0}
run_shape a
[ $RC_A -eq 0 ] || fail "leg A: 500k rc=$RC_A"
[ $RC_B -eq 0 ] || fail "leg A: 245k B rc=$RC_B"
[ $RC_C -eq 0 ] || fail "leg A: 245k C rc=$RC_C"
RJ1=$(metric ds4_cont_admit_rejects_total); RJ1=${RJ1:-0}
[ "$RJ1" = "$RJ0" ] || fail "leg A: rejects grew $RJ0 -> $RJ1 under the truth credit"
U=$(peak_union_mib)
[ -n "$U" ] && [ "$U" -ge 1000 ] || fail "leg A: peak union $U MiB implausible"
LOADED=$(decode_stamp a1)
[ -n "$LOADED" ] || fail "leg A: no loaded decode stamp"
STAMP=$(python3 -c "print(round($LOADED / $AT_REST, 3))")
log "leg A PASS: all admit beside the 500k; peak union $U MiB; V7 LOADED-DECODE STAMP at-rest ${AT_REST}s vs loaded ${LOADED}s per 128 tok (slowdown x$STAMP)"
python3 -c "import sys; sys.exit(0 if $STAMP <= 1.30 else 1)" || \
  fail "leg A: loaded decode slowdown x$STAMP exceeds the 1.30 tripwire"
scp -q "$R:$SRV" "$OUT/srv_legA.log" 2>/dev/null

# ---------- Leg B: tranche OFF, budget pinned to 1.2 x U ---------------------
# Factor derivation (agent shape, banded ~4.23 KiB/tok): leg-A peak union
# ~4.7 GiB; leg B's first bust (A@786k + B@633k credits) ~6.0 GiB -> any
# factor in (1.05, 1.28) discriminates; 1.2 leaves >=7% margin both sides.
PIN=$(( U * 12 / 10 ))
log "leg B: pinning budget to $PIN MiB (1.2 x measured truth union)"
boot "DS4_CONT_ADMIT_TRANCHE=0 DS4_BATCH_VMM_BUDGET_MB=$PIN"
RJ0=$(metric ds4_cont_admit_rejects_total); RJ0=${RJ0:-0}
run_shape b
[ $RC_A -eq 0 ] || fail "leg B: the 500k itself must admit under the pin (rc=$RC_A) -- pin too tight, not a pessimism repro"
RJ1=$(metric ds4_cont_admit_rejects_total); RJ1=${RJ1:-0}
[ "$RJ1" -gt "$RJ0" ] || fail "leg B: legacy credit produced no refusal at pin $PIN MiB (rejects $RJ0 -> $RJ1)"
[ $RC_B -ne 0 ] || [ $RC_C -ne 0 ] || fail "leg B: both 245k served despite rejects counter -- shape drifted"
log "leg B PASS: legacy phantom credit refused under the ceiling the truth credit fit (rejects $RJ0 -> $RJ1)"
scp -q "$R:$SRV" "$OUT/srv_legB.log" 2>/dev/null

cleanup
log "ALL LEGS PASS — truth credit serves the charter shape (union $U MiB, loaded-decode x$STAMP); legacy reproduces the refusal at 1.2x; artifacts in $OUT ($BIN killed, $R left free)"
