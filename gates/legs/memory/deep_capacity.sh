#!/bin/bash
# gates/legs/memory/deep_capacity.sh — the v0.6 release-claim proving leg:
# how many tokens can sit resident and warm on one box at SHIPPED
# defaults, without pressure (floor intact, zero refusals).
#
# Shape: zero-config -c 786432 boot (4 banks by the MT-5 default); four
# sequential needle admits of TOK each (default 450000 -> ~1.8M total,
# sized from the measured boot-free 12.5 GiB minus the 4 GiB floor at
# the memcal-measured 4.67 KiB/tok all-in, with ~0.5 GiB margin on the
# last admission).  Asserts per admit: rc=0, needle answered EXACTLY,
# prompt_tokens >= 96% of target.  End state: zero admission rejects,
# no commit-rate anomaly, MemAvailable still above the floor, and the
# loaded-decode stamp at full depth (<= 1.30 tripwire vs at-rest).
# The FINAL LINE prints the proven resident-token total: that number,
# not the target, is what the announcement may claim.
#
# Runs FROM the Mac over SSH + tunnel.  ~55 min.  End state: server
# killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL:-18002}
CTX=${CTX:-786432}
TOK=${TOK:-450000}
N=${N:-4}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${OUT:-/tmp/deep_capacity_$$}
SRV=/tmp/deep_capacity_srv.log
URL="http://127.0.0.1:$TUNNEL/v1/chat/completions"
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
wd(){ perl -e 'alarm shift @ARGV; exec @ARGV or die' "$@"; }
cleanup(){ ssh "$R" "pkill -x ${BIN:0:15}; exit 0" 2>/dev/null
           pkill -f "[s]sh -f -N -L $TUNNEL:" 2>/dev/null; }
fail(){ log "FAIL: $*"; scp -q "$R:$SRV" "$OUT/srv_fail.log" 2>/dev/null; cleanup; exit 1; }
metric(){ curl -s -m 5 "http://127.0.0.1:$TUNNEL/metrics" | grep "^$1" | awk '{print $2}' | head -1; }
avail_mib(){ ssh "$R" "awk '/MemAvailable/{printf \"%d\", \$2/1024}' /proc/meminfo"; }
decode_stamp(){ # wall seconds of a fixed 128-token probe (see deep_admit_ab_gate)
  wd 300 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/PR.json" "$URL" "$1" > "$OUT/$1.out" 2>&1 || \
    fail "$1 decode probe transport"
  sed -n 's/.*total=\([0-9.]*\)s.*/\1/p' "$OUT/$1.out" | head -1; }

CODES=("7413-DELTA-ONYX" "5527-IRON-MESA" "9082-CEDAR-VAULT" "2201-BRASS-FJORD" "6640-SLATE-HARBOR" "3318-COPPER-GLEN")
log "build $N prompts of ~$TOK tokens (agent shape, no max_tokens)"
for i in $(seq 0 $((N-1))); do
  python3 "$REPO/gates/lib/needle_prompt.py" "$TOK" "$OUT/P$i.json" 0.5 "${CODES[$i]}" deepseek-v4-flash "$i" || fail "prompt $i"
  python3 - "$OUT/P$i.json" <<'EOF' || fail "shape strip"
import json, sys
b = json.load(open(sys.argv[1])); b.pop("max_tokens", None); json.dump(b, open(sys.argv[1], "w"))
EOF
done
cat > "$OUT/PR.json" <<'EOF'
{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Count from one to two hundred as English words, comma separated, no other text."}],"max_tokens":128}
EOF

log "boot: fresh $BIN -c $CTX (zero-config)"
ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
ssh "$R" 'prev=$(awk "/MemAvailable/{print \$2}" /proc/meminfo); i=0
  while [ $i -lt 24 ]; do sleep 5
    cur=$(awk "/MemAvailable/{print \$2}" /proc/meminfo)
    d=$((cur - prev)); [ $d -lt 0 ] && d=$((-d))
    [ $d -lt 512000 ] && exit 0
    prev=$cur; i=$((i+1)); done; exit 0'
ssh "$R" ": > $SRV; cd $BINDIR; env DS4_SERVER_DEFAULT_TEMP=0 DS4_ADMIT_DEBUG=1 \
    setsid nohup ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || \
    fail "BOOT-DIED: $(ssh "$R" "tail -3 $SRV" 2>/dev/null | tr '\n' ' ')"
  sleep 10; n=$((n+10)); [ $n -ge 1500 ] && fail "boot timeout"
done
pkill -f "[s]sh -f -N -L $TUNNEL:" 2>/dev/null; sleep 1
ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" || fail "tunnel"
sleep 2
AV0=$(avail_mib)
log "boot up; MemAvailable ${AV0} MiB"

AT_REST=$(decode_stamp d0)
[ -n "$AT_REST" ] || fail "no at-rest stamp"
log "at-rest decode probe ${AT_REST}s / 128 tok"
RJ0=$(metric ds4_cont_admit_rejects_total); RJ0=${RJ0:-0}

TOTAL=0
for i in $(seq 0 $((N-1))); do
  log "admit $((i+1))/$N (~$TOK tok)"
  wd 3000 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/P$i.json" "$URL" "A$i" "$OUT/R$i.json" > "$OUT/A$i.out" 2>&1
  RC=$?; tail -2 "$OUT/A$i.out"
  [ $RC -eq 0 ] || fail "admit $((i+1)) rc=$RC"
  grep -q "${CODES[$i]}" "$OUT/A$i.out" || fail "admit $((i+1)): needle not answered exactly"
  PT=$(python3 -c "import json; print(json.load(open('$OUT/R$i.json'))['usage']['prompt_tokens'])")
  [ "$PT" -ge $((TOK * 96 / 100)) ] || fail "admit $((i+1)): prompt_tokens $PT < 96% of $TOK"
  TOTAL=$((TOTAL + PT))
  log "admit $((i+1)) OK: $PT tokens (running total $TOTAL); MemAvailable $(avail_mib) MiB"
done

RJ1=$(metric ds4_cont_admit_rejects_total); RJ1=${RJ1:-0}
[ "$RJ1" = "$RJ0" ] || fail "admission rejects grew $RJ0 -> $RJ1 (not refusal-free)"
AV1=$(avail_mib)
[ "$AV1" -ge 4096 ] || fail "MemAvailable $AV1 MiB below the 4 GiB floor (pressure)"
WARN=$(ssh "$R" "grep -c 'cont commit rate .* exceeds' $SRV || true" | tail -1)
[ "${WARN:-0}" = "0" ] || fail "commit-rate anomaly warning fired"
OBS=$(metric 'ds4_cont_commit_bytes_per_token{kind="observed"}')
PHYS=$(metric 'ds4_cont_commit_bytes_per_token{kind="packed"}')
LOADED=$(decode_stamp d1)
[ -n "$LOADED" ] || fail "no loaded stamp"
STAMP=$(python3 -c "print(round($LOADED / $AT_REST, 3))")
python3 -c "import sys; sys.exit(0 if $STAMP <= 1.30 else 1)" || \
  fail "loaded decode slowdown x$STAMP exceeds 1.30 at $TOTAL resident"
curl -s -m 10 "http://127.0.0.1:$TUNNEL/metrics" > "$OUT/metrics_end.txt"
scp -q "$R:$SRV" "$OUT/srv.log" 2>/dev/null
cleanup
log "CAPACITY LEG PASS — $TOTAL tokens resident and warm, zero refusals, MemAvailable $AV1 MiB (floor intact), decode x$STAMP vs at-rest (${AT_REST}s -> ${LOADED}s / 128 tok), rate obs=$OBS packed=$PHYS B/tok; artifacts in $OUT"
