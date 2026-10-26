#!/bin/bash
# perf/width384_stamp.sh — v0.5.6 cut stamp (plan §7 acceptance): achieved LIVE
# WIDTH + throughput for the OMITTED-budget default (max_tokens absent ->
# the 384K normalized budget; admission hard-promises the full budget, so
# this is the case the credit projection can narrow).  Also stamps
# decode-step p99 client-side from an N=1 stream (inter-delta gaps; SSE
# arrival jitter noted -- steps are ~20-50ms at N=1 so the p99 signal
# dominates).  A STAMP, not a pass/fail gate: it asserts ENGAGEMENT
# (admissions happened, rows went live) and prints the numbers for the
# release notes; the acceptance judgment lives in the receipt.
#
# Legs:
#   p99_n1   one streaming chat (~400 tok), p50/p99 inter-delta ms
#   width    W concurrent OMITTED-budget streaming chats; samples
#            banks_live each second during the burst; prints max live,
#            admit/reject deltas, floor rejections, window tok/s
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box free.
# Env: R (${DS4_BOX33:?}) BINDIR (${DS4_BOX_TREE:?})
#      PORT (8000) TUNNEL_PORT (18000) CTX (524288) W (12)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-524288}
W=${W:-12}
RWORK=/tmp/width384_stamp
OUT=${OUT:-/tmp/width384_stamp_$$}
mkdir -p "$OUT"
BASE="http://127.0.0.1:$TUNNEL_PORT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; exit 1; }

tunnel_up(){
  curl -s -m 5 "$BASE/v1/models" >/dev/null 2>&1 && return 0
  ssh -f -N -L "$TUNNEL_PORT:127.0.0.1:$PORT" "$R" 2>/dev/null || true
  sleep 2
  curl -s -m 10 "$BASE/v1/models" >/dev/null 2>&1
}
wait_mem(){
  local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge "$1" ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable ${got:-?}G never reached ${1}G"
    sleep 5
  done
}
boot(){
  SRV=$RWORK/srv.log
  log "boot: killing old ds4-server on $R"
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; mkdir -p $RWORK; rm -f /tmp/ds4.lock; exit 0"
  wait_mem 100
  ssh "$R" ": > $SRV; cd $BINDIR; setsid nohup ./ds4-server -c $CTX --port $PORT \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
      sleep 3
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" 2>/dev/null | tr '\n' ' ')"
    fi
    sleep 10; n=$((n+10)); [ $n -ge 1800 ] && fail "boot timeout"
  done
  tunnel_up || fail "tunnel :$TUNNEL_PORT unreachable"
  log "boot: up (-c $CTX shipped defaults)"
}
# METRICS-HELPER TRAP: extract the number BEFORE head -1.
m(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$' | head -1; }

boot

# ---- p99_n1: client-side inter-delta p50/p99 on a single stream --------
log "p99_n1: streaming ~400 tokens"
python3 - "$BASE" > "$OUT/p99.txt" <<'PYEOF'
import json, sys, time, urllib.request
base = sys.argv[1]
body = json.dumps({"model":"m","max_tokens":400,"temperature":0,"stream":True,
  "messages":[{"role":"user","content":"Write a 300 word story about a mountain village."}]}).encode()
req = urllib.request.Request(base+"/v1/chat/completions", data=body,
                             headers={"Content-Type":"application/json"})
gaps, last = [], None
with urllib.request.urlopen(req, timeout=240) as r:
    for line in r:
        if not line.startswith(b"data: ") or b"[DONE]" in line: continue
        now = time.monotonic()
        if last is not None: gaps.append((now-last)*1000.0)
        last = now
gaps.sort()
if len(gaps) < 50: print("TOO_FEW", len(gaps)); sys.exit(1)
p = lambda q: gaps[min(len(gaps)-1, int(q*len(gaps)))]
print(f"deltas={len(gaps)} p50={p(0.50):.1f}ms p90={p(0.90):.1f}ms p99={p(0.99):.1f}ms max={gaps[-1]:.1f}ms")
PYEOF
[ $? -eq 0 ] || fail "p99_n1 leg failed"
log "p99_n1: $(cat "$OUT/p99.txt")"

# ---- width: W concurrent OMITTED-budget streams ------------------------
adm0=$(( $(m 'ds4_admits_total{kind="cold"}' 2>/dev/null || echo 0) ))
rej0=$(m ds4_cont_admit_rejects_total); rej0=${rej0:-0}
floor0=$(ssh "$R" "grep -cF 'cont admit rejected on memory floor' $RWORK/srv.log" 2>/dev/null); floor0=${floor0:-0}
log "width: firing $W concurrent OMITTED-budget streams"
pids=()
for i in $(seq 1 "$W"); do
  ( curl -s -m 300 --no-buffer -o "$OUT/w$i.sse" "$BASE/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"m","temperature":0,"stream":true,"messages":[{"role":"user","content":"Write a 250 word story about ship number '"$i"' crossing an ocean."}]}' \
      >/dev/null 2>&1 ) &
  pids+=($!)
done
maxlive=0
for t in $(seq 1 90); do
  live=$(m ds4_banks_live); live=${live:-0}
  [ "$live" -gt "$maxlive" ] && maxlive=$live
  alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
  [ "$alive" = 0 ] && break
  sleep 1
done
for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
done_n=0
for i in $(seq 1 "$W"); do grep -q 'data: \[DONE\]' "$OUT/w$i.sse" && done_n=$((done_n+1)); done
adm1=$(( $(m ds4_admits_cold_total 2>/dev/null || echo 0) ))
rej1=$(m ds4_cont_admit_rejects_total); rej1=${rej1:-0}
floor1=$(ssh "$R" "grep -cF 'cont admit rejected on memory floor' $RWORK/srv.log" 2>/dev/null); floor1=${floor1:-0}
tok_s=$(curl -s -m 10 "$BASE/metrics" | grep -F ds4_decode_tok_s | grep -oE '[0-9.]+$' | head -1)
[ "$maxlive" -ge 1 ] || fail "width: no row ever went live"
[ "$done_n" -ge 1 ] || fail "width: no stream completed"
log "width: max banks_live=$maxlive of W=$W requested; completed=$done_n; admits +$((adm1-adm0)); cont rejects ${rej0}->${rej1}; floor rejections ${floor0}->${floor1}; window decode ${tok_s:-?} tok/s"
log "STAMP COMPLETE — artifacts in $OUT"
ssh "$R" "pkill -x ds4-server; exit 0"
