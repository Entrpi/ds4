#!/bin/bash
# gates/legs/serving/warm_equal_match.sh — #10: equal-length record vs the
# full-match scan, ranked by DELIVERED TOKENS.
#
# The full-match scan requires record.len < prompt.len STRICTLY, so a
# prompt equal to a record (the 378855/41 aborted-thinking
# prompt-as-record shape) could never win it, and an older shorter
# full-prefix record was preferred -- re-prefilling the tail.  The fix
# defers the full match when the best partial candidate's ACTUAL token
# cut strictly beats the full record's committed length (bytes are only
# a prefilter: sampled-vs-canonical divergence -- think blocks, splice
# points -- caps the canonical LCP, so byte ranking over-promises;
# first gate run measured a 160-token loss on a small-delta shape).
#
# Drive (the 378855/41 field chain itself, all on one ~12k request):
#   T1 send P, kill mid-PREFILL (~6 s of ~15 s) -> aborted admission
#      keeps the WATERMARK as a record (canonical tokens; bank A, the
#      short full-match competitor; parse WA from the log line).
#   T2 resend P, kill mid-THINK (~22 s) -> aborted thinking row keeps
#      the PROMPT as its record (bank B, equal-length; committed =
#      canonical watermark splice + think tail).
#   T3 resend P to completion: the probe.
#
# ADJUDICATION (08-05, three gate iterations): in SPLICE lineages (any
# bank whose admission reused another record) the canonical token-LCP
# dies at the splice/think boundary, so an equal-length record can never
# deliver more than the full-match competitor -- the docket premise
# ("prompt record 39136 was reusable") was FALSE: its deliverable was
# the observed cut=26208 < trunk 26210.  The refined matcher measures
# delivered tokens per admission and only defers the full match when the
# partial strictly wins (cold-canonical lineages).  This gate asserts
# THE GUARD on the splice shape: byte-bait must NOT flip the pick, and
# the full match's exact-splice reuse is preserved.  (Counterfactual
# receipt: the interim byte-LCP build DID flip and delivered 160 fewer
# tokens -- warm_equal run 00:34-00:37 in the ledger.)
#
# Leg guard (BIN): T3 shows NO preference line, a warm/fork admit with
# cached == the watermark (the exact-splice reuse), 200 completion.
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
REFBIN=${REFBIN:-ds4-server-v055i17}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
CTX=${CTX:-32768}
SRV=/tmp/warm_equal_gate.log
OUT=${OUT:-/tmp/warm_equal_gate_$$}
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ ssh "$R" "pkill -x ds4-server-v055 2>/dev/null; pkill -x ds4-server 2>/dev/null; sleep 2; pkill -9 -x ds4-server 2>/dev/null; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; ssh "$R" "tail -5 $SRV" 2>/dev/null; kill_all; exit 1; }

wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable never reached 100G"; sleep 5
  done }

boot(){ # $1 = binary
  kill_all; wait_mem
  ssh "$R" ": > $SRV; cd $BINDIR; DS4_SERVER_COALESCE_MAX=4 setsid nohup ./$1 -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -f 'ds4-server.*--port $PORT' >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
  curl -s -m 5 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1 || {
    ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true; sleep 2
    curl -s -m 10 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null || fail "tunnel"
  }
}

p_body(){ python3 - <<'PY'
import json
pad = ("ledger row: alpha bravo charlie delta echo foxtrot golf hotel india " * 1200).strip()
print(json.dumps({"messages": [{"role": "user", "content":
  pad + "\nList three notable words from the ledger and write a detailed "
        "analysis of its structure in at least 600 words."}],
  "max_tokens": 3000, "reasoning_effort": "low", "stream": True}))
PY
}

fire_kill(){ # $1 = out tag, $2 = kill after seconds
  curl -s -N -m 600 "http://127.0.0.1:$TUNNEL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$OUT/p.json" > "$OUT/$1.sse" 2>&1 &
  local pid=$!
  sleep "$2"
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
}

drive(){ # $1 = leg tag; sets WA (watermark tokens)
  p_body > "$OUT/p.json"

  log "-- $1 T1: kill mid-prefill (watermark record = the short competitor)"
  fire_kill "$1_t1" 6
  sleep 4
  local wline
  wline=$(ssh "$R" "grep 'committed tokens as warm record' $SRV | tail -1")
  [ -n "$wline" ] || fail "$1 T1: no aborted-admission watermark record (kill landed outside prefill?)"
  WA=$(echo "$wline" | grep -oE "keeps [0-9]+ committed" | grep -oE "[0-9]+")
  log "$1 T1 watermark record: $WA tokens"
  [ "$WA" -ge 1000 ] || fail "$1 T1 watermark only $WA tokens (prefill too fast; raise prompt size)"

  log "-- $1 T2: resend, kill mid-think (prompt-as-record = the equal-length trap)"
  fire_kill "$1_t2" 22
  sleep 4
  ssh "$R" "grep -q 'token prompt as warm record' $SRV" \
    || fail "$1 T2: no prompt-as-record (kill landed outside the think row?)"

  log "-- $1 T3: RETRY to completion (the probe)"
  MARK=$(ssh "$R" "wc -l < $SRV")
  curl -s -N -m 600 "http://127.0.0.1:$TUNNEL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$OUT/p.json" > "$OUT/$1_t3.sse" \
    || fail "$1 T3 request"
  grep -q "data:" "$OUT/$1_t3.sse" || fail "$1 T3 produced no output"
  ssh "$R" "tail -n +$((MARK+1)) $SRV" > "$OUT/$1_retry.log"
}

log "== leg guard ($BIN) =="
boot "$BIN"
drive guard
grep -q "warm pick prefers partial trunk" "$OUT/guard_retry.log" \
  && fail "guard leg: preference fired on a splice lineage (byte-bait flipped the pick?!)"
GC=$(grep -oE "(warm|fork) admit.*cached=[0-9]+" "$OUT/guard_retry.log" | grep -oE "cached=[0-9]+" | tail -1 | cut -d= -f2)
[ -n "$GC" ] || fail "guard leg: retry produced no warm/fork admit (trap not built?)"
[ "$GC" -ge "$WA" ] \
  || fail "guard leg: retry reused $GC < watermark $WA (exact-splice reuse lost)"
log "guard leg PASS: no preference on splice bait; retry reused cached=$GC (watermark $WA)"

kill_all
log "WARM-EQUAL-MATCH-GATE PASS (guard: splice lineage kept the full match, cached=$GC >= watermark=$WA)"
