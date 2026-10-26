#!/bin/bash
# serving/reasoning_effort.sh — v0.5.4: --reasoning-effort default + efforts
# honored at any context (378855/30; the server no longer applies the
# 393216 floor).
#
#   leg flags    (seconds): --reasoning-effort banana exits non-zero.
#   leg explicit (-c 2048, stock): request-level reasoning_effort engages
#       BELOW the old floor — prompt_tokens grows by the prefix for high,
#       more for max; "off" strips reasoning_content entirely.
#   leg default  (-c 2048 --reasoning-effort max): boot logs the default +
#       the 384K advisory; a field-less request renders the max prefix
#       (prompt_tokens == the explicit-max count from leg explicit); an
#       explicit "low" still wins over the server default.
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
SRV=/tmp/reasoning_effort_gate.log
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; ssh "$R" "tail -4 $SRV" 2>/dev/null; kill_all; exit 1; }

boot(){ # $1 = extra args
  ssh "$R" ": > $SRV; cd $BINDIR; setsid nohup ./$BIN -c 2048 $1 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 600 ] && fail "boot timeout"
  done
}
# ask <effort|omit> -> "prompt_tokens reasoning_present"
ask(){
  local field=""
  [ "$1" != "omit" ] && field="\"reasoning_effort\":\"$1\","
  ssh "$R" "curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"ds4\",${field}\"max_tokens\":40,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}]}'" |
    python3 -c "import json,sys; j=json.load(sys.stdin); m=j['choices'][0]['message']; print(j['usage']['prompt_tokens'], 1 if m.get('reasoning_content') else 0)"
}

log "== leg flags =="
ssh "$R" "cd $BINDIR && ./$BIN --reasoning-effort banana --port $PORT" >/dev/null 2>&1 && fail "bad level accepted"
log "flags PASS (banana rejected)"

log "== leg explicit (stock boot, ctx far below the old floor) =="
kill_all
boot ""
read -r pt_base r_base <<<"$(ask omit)"
read -r pt_low  r_low  <<<"$(ask low)"
read -r pt_high r_high <<<"$(ask high)"
read -r pt_max  r_max  <<<"$(ask max)"
read -r pt_off  r_off  <<<"$(ask off)"
log "prompt_tokens omit=$pt_base low=$pt_low high=$pt_high max=$pt_max off=$pt_off"
[ "$pt_low" -eq "$pt_base" ] || fail "low != default rendering ($pt_low vs $pt_base)"
[ "$pt_high" -ge $((pt_base + 40)) ] || fail "high prefix did not engage below the old floor"
[ "$pt_max" -gt "$pt_high" ] || fail "max prefix not above high"
[ "$r_high" -eq 1 ] && [ "$r_max" -eq 1 ] || fail "reasoning_content missing at high/max"
[ "$r_off" -eq 0 ] || fail "off still produced reasoning_content"
kill_all
log "explicit PASS"

log "== leg default (--reasoning-effort max) =="
boot "--reasoning-effort max"
ssh "$R" "grep -q 'default reasoning effort: max' $SRV" || fail "default not logged"
ssh "$R" "grep -q '384K-token output budget' $SRV" || fail "shallow-ctx advisory not logged"
read -r pt_dmax r_dmax <<<"$(ask omit)"
read -r pt_dlow r_dlow <<<"$(ask low)"
log "prompt_tokens omit=$pt_dmax explicit-low=$pt_dlow"
[ "$pt_dmax" -eq "$pt_max" ] || fail "field-less request did not render the max prefix ($pt_dmax vs $pt_max)"
[ "$pt_dlow" -eq "$pt_base" ] || fail "explicit low did not win over the server default"
kill_all
log "default PASS"
log "reasoning_effort_gate PASS"
