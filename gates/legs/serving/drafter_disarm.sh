#!/bin/bash
# serving/drafter_disarm.sh — v0.5.4: a draft-phase failure DISARMS spec
# decode (loud, once, process-wide) and the cont batch continues serving
# plain — it must never die.  Field shape: cublas-q8 auto-downgrade ->
# "cont-mtp step 1 FAIL @draft" -> "continuous batch failed" killed deep
# cont batches (376884/113 sweep, secondary finding).
#
#   leg disarm (-c 2048 stock + DS4_MTP_DRAFT_FAIL_STEP=3): request 1
#       engages spec (window-1 ACCEPT drafts>0), the injected failure at
#       step 3 prints the loud disarm line EXACTLY ONCE, the request still
#       completes; request 2 decodes fully plain (window-2 ACCEPT line
#       present with drafts=0); zero "continuous batch failed" / "FAIL
#       @draft" lines; server alive end to end.
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
SRV=/tmp/drafter_disarm_gate.log
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; ssh "$R" "tail -4 $SRV" 2>/dev/null; kill_all; exit 1; }

log "== leg disarm (stock boot, injected drafter failure at step 3) =="
kill_all
ssh "$R" ": > $SRV; cd $BINDIR; DS4_MTP_DRAFT_FAIL_STEP=3 setsid nohup ./$BIN -c 2048 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
  sleep 10; n=$((n+10)); [ $n -ge 600 ] && fail "boot timeout"
done

# ask -> "finish_reason completion_tokens"; window drafts read from $SRV.
ask(){
  ssh "$R" "curl -s -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"ds4\",\"max_tokens\":120,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Explain in a few sentences why tidal energy is predictable.\"}]}'" |
    python3 -c "import json,sys; j=json.load(sys.stdin); c=j['choices'][0]; print(c['finish_reason'], j['usage']['completion_tokens'])"
}
# window_drafts <byte-offset> -> "accept_lines drafts_sum" over the log tail
window_drafts(){
  ssh "$R" "tail -c +$(( $1 + 1 )) $SRV" | python3 -c "
import sys
lines = [l for l in sys.stdin if 'CONT_MTP_ACCEPT' in l]
d = sum(int(kv.split('=')[1]) for l in lines for kv in l.split() if kv.startswith('drafts='))
print(len(lines), d)"
}

off1=$(ssh "$R" "stat -c %s $SRV")
read -r fin1 ct1 <<<"$(ask)"
sleep 3
read -r nacc1 drafts1 <<<"$(window_drafts "$off1")"
[ "$ct1" -gt 0 ] || fail "request 1 produced no tokens (finish=$fin1)"
[ "$nacc1" -ge 1 ] || fail "request 1: no ACCEPT line (spec mode not armed?)"
[ "$drafts1" -gt 0 ] || fail "request 1: drafts=0 — spec never engaged before the injected failure"
log "request 1 PASS (finish=$fin1 ct=$ct1 drafts=$drafts1: spec engaged, then failed at step 3)"

nloud=$(ssh "$R" "grep -c 'speculative decode DISABLED for this process' $SRV")
[ "$nloud" -eq 1 ] || fail "disarm line count $nloud != 1"
ssh "$R" "grep -q 'drafter forward FAILED at cont step 3' $SRV" || fail "disarm line missing the failing step"
ssh "$R" "grep -q 'continuous batch failed' $SRV" && fail "batch DIED — disarm did not hold"
ssh "$R" "grep -q 'FAIL @draft' $SRV" && fail "old batch-death path still taken"
log "disarm line PASS (loud, once, no batch death)"

off2=$(ssh "$R" "stat -c %s $SRV")
read -r fin2 ct2 <<<"$(ask)"
sleep 3
read -r nacc2 drafts2 <<<"$(window_drafts "$off2")"
[ "$ct2" -gt 0 ] || fail "request 2 produced no tokens (finish=$fin2)"
[ "$nacc2" -ge 1 ] || fail "request 2: no ACCEPT line"
[ "$drafts2" -eq 0 ] || fail "request 2: drafts=$drafts2 — disarm did not persist"
nloud=$(ssh "$R" "grep -c 'speculative decode DISABLED for this process' $SRV")
[ "$nloud" -eq 1 ] || fail "disarm line re-fired ($nloud)"
ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" || fail "server died"
log "request 2 PASS (finish=$fin2 ct=$ct2 drafts=0: plain decode, disarm persisted)"

kill_all
log "drafter_disarm_gate PASS"
