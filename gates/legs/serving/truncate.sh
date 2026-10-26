#!/bin/bash
# serving/truncate.sh — v0.5.3 inc1: deep partial-truncate regression gate.
#
# Field class (forum 376884 posts 113+122, chased 2026-08-02 on .33, 18
# attempts — full receipts in the ledger): an agent harness's history
# compression rewrites a deep conversation, driving in-place TRUNCATE
# reuse of a warm bank ("partial truncate admit bank=N cut=C suffix=S").
# The reporter's v0.5.0 box crashed (cuBLAS f16 GemmEx status 14 ->
# illegal memory access, poisoned context) on the first truncate whose
# committed length straddled ctx/2 with the cut below it, after three
# earlier truncates served fine.  Local repro of every observable
# predicate — exact geometry, concurrent spec load, multi-turn
# rollback-built trunks, sequential truncate chains, the ctx/2 straddle
# — serves CLEAN on v0.5.0 and v0.5.2; the remaining field delta is
# hours-scale drafter-residency memory pressure (v0.5.0-only machinery,
# removed by the v0.5.1 revert b8702cd).  This gate pins the whole
# validated envelope so the truncate path can never regress silently:
#
#   leg truncate_seq  -c 262144 zero-config, COALESCE_MAX=2.  Drives the
#       attempt-17 sequence: cold 67.6k doc, then SEVEN sequential
#       truncate-reuses of one bank via single-message document rewrites
#       (multi-turn rewrites cannot place cuts past the last prompt
#       boundary — assistant-render LCP divergence, see ledger), ending
#       at the field's crash shape: cut~130.5k < ctx/2=131072 < committed
#       (usage-receipt-proven) with suffix ~8.4k against a 6x-truncated
#       bank.  asserts: exactly 7 "partial truncate admit" lines; final
#       cut in [128000,131071]; u7 committed > 131072; every request 200
#       + finish=stop; needle INSIDE the truncate-replayed suffix answered
#       correctly (the word fjord exists only in the final tail); >=1 spec-accept
#       line with drafts>0; ZERO cuBLAS/illegal/synchronize-failed/
#       continuous-batch-failed lines; server alive.
#   leg partial_off   same boot + DS4_SERVER_FORK_PARTIAL=0 (the field
#       workaround).  Short sequence (cold + one rewrite).  asserts: ZERO
#       partial AND zero warm admit lines (the rewrite re-prefilled COLD
#       by elimination; with a virgin bank free there is no evict line),
#       all 200, server alive — proves the kill switch routes to cold.
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
SRV=/tmp/truncate_gate.log
OUT=${OUT:-/tmp/truncate_gate_$$}
mkdir -p "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ ssh "$R" "pkill -x ds4-server; pkill -x '${BIN:0:15}'; sleep 2; pkill -9 -x ds4-server 2>/dev/null; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; kill_all; exit 1; }
wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge "$1" ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "mem gate"; sleep 5
  done }

boot(){ # $1 = extra env
  kill_all
  wait_mem 100
  ssh "$R" ": > $SRV; cd $BINDIR; env DS4_SERVER_COALESCE_MAX=2 $1 setsid nohup \
      ./$BIN -c 262144 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ds4-server >/dev/null || pgrep -x '${BIN:0:15}' >/dev/null" 2>/dev/null || \
      fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
  curl -s -m 5 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1 || \
    { ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null; sleep 2; }
}

driver(){ # $1 = mode: full|short
  python3 - "$1" "http://127.0.0.1:$TUNNEL" <<'PY'
import json, sys, time, urllib.request
mode, base = sys.argv[1], sys.argv[2]
URL = base + "/v1/chat/completions"
def words(n, start=0, alt=0):
    base_w = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india","juliet"]
    alts = [["kilo","lima","mike","november","oscar","papa","quebec","romeo","sierra","tango"],
            ["uniform","victor","whiskey","xray","yankee","zulu","anchor","beacon","cinder","dune"],
            ["ember","fjord","garnet","harbor","iris","jasper","krill","lumen","meadow","nectar"]]
    src = alts[(alt-1) % 3] if alt else base_w
    return [("mark%d" % i) if i % 1000 == 0 else src[i % 10] for i in range(start, start+n)]
ASK = "\n\nIgnore the word pattern above. Do not continue it. Reply with exactly: chunk received"
def post(msgs, mt):
    body = json.dumps({"messages": msgs, "max_tokens": mt}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"], d["choices"][0].get("finish_reason","?"), d.get("usage",{})
def logp(m): print("[gate-py %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)
def serve(doc, label, ask=ASK, mt=1600, tries=3):
    for i in range(tries):
        c, f, u = post([{"role":"user","content":" ".join(doc)+ask}], mt)
        logp("%s try%d finish=%s total=%s" % (label, i+1, f, u.get("total_tokens")))
        if f == "stop": return c, u
    raise SystemExit("STAMP: %s finish=length x%d" % (label, tries))
if mode == "short":
    u1, _ = serve(words(45000), "u1 cold")
    serve(words(44880) + words(1000, start=44880, alt=2), "u2 rewrite (expect COLD under FORK_PARTIAL=0)")
    raise SystemExit(0)
u = words(45000); serve(u, "u1 cold ~67.6k")
u = u[:44880] + words(1000, start=44880, alt=2); serve(u, "u2 T1")
u = u + words(11300, start=100000);              serve(u, "u3 grow")
u = u[:-60] + words(5, start=990001, alt=2);     serve(u, "u4 T2")
u = u + words(24300, start=200000);              serve(u, "u5 grow")
u = u[:-60] + words(5, start=991001, alt=2);     serve(u, "u6 T3")
u = u + words(6200, start=300000)
_, u7 = serve(u, "u7 cross ctx/2")
if u7.get("total_tokens", 0) <= 131072:
    raise SystemExit("PREDICATE: u7 committed %s <= 131072" % u7.get("total_tokens"))
logp("u7 committed=%d > 131072 OK" % u7["total_tokens"])
final = u[:86730] + words(4900, start=305000, alt=3)
# needle: 'fjord' exists ONLY in the truncate-replayed suffix corpus
c, _ = serve(final, "u8 FIELD SHAPE",
             ask="\n\nIgnore the word pattern above. Does the word fjord appear anywhere in this document? Reply with exactly one word: yes or no.",
             mt=4000)
if "yes" not in c.lower():
    raise SystemExit("NEEDLE: expected yes (fjord is in the replayed suffix), got: %r" % c[:120])
logp("needle OK (fjord visible in truncate-replayed suffix)")
PY
}

log "== leg truncate_seq =="
boot ""
driver full || fail "truncate_seq driver"
pt=$(ssh "$R" "grep -ac 'partial truncate admit' $SRV")
[ "$pt" = 7 ] || fail "expected 7 partial truncate admits, got $pt"
lastcut=$(ssh "$R" "grep -a 'partial truncate admit' $SRV | tail -1 | sed 's/.*cut=\([0-9]*\).*/\1/'")
[ "$lastcut" -ge 128000 ] && [ "$lastcut" -lt 131072 ] || fail "final cut $lastcut outside [128000,131072)"
bad=$(ssh "$R" "grep -acE 'cuBLAS|illegal|synchronize failed|continuous batch failed' $SRV")
[ "$bad" = 0 ] || fail "$bad error lines: $(ssh "$R" "grep -aE 'cuBLAS|illegal|synchronize failed|continuous batch failed' $SRV | head -3")"
spec=$(ssh "$R" "grep -a 'CONT_MTP_ACCEPT' $SRV | grep -vc 'drafts=0 '")
[ "${spec:-0}" -ge 1 ] || fail "no spec-accept line with drafts>0"
ssh "$R" "pgrep -x ds4-server >/dev/null || pgrep -x '${BIN:0:15}' >/dev/null" || fail "server died"
log "truncate_seq PASS (7 truncates, final cut=$lastcut, spec engaged, clean log)"

log "== leg partial_off =="
boot "DS4_SERVER_FORK_PARTIAL=0"
driver short || fail "partial_off driver"
pt=$(ssh "$R" "grep -ac 'partial truncate admit\|partial fork admit' $SRV")
[ "$pt" = 0 ] || fail "partial admits under FORK_PARTIAL=0: $pt"
wa=$(ssh "$R" "grep -ac 'warm admit' $SRV")
[ "$wa" = 0 ] || fail "unexpected warm admit under FORK_PARTIAL=0 ($wa) — rewrite should re-prefill cold"
bad=$(ssh "$R" "grep -acE 'cuBLAS|illegal|synchronize failed|continuous batch failed' $SRV")
[ "$bad" = 0 ] || fail "$bad error lines in partial_off"
log "partial_off PASS (0 partial admits, 0 warm admits — rewrite went cold; clean log)"

kill_all
log "truncate_gate PASS"
