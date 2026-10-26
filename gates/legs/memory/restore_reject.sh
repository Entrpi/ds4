#!/bin/bash
# memory/restore_reject.sh — v0.5.4 item 9 (forum 378855/30): the disk-restore ->
# budget-reject -> serial-fallback cycle must serve, not corrupt the context.
#
# Field shape (GaelicThndr, v0.5.0): a starved box restores a ~56k-token bank
# from the disk KV tier, the cont admission is then rejected on the comp-cache
# budget, the request falls back to the serial lane — and the box dies with a
# deterministic illegal memory access.  No release gate exercised the reject
# half of that cycle before this one: every restore gate went restore->admit.
#
#   leg store  (-c 131072, stock budget): deep turn-1 (~56k tok) commits a
#       bank, SIGTERM persists it to the disk tier (STAMP law: the turn must
#       EOS-finish, driver retries length-draws).
#   leg cycle  (DS4_BATCH_VMM_BUDGET_MB=192 pins the budget below the record's
#       resident footprint; DS4_SERVER_FORK=0 forces the field's in-place
#       shape): turn-2 byte-extends the record.  Asserts, in order: restore
#       hit -> cont admit rejected on comp-cache budget -> serial "prompt
#       start" -> request completes with the exact reply.  Zero illegal-access
#       or CUDA-failure lines, server alive at the end.
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, kv dir removed.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
KVDIR=${DS4_BOX_SCRATCH:?}/.ds4/server-kv-rrgate
SRV=/tmp/restore_reject_gate.log
DRV=${DS4_BOX_SCRATCH:?}/rrgate_driver.py
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; exit 0" 2>/dev/null; }
cleanup(){ kill_all; ssh "$R" "rm -rf $KVDIR ${DS4_BOX_SCRATCH:?}/rrgate_state.json $DRV; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; ssh "$R" "tail -6 $SRV" 2>/dev/null; cleanup; exit 1; }

ssh "$R" "cat > $DRV" <<'PYEOF'
import json, sys, time, urllib.request
PORT, STATE = 8000, sys.argv[2]   # the state file path comes from the gate (rig-relative)
def gen_doc(nwords=23000):
    V = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel",
         "india","juliet","kilo","lima","mike","november","oscar","papa"]
    out = []
    for i in range(nwords):
        if i % 1000 == 0: out.append("section%d:" % (i // 1000))
        out.append(V[(i * 7 + i // 13) % len(V)] + str(i % 97))
    return " ".join(out)
def post(messages, mt=200):
    body = json.dumps({"model": "ds4", "messages": messages,
                       "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        j = json.loads(r.read())
    ch = j["choices"][0]
    print("finish=%s usage=%s" % (ch["finish_reason"], json.dumps(j.get("usage"))))
    print("content=%r" % (ch["message"].get("content") or "")[:80])
    sys.stdout.flush()
    return j
def t1(doc):
    return [{"role": "user", "content": doc +
             "\n\nIgnore the word list above completely. Reply with exactly: OK"}]
if sys.argv[1] == "store":
    doc = gen_doc()
    for _ in range(3):
        j = post(t1(doc))
        if j["choices"][0]["finish_reason"] == "stop": break
        print("RETRY: length draw")
    else:
        print("STORE-FAIL"); sys.exit(1)
    json.dump({"doc": doc, "reply": j["choices"][0]["message"]["content"]}, open(STATE, "w"))
    print("STORE-DONE")
else:
    st = json.load(open(STATE))
    j = post(t1(st["doc"]) + [{"role": "assistant", "content": st["reply"]},
                              {"role": "user", "content": "Now reply with exactly: DONE"}])
    if "DONE" not in (j["choices"][0]["message"].get("content") or ""):
        print("CYCLE-FAIL: wrong reply"); sys.exit(1)
    print("CYCLE-DONE")
PYEOF

boot(){ # $1 = extra env
  ssh "$R" ": > $SRV; cd $BINDIR; env $1 DS4_SERVER_COALESCE_MAX=2 setsid nohup ./$BIN -c 131072 --kv-disk-dir $KVDIR --kv-disk-space-mb 32768 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 12; n=$((n+12)); [ $n -ge 900 ] && fail "boot timeout"
  done
}
run_drv(){ # $1 = mode, $2 = done marker
  ssh "$R" ": > /tmp/rrgate_drv.out; setsid nohup python3 $DRV $1 ${DS4_BOX_SCRATCH:?}/rrgate_state.json > /tmp/rrgate_drv.out 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q '$2\|FAIL\|Traceback' /tmp/rrgate_drv.out" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "server died during $1"
    sleep 10; n=$((n+10)); [ $n -ge 600 ] && fail "$1 timeout"
  done
  ssh "$R" "grep -q '$2' /tmp/rrgate_drv.out" || fail "$1: $(ssh "$R" "tail -3 /tmp/rrgate_drv.out" | tr '\n' ' ')"
}

log "== leg store =="
kill_all
ssh "$R" "rm -rf $KVDIR ${DS4_BOX_SCRATCH:?}/rrgate_state.json; mkdir -p $KVDIR"
boot ""
run_drv store STORE-DONE
kill_all; sleep 6
ssh "$R" "grep -q 'bank persisted.*bank-shutdown' $SRV" || fail "no shutdown persist"
tok=$(ssh "$R" "grep 'bank persisted.*bank-shutdown' $SRV | sed 's/.*tokens=\([0-9]*\).*/\1/' | tail -1")
[ "$tok" -ge 50000 ] || fail "record too shallow: $tok"
log "store PASS (record $tok tokens)"

log "== leg cycle (pinned budget, in-place) =="
boot "DS4_BATCH_VMM_BUDGET_MB=192 DS4_SERVER_FORK=0"
run_drv cycle CYCLE-DONE
# Engagement, in order: restore -> reject -> serial.  Then zero corruption.
ssh "$R" "grep -q 'kv cache bank restore hit' $SRV" || fail "restore never fired"
ssh "$R" "grep -q 'cont admit rejected on comp-cache budget' $SRV" || fail "budget reject never fired"
ssh "$R" "grep -q 'prompt start' $SRV" || fail "serial fallback never engaged"
r=$(ssh "$R" "grep -n 'restore hit\|rejected on comp-cache\|prompt start' $SRV | head -3 | cut -d: -f1 | tr '\n' ' '")
set -- $r
{ [ "$1" -lt "$2" ] && [ "$2" -lt "$3" ]; } || fail "cycle order wrong: lines $r"
ssh "$R" "grep -qi 'illegal memory access\|CUDA.*failed\|cuBLAS.*failed' $SRV" && fail "corruption line present"
ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" || fail "server not alive at end"
log "cycle PASS (restore -> reject -> serial served clean)"
cleanup
log "restore_reject_gate PASS"
