#!/bin/bash
# gates/legs/serving/completion_stream.sh — Inc 0b: legacy /v1/completions
# streaming on the CONTINUOUS lane emits the legacy shape.
#
# The cont lane used to project completion streams through the chat delta
# machine (chat.completion.chunk objects with delta framing on a
# text_completion route — the Inc 0a negative fixture, recorded in
# docs/ds4-api-surface-matrix.md).  This gate boots zero-config, streams one
# completion, and asserts:
#
#   schema:     every SSE data record is "object":"text_completion" with
#               choices[].text framing; no chat.completion.chunk, no
#               "delta"; final chunk carries a finish_reason; [DONE] last.
#   engagement: ds4_route_requests_total{surface="openai_completion",
#               lane="continuous"} advanced — the request PROVABLY rode the
#               cont lane (a serial-routed run would pass schema vacuously;
#               the lane-specific-oracle law from the Inc 0a receipts).
#   buffered:   a non-stream completion still returns text_completion.
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
CTX=${CTX:-16384}
SRV=/tmp/completion_stream_gate_srv.log
OUT=${OUT:-/tmp/completion_stream_gate_$$}
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null; exit 1; }

wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable never reached 100G"; sleep 5
  done }

route_count(){ # surface lane
  ssh "$R" "curl -s http://127.0.0.1:$PORT/metrics" | \
    grep "ds4_route_requests_total{surface=\"$1\",lane=\"$2\"}" | awk '{print $2}'
}

log "boot: killing old $BIN on $R"
ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
wait_mem
ssh "$R" ": > $SRV; cd $BINDIR; setsid nohup ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
  sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
done
log "boot up"

log "== leg A: streaming completion (schema + cont engagement) =="
C0=$(route_count openai_completion continuous)
ssh "$R" "curl -s -m 120 http://127.0.0.1:$PORT/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"prompt\":\"The first three prime numbers are\",\"max_tokens\":24,\"stream\":true}'" \
  > "$OUT/a.sse" || fail "leg A request"
C1=$(route_count openai_completion continuous)
python3 - "$OUT/a.sse" <<'PY' || fail "leg A schema"
import json, sys
recs = []
for chunk in open(sys.argv[1]).read().split("\n\n"):
    chunk = chunk.strip()
    if not chunk.startswith("data: "): continue
    recs.append(chunk[6:])
assert recs, "no SSE records"
assert recs[-1] == "[DONE]", f"last record {recs[-1]!r} != [DONE]"
text = ""
finishes = []
for r in recs[:-1]:
    d = json.loads(r)
    assert d["object"] == "text_completion", d["object"]
    c = d["choices"][0]
    assert "delta" not in c, "chat delta framing on a completion stream"
    text += c.get("text") or ""
    if c.get("finish_reason"): finishes.append(c["finish_reason"])
assert text.strip(), "empty reassembled text"
assert len(finishes) == 1, f"finish_reason count {len(finishes)}"
print(f"schema OK: {len(recs)-1} chunks, finish={finishes[0]}, text={text[:40]!r}")
PY
[ -n "$C0" ] && [ -n "$C1" ] && [ "$C1" -gt "$C0" ] || \
  fail "no cont engagement (openai_completion continuous $C0 -> $C1)"
log "A PASS (cont engagement $C0 -> $C1)"

log "== leg B: buffered completion regression =="
ssh "$R" "curl -s -m 120 http://127.0.0.1:$PORT/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"prompt\":\"The capital of France is\",\"max_tokens\":16}'" \
  > "$OUT/b.json" || fail "leg B request"
python3 - "$OUT/b.json" <<'PY' || fail "leg B schema"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["object"] == "text_completion", d["object"]
assert d["choices"][0].get("text"), "empty text"
print("buffered OK")
PY
log "B PASS"

ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null
log "ALL LEGS PASS — artifacts in $OUT ($BIN killed, $R left free)"
