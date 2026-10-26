#!/bin/bash
# gates/legs/serving/finish_reason.sh — #13: budget-cut mid-toolcall honesty.
#
# Field: a max_tokens cut that lands inside a tool-call emission reported
# finish_reason "error" (agent harnesses retried/aborted); the honest finish
# is "length".  Fix: the unterminated-toolcall path preserves a length stop,
# returns the partial call as assistant text, and SKIPS the tool-error
# recovery continuation (which would decode past the exhausted budget).
#
# Legs (greedy; tools batch is greedy-only):
#   A control: ample budget            -> finish_reason=tool_calls, calls>=1;
#              records C = completion_tokens (deterministic per boot).
#   B cut:     max_tokens = C-4        -> finish_reason=length,
#              completion_tokens==C-4, content non-empty, NO tool_calls,
#              server log gains 'tool call cut by token budget' (engagement)
#              and gains NO 'tool-error continuation appended' (recovery
#              skipped = budget honesty).
#   C stream:  same cut, stream:true   -> final chunk finish_reason=length,
#              same log deltas.
#
# Runs FROM the Mac over SSH. End state: ds4-server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
SRV=/tmp/finish_reason_gate_srv.log
OUT=${OUT:-/tmp/finish_reason_gate_$$}
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null; exit 1; }
srv_count(){ local c; c=$(ssh "$R" "grep -c \"$1\" $SRV 2>/dev/null || true" 2>/dev/null | tail -1); echo "${c:-0}"; }

wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable never reached 100G"; sleep 5
  done }

log "boot: killing old $BIN on $R"
ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
wait_mem
ssh "$R" ": > $SRV; cd $BINDIR; setsid nohup ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
  sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
done
curl -s -m 5 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1 || {
  ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true; sleep 2
  curl -s -m 10 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null || fail "tunnel"
}
log "boot up"

body(){ # $1=max_tokens $2=stream(1/0)
  python3 - "$1" "$2" <<'PY'
import json, sys
print(json.dumps({
  "messages": [{"role": "user", "content":
    "Look up the current weather in Paris with the get_weather tool. "
    "You must call the tool."}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {
      "city": {"type": "string"}}, "required": ["city"]}}}],
  "temperature": 0,
  "reasoning_effort": "off",
  "max_tokens": int(sys.argv[1]),
  "stream": sys.argv[2] == "1",
}))
PY
}

jget(){ python3 - "$1" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
cur = d
for k in sys.argv[2].split("."):
    cur = cur[int(k)] if isinstance(cur, list) else cur.get(k)
    if cur is None: break
print("" if cur is None else cur)
PY
}

log "== leg A: control (max_tokens 600) =="
body 600 0 > "$OUT/a_body.json"
curl -s -m 300 "http://127.0.0.1:$TUNNEL/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @"$OUT/a_body.json" > "$OUT/a.json" \
  || fail "leg A request"
A_FINISH=$(jget "$OUT/a.json" "choices.0.finish_reason")
A_COMP=$(jget "$OUT/a.json" "usage.completion_tokens")
A_NCALLS=$(python3 -c "import json,sys; d=json.load(open('$OUT/a.json')); print(len(d['choices'][0]['message'].get('tool_calls') or []))")
log "A finish=$A_FINISH completion=$A_COMP calls=$A_NCALLS"
[ "$A_FINISH" = "tool_calls" ] || fail "A finish=$A_FINISH (want tool_calls)"
[ "$A_NCALLS" -ge 1 ] || fail "A no tool calls parsed"
[ -n "$A_COMP" ] && [ "$A_COMP" -gt 8 ] || fail "A completion_tokens=$A_COMP unusable"

# The cont path is NOT run-to-run deterministic at temp 0 (capture-arc
# law: width-regime FP noise moves emissions by a few tokens), so a
# fixed A-derived cut can land past a shorter re-emission (first run:
# A=74, cut=70, B re-emitted 69 and finished tool_calls honestly).
# Adaptive: walk the cut down using each observed emission until the
# ENGAGEMENT LINE (the only true mid-call oracle) fires.
CUT=$((A_COMP - 6))
B_OK=0
for attempt in 1 2 3; do
  [ "$CUT" -ge 8 ] || fail "cut walked below 8 without landing mid-call"
  log "== leg B attempt $attempt: cut (max_tokens $CUT) =="
  ENG0=$(srv_count "tool call cut by token budget")
  REC0=$(srv_count "tool-error continuation appended")
  body "$CUT" 0 > "$OUT/b_body.json"
  curl -s -m 300 "http://127.0.0.1:$TUNNEL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$OUT/b_body.json" > "$OUT/b.json" \
    || fail "leg B request"
  B_FINISH=$(jget "$OUT/b.json" "choices.0.finish_reason")
  B_COMP=$(jget "$OUT/b.json" "usage.completion_tokens")
  B_NCALLS=$(python3 -c "import json,sys; d=json.load(open('$OUT/b.json')); print(len(d['choices'][0]['message'].get('tool_calls') or []))")
  B_CLEN=$(python3 -c "import json,sys; d=json.load(open('$OUT/b.json')); print(len(d['choices'][0]['message'].get('content') or ''))")
  ENG1=$(srv_count "tool call cut by token budget")
  REC1=$(srv_count "tool-error continuation appended")
  log "B finish=$B_FINISH completion=$B_COMP calls=$B_NCALLS content_len=$B_CLEN eng=+$((ENG1-ENG0)) rec=+$((REC1-REC0))"
  if [ $((ENG1-ENG0)) -ge 1 ]; then
    [ "$B_FINISH" = "length" ] || fail "B finish=$B_FINISH (want length)"
    [ "$B_COMP" = "$CUT" ] || fail "B completion=$B_COMP != cut budget $CUT"
    [ "$B_NCALLS" -eq 0 ] || fail "B parsed tool calls from a cut emission"
    [ "$B_CLEN" -gt 0 ] || fail "B empty content (partial call text lost)"
    [ $((REC1-REC0)) -eq 0 ] || fail "B recovery continuation ran past the budget"
    B_OK=1
    break
  fi
  [ "$B_FINISH" = "tool_calls" ] || fail "B attempt $attempt: no engagement yet finish=$B_FINISH (want tool_calls while walking)"
  CUT=$((B_COMP - 6))   # this run's emission was shorter: cut below IT
done
[ "$B_OK" = 1 ] || fail "B never landed mid-call in 3 attempts"

C_OK=0
for attempt in 1 2; do
  log "== leg C attempt $attempt: stream cut (max_tokens $CUT) =="
  ENG0=$(srv_count "tool call cut by token budget")
  REC0=$(srv_count "tool-error continuation appended")
  body "$CUT" 1 > "$OUT/c_body.json"
  curl -s -N -m 300 "http://127.0.0.1:$TUNNEL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$OUT/c_body.json" > "$OUT/c.sse" \
    || fail "leg C request"
  C_FINISH=$(grep -oE '"finish_reason":"[a-z_]+"' "$OUT/c.sse" | tail -1 | cut -d'"' -f4)
  ENG1=$(srv_count "tool call cut by token budget")
  REC1=$(srv_count "tool-error continuation appended")
  log "C finish=$C_FINISH eng=+$((ENG1-ENG0)) rec=+$((REC1-REC0))"
  if [ $((ENG1-ENG0)) -ge 1 ]; then
    [ "$C_FINISH" = "length" ] || fail "C stream finish=$C_FINISH (want length)"
    [ $((REC1-REC0)) -eq 0 ] || fail "C recovery continuation ran past the budget"
    C_OK=1
    break
  fi
  CUT=$((CUT - 4))   # stream run jittered shorter: cut below it
done
[ "$C_OK" = 1 ] || fail "C never landed mid-call in 2 attempts"

ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null
log "FINISH-REASON-GATE PASS (A tool_calls/$A_COMP, cut=$CUT -> length, stream length, recovery skipped)"
