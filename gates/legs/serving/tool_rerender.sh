#!/bin/bash
# gates/legs/serving/tool_rerender.sh — #11 / N5: tools re-render asymmetry
# MEASUREMENT (docket item 11; ledger 08-03 "committed +23 vs re-render
# +13" with the FORK=0 confound noted -- this is the clean instrument).
#
# Question: after a tool-call turn on the CONT path, does the next
# request's re-rendered prompt still byte-align with the bank's committed
# record?  The exact-DSML replay machinery (tool_memory: raw sampled DSML
# keyed by call id, spliced at render time) exists to make this true; if
# it misses, the canonical render diverges at the tool boundary and agent
# tool traffic partial-forks (or goes cold) on EVERY turn.
#
# Drive (both legs, reasoning off so token arithmetic is exact):
#   T1 chat+tools (stream, temp 0) -> tool_calls response.
#     committed K = prompt_tokens + completion_tokens.
#   T2 [user, assistant{content,tool_calls}, tool{result}] -> usage
#     cached_tokens CT2.  ASYMMETRY = K - CT2 (boundary-group alignment
#     makes small positive values benign; a value ~= completion_tokens
#     means the whole assistant turn re-paid; ~= K means cold).
#
# Legs:
#   replay   (default boot)                       -> expect CT2 ~= K
#   canon    (--disable-exact-dsml-tool-replay)   -> the canonical floor;
#            the delta between legs = what exact replay saves on cont
#   restart  (RUN_RESTART=1, optional): kill + reboot between T1 and T2
#            (disk kv on) -> does tool-memory restore keep alignment
#            across a restart?
#
# This is a MEASUREMENT gate: it asserts drive mechanics (tool_calls
# finish, 200s) and PRINTS the verdict numbers for the ledger; alignment
# itself is reported, not asserted (the fix increment sets the bar).
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
SRV=/tmp/tool_rerender_gate.log
OUT=${OUT:-/tmp/tool_rerender_gate_$$}
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; ssh "$R" "tail -4 $SRV" 2>/dev/null; kill_all; exit 1; }

wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable never reached 100G"; sleep 5
  done }

boot(){ # $1 = extra args
  kill_all; wait_mem
  ssh "$R" ": > $SRV; cd $BINDIR; DS4_SERVER_COALESCE_MAX=4 setsid nohup ./$BIN -c $CTX $1 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
  curl -s -m 5 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1 || {
    ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true; sleep 2
    curl -s -m 10 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null || fail "tunnel"
  }
}

t1_body(){ python3 - <<'PY'
import json
pad = ("context block: alpha bravo charlie delta echo foxtrot golf hotel " * 60).strip()
print(json.dumps({"messages": [{"role": "user", "content":
  pad + "\nLook up the weather in Paris with the get_weather tool. "
        "You must call the tool."}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {
      "city": {"type": "string"}}, "required": ["city"]}}}],
  "temperature": 0, "max_tokens": 400, "reasoning_effort": "off",
  "stream": False}))
PY
}

t2_body(){ # $1 = T1 response json
  python3 - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
msg = r["choices"][0]["message"]
pad = ("context block: alpha bravo charlie delta echo foxtrot golf hotel " * 60).strip()
asst = {"role": "assistant", "content": msg.get("content") or None}
if msg.get("tool_calls"): asst["tool_calls"] = msg["tool_calls"]
tid = msg["tool_calls"][0]["id"]
print(json.dumps({"messages": [
  {"role": "user", "content":
    pad + "\nLook up the weather in Paris with the get_weather tool. "
          "You must call the tool."},
  asst,
  {"role": "tool", "tool_call_id": tid,
   "content": "{\"city\":\"Paris\",\"temp_c\":18,\"sky\":\"overcast\"}"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {
      "city": {"type": "string"}}, "required": ["city"]}}}],
  "temperature": 0, "max_tokens": 200, "reasoning_effort": "off",
  "stream": False}))
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

drive(){ # $1 = leg tag, $2 = restart(1/0)
  t1_body > "$OUT/$1_t1.json"
  curl -s -m 300 "http://127.0.0.1:$TUNNEL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$OUT/$1_t1.json" > "$OUT/$1_r1.json" \
    || fail "$1 T1 request"
  local fin p1 c1
  fin=$(jget "$OUT/$1_r1.json" "choices.0.finish_reason")
  [ "$fin" = "tool_calls" ] || fail "$1 T1 finish=$fin (want tool_calls)"
  p1=$(jget "$OUT/$1_r1.json" "usage.prompt_tokens")
  c1=$(jget "$OUT/$1_r1.json" "usage.completion_tokens")
  local K=$((p1 + c1))

  if [ "$2" = "1" ]; then
    log "$1: restarting the server between T1 and T2 (disk-kv tool-memory leg)"
    boot "$LEGARGS"
  fi

  MARK=$(ssh "$R" "wc -l < $SRV")
  t2_body "$OUT/$1_r1.json" > "$OUT/$1_t2.json"
  local http
  http=$(curl -s -o "$OUT/$1_r2.json" -w '%{http_code}' -m 300 \
    "http://127.0.0.1:$TUNNEL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$OUT/$1_t2.json")
  [ "$http" = "200" ] || fail "$1 T2 got $http"
  local p2 ct2
  p2=$(jget "$OUT/$1_r2.json" "usage.prompt_tokens")
  ct2=$(jget "$OUT/$1_r2.json" "usage.prompt_tokens_details.cached_tokens")
  [ -n "$ct2" ] || ct2=0
  local admit
  admit=$(ssh "$R" "tail -n +$((MARK+1)) $SRV" | grep -E "warm admit|fork admit|partial (fork|truncate) admit|prompt start" | tail -2 | tr '\n' ';')
  log "$1 MEASUREMENT: T1 committed K=$K (prompt=$p1 gen=$c1); T2 prompt=$p2 cached=$ct2; ASYMMETRY=$((K - ct2)) tokens; admit: ${admit:-none}"
}

log "== leg replay (exact DSML replay ON, default) =="
LEGARGS=""
boot "$LEGARGS"
drive replay 0

log "== leg canon (--disable-exact-dsml-tool-replay) =="
LEGARGS="--disable-exact-dsml-tool-replay"
boot "$LEGARGS"
drive canon 0

if [ "${RUN_RESTART:-0}" = "1" ]; then
  log "== leg restart (replay ON, server restart between turns) =="
  LEGARGS=""
  boot "$LEGARGS"
  drive restart 1
fi

kill_all
log "TOOL-RERENDER-GATE DONE (measurement; copy the MEASUREMENT lines to the ledger)"
