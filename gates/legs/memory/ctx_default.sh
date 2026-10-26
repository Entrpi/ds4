#!/bin/bash
# gates/legs/memory/ctx_default.sh — gate for the CUDA default context
# change (32768 -> 262144, Metal/CPU keep 32768).  Two boots on .33:
#   A defaults: bare `ds4-server` (no -c) must plan ctx 262144, serve an
#     agent-shape request (max_tokens OMITTED — the old-default footgun),
#     and refuse prompt+budget > 262144 with a typed 400.
#   B override: `-c 32768` must be honored unchanged.
# Runs FROM the Mac.  Requires the .33 box lock to be OURS and the
# capacity probe driver to be gone (self-collision law: never run box
# experiments beside a live leg — a build alone can skew its decode
# probes).  End state: both engines killed by PID, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT_A=${PORT_A:-18011}
PORT_B=${PORT_B:-18012}
SRV_A=/tmp/ctxgate_a.log
SRV_B=/tmp/ctxgate_b.log
SKIP_BUILD=${SKIP_BUILD:-0}
PROBE_DRIVER_PID=${PROBE_DRIVER_PID:-37465}

# LAUNCH LAW (measured on .33, manifest-gate runs 1-5): daemonizing ssh
# must be PLAIN ssh + `setsid --fork` + a local alarm.  `& echo $!`
# hangs (wrapper bash in do_wait); ssh -o ServerAlive* options hang the
# launching client even with `& exit 0`.  PID capture is a SEPARATE
# call afterwards — here scoped by PORT via ss, never by process name,
# so this gate can never confuse another session's ds4-server.
SSH(){ perl -e 'alarm 90; exec @ARGV or die' ssh "$@"; }
SSHB(){ perl -e 'alarm 1200; exec @ARGV or die' ssh "$@"; }

log(){ echo "[$(date +%H:%M:%S)] $*"; }
PID_A=""; PID_B=""
cleanup(){
  [ -n "$PID_A" ] && SSH "$R" "kill $PID_A 2>/dev/null; exit 0"
  [ -n "$PID_B" ] && SSH "$R" "kill $PID_B 2>/dev/null; exit 0"
}
fail(){ log "FAIL: $*"; cleanup; exit 1; }

# --- guards ------------------------------------------------------------------
if ps -p "$PROBE_DRIVER_PID" >/dev/null 2>&1; then
  fail "capacity probe driver $PROBE_DRIVER_PID still running; gate must wait"
fi
SSH "$R" "test -f /tmp/ds4_box_lock/nonce" || \
  fail "no box lock on $R; take it before gating"

# --- sync + build ------------------------------------------------------------
if [ "$SKIP_BUILD" != "1" ]; then
  log "rsync tree -> $R:$BINDIR"
  rsync -a --files-from=<(git ls-files) . "$R:$BINDIR/" || fail "rsync"
  log "build: make -j6 cuda-spark (this is the only sanctioned .33 build)"
  SSHB "$R" "cd $BINDIR && make -j6 cuda-spark" >/tmp/ctxgate_build.log 2>&1 \
    || fail "build (see /tmp/ctxgate_build.log)"
  SSH "$R" "/usr/local/cuda/bin/cuobjdump $BINDIR/ds4-server 2>/dev/null | grep -m1 -o 'sm_121[a]*'" \
    | grep -q sm_121 || fail "binary is not native sm_121"
fi

# boot <port> <log> <extra-args...> — PID on stdout, diagnostics on
# stderr ONLY (the caller captures stdout; a log line in it becomes a
# garbage "PID" and cleanup kills nothing — run-3 lesson).  Readiness is
# the engine's own 'listening on http' line; there is deliberately NO
# death-grep: healthy boot lines contain "refused 0" (range plan) and a
# fuzzy pattern killed a good boot 2 s before it listened.  Real deaths
# surface as the timeout, with the log tail attached.
boot(){
  local port=$1 slog=$2; shift 2
  SSH "$R" "cd $BINDIR && rm -f $slog; setsid --fork nohup ./ds4-server \
      --port $port $* > $slog 2>&1 < /dev/null; exit 0" || { echo ""; return 1; }
  local n=0
  until SSH "$R" "grep -q 'listening on http' $slog" 2>/dev/null; do
    n=$((n+1))
    if [ $n -gt 120 ]; then
      echo "boot timeout on :$port: $(SSH "$R" "tail -3 $slog" | tr '\n' ' ')" >&2
      echo ""; return 1
    fi
    sleep 5
  done
  SSH "$R" "ss -ltnp 2>/dev/null | grep ':$port ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2"
}
numeric(){ case "$1" in (''|*[!0-9]*) return 1;; (*) return 0;; esac; }

req(){ # req <port> <json>; line 1 = HTTP code, rest = body
  SSH "$R" "curl -s -m 120 -o /tmp/ctxgate_body -w '%{http_code}\n' \
      -H 'Content-Type: application/json' \
      -d '$2' http://127.0.0.1:$1/v1/chat/completions; cat /tmp/ctxgate_body" 2>/dev/null
}

# --- leg A: defaults boot ----------------------------------------------------
log "leg A: bare ds4-server (no -c) on :$PORT_A"
PID_A=$(boot "$PORT_A" "$SRV_A")
numeric "$PID_A" || fail "leg A: no numeric pid on :$PORT_A (got '$PID_A')"
SSH "$R" "grep -q 262144 $SRV_A" || fail "leg A: boot log never mentions ctx 262144"
log "leg A: ctx 262144 in boot log (pid $PID_A)"
SSH "$R" "grep -iE 'coalesce|bank' $SRV_A | head -3" || true

AGENT_SHAPE='{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Reply with the single word ready."}]}'
OUT=$(req "$PORT_A" "$AGENT_SHAPE")
CODE=$(printf '%s\n' "$OUT" | head -1)
[ "$CODE" = "200" ] || fail "leg A: agent-shape request (max_tokens omitted) got HTTP $CODE — the footgun this change removes"
log "leg A: agent-shape request (no max_tokens) served: HTTP 200"

# Designed semantics, both asserted: an oversized max_tokens CLAMPS
# (budget cuts report `length`, never an error — v0.5.x); an oversized
# PROMPT gets the typed 400 (request_exceeds_context, prompt.len >= ctx).
OVER_BUDGET='{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":262144}'
OUT=$(req "$PORT_A" "$OVER_BUDGET")
CODE=$(printf '%s\n' "$OUT" | head -1)
[ "$CODE" = "200" ] || fail "leg A: oversized max_tokens should clamp and serve, got HTTP $CODE"
log "leg A: oversized max_tokens clamped and served (designed)"

SSH "$R" "python3 - <<'PYEOF'
import json
body={'model':'deepseek-v4-flash','messages':[{'role':'user','content':'word '*700000}],'max_tokens':8}
open('/tmp/ctxgate_over.json','w').write(json.dumps(body))
PYEOF" || fail "leg A: over-prompt build"
OUT=$(SSH "$R" "curl -s -m 60 -o /tmp/ctxgate_body -w '%{http_code}\n' -H 'Content-Type: application/json' \
    -d @/tmp/ctxgate_over.json http://127.0.0.1:$PORT_A/v1/chat/completions; cat /tmp/ctxgate_body")
CODE=$(printf '%s\n' "$OUT" | head -1)
[ "$CODE" = "400" ] || fail "leg A: over-context prompt got HTTP $CODE, want typed 400"
echo "$OUT" | grep -qi "context" || fail "leg A: 400 body does not name context"
log "leg A: over-context prompt typed-400 as designed"

# The engine refuses to start beside a live ds4 process (single-instance
# lock), so leg B must WAIT for leg A's teardown, not just fire the kill
# (run-4 lesson: a ~100 GiB process takes seconds to exit).
SSH "$R" "kill $PID_A 2>/dev/null; exit 0"
n=0
while SSH "$R" "kill -0 $PID_A 2>/dev/null"; do
  n=$((n+1)); [ $n -gt 24 ] && fail "leg A engine $PID_A did not exit after kill"
  sleep 5
done
PID_A=""

# --- leg B: explicit -c override --------------------------------------------
log "leg B: ds4-server -c 32768 on :$PORT_B"
PID_B=$(boot "$PORT_B" "$SRV_B" "-c 32768")
numeric "$PID_B" || fail "leg B: no numeric pid on :$PORT_B (got '$PID_B')"
SSH "$R" "grep -q 32768 $SRV_B" || fail "leg B: boot log never mentions ctx 32768"
SSH "$R" "grep -q 262144 $SRV_B" && fail "leg B: default leaked into an explicit -c 32768 boot"
SMALL='{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Reply with the single word ready."}],"max_tokens":8}'
OUT=$(req "$PORT_B" "$SMALL")
CODE=$(printf '%s\n' "$OUT" | head -1)
[ "$CODE" = "200" ] || fail "leg B: smoke request got HTTP $CODE"
log "leg B: override honored, smoke served"
SSH "$R" "kill $PID_B 2>/dev/null; exit 0"; PID_B=""

log "ALL LEGS PASS: CUDA default ctx 262144 planned, agent shape fits a defaults boot, ceiling typed-400s, -c override intact"
