#!/bin/bash
# serving/registry_close.sh — v0.5.6 API Inc 5 CLOSE gate: the plan §5/§6.2 gate-line
# items not already covered by serving/cont_registry.sh (which owns T1->T2 per
# protocol, full-history T2, worker-race 409, retention, the terminal
# transaction, and the reservation rollback).
#
# Boot A (policy-zero) legs:
#   interleave   plan §6.2 "A T1 -> B T1 -> A T2 -> B T2" on the ONE serial
#                session: B T1 supersedes A's record (published +1, demoted
#                +1), so A's output-only T2 refuses 400 while A's
#                FULL-HISTORY T2 serves 200 (exact-DSML replay); that replay
#                is itself a serial win that demotes B's record, so B's
#                output-only T2 refuses and B's full-history T2 serves.
#                Honest refusal + total replay recovery, interleaved live --
#                and A's ids never resolve against B's LIVE record (the
#                record-level bleed check).
#   crossproto   an Anthropic toolu_ id sent as a Responses
#                function_call_output -> 400 (the (protocol, call_id) index
#                never resolves across namespaces; §4.6 collision/bleed,
#                live parse-side)
#
# Boot B + C (restart/full-replay):
#   restart_t1     boot B: T1 tool turn publishes (records_live 1)
#   restart_replay boot C (a FULL server restart -- registry and tool
#                  memory empty): the pre-restart output-only T2 refuses
#                  400; the FULL-HISTORY T2 (client-rendered tool_use
#                  replay, exact DSML gone with the restart) serves 200 --
#                  plan §6.2 "A full replay after restart (stateless
#                  success)"
#
# Runs FROM the Mac over SSH like the other gates.  Boots kill any running
# ds4-server on $R.  End state: ds4-server killed, box left free.
#
# Env overrides: R (${DS4_BOX33:?}) BINDIR (${DS4_BOX_TREE:?})
#                PORT (8000) TUNNEL_PORT (18000) CTX (16384)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/inc5_close_gate
OUT=${OUT:-/tmp/inc5_close_gate_$$}
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

wait_mem(){ # $1=min MemAvailable GiB
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
  ssh "$R" ": > $SRV; cd $BINDIR; env ${BOOT_ENV:-} setsid nohup ./ds4-server -c $CTX --port $PORT \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
      sleep 3
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" 2>/dev/null | tr '\n' ' ')"
    fi
    sleep 10; n=$((n+10)); [ $n -ge 1200 ] && fail "boot timeout"
  done
  tunnel_up || fail "tunnel :$TUNNEL_PORT unreachable"
  log "boot: up"
}

post(){
  curl -s -m 240 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
post_file(){
  curl -s -m 240 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d @"$3"
}
has(){ grep -q "$2" "$OUT/$1.json" || fail "$1: missing [$2] in $(head -c 300 "$OUT/$1.json")"; }
code_is(){ [ "$2" = "$3" ] || fail "$1: HTTP $2, want $3 ($(head -c 300 "$OUT/$1.json"))"; }
# METRICS-HELPER TRAP: extract the number BEFORE head -1.
m(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$' | head -1; }

TOOLS_ANTH='[{"name":"list_files","description":"List the files in a directory","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]'
TOOLS_RESP='[{"type":"function","name":"list_files","description":"List the files in a directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]'

anth_t1(){ # $1=name $2=path-arg -> sets T1_ID
  local c
  c=$(post "$1" /v1/messages '{"model":"m","max_tokens":1200,"temperature":0,"messages":[{"role":"user","content":"Use the list_files tool to list the files in '"$2"'. Call the tool."}],"tools":'"$TOOLS_ANTH"'}')
  code_is "$1" "$c" 200
  has "$1" '"type":"tool_use"'
  T1_ID=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1])); print(next(b["id"] for b in t["content"] if b.get("type")=="tool_use"))' "$OUT/$1.json") || fail "$1: no tool_use id"
}
t2_out_body(){ # $1=id
  echo '{"model":"m","max_tokens":1200,"temperature":0,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"'"$1"'","content":"file1.txt\nfile2.txt"}]}],"tools":'"$TOOLS_ANTH"'}'
}
build_full_replay(){ # $1=t1-json $2=id $3=path-arg -> $OUT/replay_$2.json
  python3 - "$OUT/$1.json" "$2" "$3" > "$OUT/replay_$2.json" <<'PY' || fail "replay build failed for $2"
import json, sys
t1 = json.load(open(sys.argv[1])); tid = sys.argv[2]; path = sys.argv[3]
msgs = [
    {"role": "user", "content": f"Use the list_files tool to list the files in {path}. Call the tool."},
    {"role": "assistant", "content": [b for b in t1["content"] if b.get("type") in ("text", "tool_use")]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": "file1.txt\nfile2.txt"}]},
]
tools = [{"name": "list_files", "description": "List the files in a directory",
          "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
print(json.dumps({"model": "m", "max_tokens": 1200, "temperature": 0, "messages": msgs, "tools": tools}))
PY
}

BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_CONT_GRACE_S=0 DS4_CONT_PIN_DEADLINE_S=0" boot

# ==================== interleave: A T1 -> B T1 -> A T2 -> B T2 ==============
anth_t1 il_a_t1 /tmp;  A_ID=$T1_ID
pubi0=$(m ds4_continuation_records_published_total)
demi0=$(m ds4_continuation_demoted_total)
anth_t1 il_b_t1 /home; B_ID=$T1_ID
[ "$A_ID" != "$B_ID" ] || fail "interleave: A and B minted the same id"
pubi1=$(m ds4_continuation_records_published_total)
demi1=$(m ds4_continuation_demoted_total)
[ "${pubi1:-0}" -gt "${pubi0:-0}" ] || fail "interleave: B T1 never published (${pubi0:-?} -> ${pubi1:-?})"
[ "${demi1:-0}" -gt "${demi0:-0}" ] || fail "interleave: B T1 never superseded A's record (demoted ${demi0:-?} -> ${demi1:-?})"
[ "$(m ds4_continuation_records_live)" -eq 1 ] || fail "interleave: want exactly one LIVE record after B T1"

# A's output-only T2: B owns the session tip; A's ids must NOT resolve
# against B's LIVE record -- honest 400, never a wrong-frontier answer.
c=$(post il_a_t2_out /v1/messages "$(t2_out_body "$A_ID")")
code_is il_a_t2_out "$c" 400
has il_a_t2_out 'continuation state is not available'
# A's full-history T2 recovers statelessly (exact DSML still in the
# REPLAY_ONLY record) -- and that serial win demotes B's LIVE record.
build_full_replay il_a_t1 "$A_ID" /tmp
c=$(post_file il_a_t2_replay /v1/messages "$OUT/replay_$A_ID.json")
code_is il_a_t2_replay "$c" 200
has il_a_t2_replay '"role":"assistant"'
[ "$(m ds4_continuation_records_live)" -le 1 ] || fail "interleave: more than one LIVE record after A's replay"
# B's output-only T2 now refuses (A's replay superseded); B's full replay serves.
c=$(post il_b_t2_out /v1/messages "$(t2_out_body "$B_ID")")
[ "$c" = 400 ] || [ "$c" = 409 ] || fail "interleave: B T2 output-only got HTTP $c, want 400/409"
build_full_replay il_b_t1 "$B_ID" /home
c=$(post_file il_b_t2_replay /v1/messages "$OUT/replay_$B_ID.json")
code_is il_b_t2_replay "$c" 200
log "interleave PASS (A T1 -> B T1 -> A T2 400/replay 200 -> B T2 400/replay 200; one LIVE tip throughout)"

# ==================== crossproto: anthropic id in a Responses turn ==========
anth_t1 cp_t1 /var; CP_ID=$T1_ID
c=$(post crossproto /v1/responses '{"model":"m","max_output_tokens":1200,"temperature":0,"input":[{"type":"function_call_output","call_id":"'"$CP_ID"'","output":"file1.txt"}],"tools":'"$TOOLS_RESP"'}')
code_is crossproto "$c" 400
has crossproto 'retry by replaying the full input history'
log "crossproto PASS (anthropic toolu_ id never resolves in the Responses namespace)"

# ==================== restart/full-replay ===================================
BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_CONT_GRACE_S=0 DS4_CONT_PIN_DEADLINE_S=0" boot
anth_t1 restart_t1 /etc; R_ID=$T1_ID
[ "$(m ds4_continuation_records_live)" -eq 1 ] || fail "restart_t1: no LIVE record"
build_full_replay restart_t1 "$R_ID" /etc   # build BEFORE the restart
log "restart_t1 PASS (record live; restarting the server)"

BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_CONT_GRACE_S=0 DS4_CONT_PIN_DEADLINE_S=0" boot
[ "$(m ds4_continuation_records_live)" -eq 0 ] || fail "restart: registry not empty after restart"
c=$(post restart_t2_out /v1/messages "$(t2_out_body "$R_ID")")
code_is restart_t2_out "$c" 400
has restart_t2_out 'continuation state is not available'
c=$(post_file restart_t2_replay /v1/messages "$OUT/replay_$R_ID.json")
code_is restart_t2_replay "$c" 200
has restart_t2_replay '"role":"assistant"'
log "restart_replay PASS (output-only refused 400 after restart; full-history replay served 200)"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT"
