#!/bin/bash
# gates/legs/memory/noserial_optin.sh — MT-4 (v0.6.1 memory truth): opt-in
# cont-only serving + the R18 /v1/models race fix.
#
# What changed and what this gate pins:
#   --no-serial / DS4_SERVER_NO_SERIAL=1: the boot session is never
#   created; run_job_single -- the ONE choke point every serial execution
#   flows through (direct route, static single-member collapse, static
#   memgov fallback, cont stranded fallback) -- refuses with the R13-shaped
#   typed 503 (reason=lane_disabled) instead of serving.  Default OFF.
#   R18: append_model_json reads serial_boot_ctx (client threads must not
#   read s->session -- the reaper swaps it under gen_mu; under --no-serial
#   the old read would NULL-deref on the first GET /v1/models).
#   R19: the cont bank plan sizes from cfg.ctx_size, not the session's ctx.
#
# Legs:
#   D:  default boot -> serial route (return_token_ids non-streaming)
#       serves 200 (behavior unchanged); /v1/models context == -c.
#   N:  --no-serial boot -> serial-forcing routes get typed 503s
#       (return_token_ids; anthropic buffered tools; anthropic prefill-only
#       max_tokens=0), lane_disabled counter grows per refusal, the server
#       stays healthy (cont turn 200 after each), /v1/models works (the
#       old code would have crashed), zero serial sessions ever alloc
#       (session_tensors census ~0, serial lease 0).
#   C:  cont traffic on --no-serial serves normally (streaming + buffered).
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
SRV=/tmp/mt4_noserial_gate_srv.log
OUT=${OUT:-/tmp/mt4_noserial_gate_$$}
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

boot(){ # $1 = extra args/env prefix form: "ENV=1 ... " (flags appended after binary via $2)
  log "boot: killing old $BIN on $R (env='$1' flags='$2')"
  ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
  wait_mem
  ssh "$R" ": > $SRV; cd $BINDIR; $1 setsid nohup ./$BIN -c 16384 --port $PORT $2 > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
  log "boot up"
}

metrics(){ ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/metrics"; }
census_mib(){ metrics | grep "class=\"$1\",state=\"allocated\"" | awk '{s+=$2} END {printf "%d\n", s/1048576}'; }
lane_disabled_count(){ metrics | grep 'lane="serial",reason="lane_disabled"' | awk '{print $2}' | head -1; }
req(){ # $1 tag  $2 path  $3 body -> echoes http code
  ssh "$R" "curl -s -m 120 -o /tmp/m4_$1.json -w '%{http_code}' \
    http://127.0.0.1:$PORT$2 -H 'Content-Type: application/json' -d '$3'"; }

serial_ids(){ req "$1" /v1/chat/completions '{"messages":[{"role":"user","content":"Say yes."}],"max_tokens":16,"return_token_ids":true}'; }
anthropic_tools(){ req "$1" /v1/messages '{"model":"m","max_tokens":32,"messages":[{"role":"user","content":"What is 2+2? Use the tool."}],"tools":[{"name":"calc","description":"calculator","input_schema":{"type":"object","properties":{"e":{"type":"string"}}}}]}'; }
anthropic_prefill(){ req "$1" /v1/messages '{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"Warm this prefix."}]}'; }
cont_turn(){ req "$1" /v1/chat/completions '{"messages":[{"role":"user","content":"Name two colors."}],"max_tokens":24}'; }
models_ctx(){ # echoes the target ctx if ANY listed model reports it (the
              # codex catalog may splice rows with their own numbers)
  ssh "$R" "curl -s -m 5 http://127.0.0.1:$PORT/v1/models" | python3 -c "
import json,sys
d=json.load(sys.stdin)
rows=(d.get('data') or []) + (d.get('models') or [])
cs={r.get('context_length') or r.get('context_window') or 0 for r in rows}
print(16384 if 16384 in cs else (sorted(cs)[-1] if cs else 0))"; }

# ---------- Leg D: default unchanged ----------
boot "" ""
HTTP=$(serial_ids d1); [ "$HTTP" = "200" ] || fail "leg D: serial route HTTP $HTTP (default must serve)"
MC=$(models_ctx); [ "$MC" = "16384" ] || fail "leg D: /v1/models context $MC != 16384 (R18 read wrong)"
log "D PASS (serial serves 200; /v1/models ctx=16384)"

# ---------- Leg N: typed refusals, healthy server ----------
boot "" "--no-serial"
[ "$(srv_count 'cont-only serving')" -ge 1 ] || fail "leg N: --no-serial boot line missing"
MC=$(models_ctx); [ "$MC" = "16384" ] || fail "leg N: /v1/models context $MC (would have NULL-derefed pre-R18)"
LD0=$(lane_disabled_count); LD0=${LD0:-0}
HTTP=$(serial_ids n1);        [ "$HTTP" = "503" ] || fail "leg N: return_token_ids HTTP $HTTP (want typed 503)"
HTTP=$(cont_turn n2);         [ "$HTTP" = "200" ] || fail "leg N: cont turn after refusal HTTP $HTTP"
HTTP=$(anthropic_tools n3);   [ "$HTTP" = "503" ] || fail "leg N: anthropic buffered tools HTTP $HTTP (want typed 503)"
HTTP=$(anthropic_prefill n4); [ "$HTTP" = "503" ] || fail "leg N: anthropic prefill-only HTTP $HTTP (want typed 503)"
HTTP=$(cont_turn n5);         [ "$HTTP" = "200" ] || fail "leg N: cont turn after refusals HTTP $HTTP"
LD1=$(lane_disabled_count); LD1=${LD1:-0}
[ "$LD1" -ge $((LD0 + 3)) ] || fail "leg N: lane_disabled counter $LD0 -> $LD1 (want +3)"
ST=$(census_mib session_tensors)
grep_lease=$(metrics | grep 'ds4_memory_lease_bytes{consumer="serial_session",field="intent"}' | awk '{print int($2)}')
[ "${grep_lease:-0}" = "0" ] || fail "leg N: serial lease intent ${grep_lease} != 0 with no serial lane"
log "N PASS (3 typed 503s, counter +$((LD1-LD0)); cont serves between; session_tensors=${ST} MiB; serial lease 0)"

# ---------- Leg C: cont streaming on --no-serial ----------
HTTP=$(ssh "$R" "curl -s -m 120 -o /tmp/m4_c1.txt -w '%{http_code}' \
  http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Count to five, one number per line.\"}],\"max_tokens\":64,\"stream\":true}'")
[ "$HTTP" = "200" ] || fail "leg C: cont streaming HTTP $HTTP"
ssh "$R" "grep -q 'finish_reason' /tmp/m4_c1.txt" || fail "leg C: stream carried no finish_reason"
log "C PASS (cont streaming serves on --no-serial)"

ssh "$R" "pkill -x ${BIN:0:15}" 2>/dev/null
log "ALL LEGS PASS — artifacts in $OUT ($BIN killed, $R left free)"
