#!/bin/bash
# memory/serial_rightsize.sh — v0.5.2 inc1: serial-session right-sizing gate.
#
# Field class (forum 378855 posts 5+9, repro'd 2026-08-02 on .33): the serial
# session's S6 lazy graph is sized by server -c, not the request, so on a
# bank-holding deep boot EVERY serial-path request (anthropic/responses APIs,
# non-streaming token-id echo, cont-admission rejects, tools-on-completions)
# 500s at the fit gate — measured: an ~11 GiB graph demand for a 26-token
# /v1/messages job at -c 250000 with 7.8 GiB free.  The fix re-creates the
# (still graphless) session at the largest fit-gate-passing ctx bounded by
# prompt+budget+headroom, refusing 503 only when even a minimal output
# window cannot fit.  All serial routes share the run_job_single choke point,
# so exercising the always-serial API shapes covers the reject-fallback route
# by construction.
#
#   leg deep_field    -c 250000 zero-config (the jbourny/kafej666 shape).
#       asserts: cont streaming control 200; anthropic /v1/messages 200 with
#       content + ONE "serial session right-sized" line; token-ids request
#       200 reusing the same session (line count still 1); NO "lazy session
#       graph alloc failed" anywhere; ds4_graph_fit_refusals_total == 0
#       (binary-search probes are quiet); a ~35k-token prompt then forces a
#       REGROW (line count 2, '[live serial state reset]' marker) and serves.
#   leg rightsize_off same boot + DS4_SERVER_SERIAL_RIGHTSIZE=0 (escape).
#       asserts: the pre-fix 500 with the exact field error string — proves
#       the kill switch works AND that deep_field's asserts are load-bearing.
#   leg refuse_503    -c 16384 + DS4_SESSION_GRAPH_HEADROOM_MB=200000
#       (fit gate can never pass).  asserts: serial request gets a clean 503
#       + "no graph fits" log line; cont streaming still 200; server alive.
#   leg shallow_ctl   -c 16384 default.  asserts: serial shapes 200 with
#       ZERO right-size lines (shallow behavior byte-identical to v0.5.1).
#
# Runs FROM the Mac over SSH. Each boot kills any running ds4-server on $R.
# End state: ds4-server killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
OUT=${OUT:-/tmp/serial_rightsize_gate_$$}
SRV=/tmp/srg_srv.log
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; ssh "$R" "pkill -x ds4-server" 2>/dev/null; exit 1; }
kill_server(){ ssh "$R" "pkill -x ds4-server" 2>/dev/null; sleep 3; }

wait_mem(){ # standing law: never boot into a dying server's reclaim
  local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge "$1" ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable ${got:-?}G never reached ${1}G"
    sleep 5
  done
}

tunnel_up(){
  curl -s -m 5 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1 && return 0
  ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true
  sleep 2
  curl -s -m 10 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1
}

boot(){ # $1=ctx $2=extra env assignments (optional)
  kill_server
  wait_mem 100
  ssh "$R" ": > $SRV; cd $BINDIR; env ${2:-} setsid nohup ./ds4-server -c $1 --port $PORT \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
      sleep 3
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "BOOT-DIED ctx=$1: $(ssh "$R" "tail -3 $SRV" 2>/dev/null | tr '\n' ' ')"
    fi
    sleep 10; n=$((n+10)); [ $n -ge 1800 ] && fail "boot timeout ctx=$1"
  done
  tunnel_up || fail "tunnel"
}

rs_count(){ ssh "$R" "grep -c 'serial session right-sized' $SRV" 2>/dev/null | tr -d '[:space:]'; }

req_stream_ctl(){ # $1=outfile-prefix -> echoes http code
  curl -s -m 180 -o "$OUT/$1.txt" -w '%{http_code}' \
    "http://127.0.0.1:$TUNNEL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16,"stream":true}'
}
req_anthropic(){ # $1=outfile-prefix $2=body-file(optional)
  if [ -n "${2:-}" ]; then
    curl -s -m 600 -o "$OUT/$1.txt" -w '%{http_code}' \
      "http://127.0.0.1:$TUNNEL/v1/messages" -H 'Content-Type: application/json' -d @"$2"
  else
    curl -s -m 180 -o "$OUT/$1.txt" -w '%{http_code}' \
      "http://127.0.0.1:$TUNNEL/v1/messages" -H 'Content-Type: application/json' \
      -d '{"model":"ds4","max_tokens":16,"messages":[{"role":"user","content":"Say hello in one word."}]}'
  fi
}
req_token_ids(){ # $1=outfile-prefix
  curl -s -m 180 -o "$OUT/$1.txt" -w '%{http_code}' \
    "http://127.0.0.1:$TUNNEL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16,"return_token_ids":true}'
}

# ---- leg deep_field --------------------------------------------------------
log "=== leg deep_field: -c 250000 zero-config ==="
boot 250000
ssh "$R" "grep -q 'persistent batch ctx ready' $SRV" || fail "deep_field: no batch ctx"
c=$(req_stream_ctl df_ctl);   [ "$c" = 200 ] || fail "deep_field control http=$c"
c=$(req_anthropic df_anth);   [ "$c" = 200 ] || fail "deep_field anthropic http=$c ($(head -c 200 "$OUT/df_anth.txt"))"
grep -q '"text":"..*"' "$OUT/df_anth.txt" || fail "deep_field anthropic empty content"
n=$(rs_count); [ "$n" = 1 ] || fail "deep_field right-size lines=$n want 1"
c=$(req_token_ids df_ids);    [ "$c" = 200 ] || fail "deep_field token-ids http=$c"
n=$(rs_count); [ "$n" = 1 ] || fail "deep_field reuse broken: right-size lines=$n want 1"
ssh "$R" "grep -q 'lazy session graph alloc failed' $SRV" && fail "deep_field saw the old 500 error"
m=$(curl -s -m 10 "http://127.0.0.1:$TUNNEL/metrics" | awk '/^ds4_graph_fit_refusals_total/{print $2}')
[ "${m:-1}" = 0 ] || fail "deep_field fit refusals=$m want 0 (probes must be quiet)"
log "deep_field serve+reuse PASS; building regrow prompt"
python3 - "$OUT/regrow_body.json" <<'PY'
import json, sys
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 3500).strip()
body = {"model": "ds4", "max_tokens": 32,
        "messages": [{"role": "user",
                      "content": words + "\nHow many times does the word alpha appear above? Reply with a number."}]}
open(sys.argv[1], "w").write(json.dumps(body))
PY
c=$(req_anthropic df_regrow "$OUT/regrow_body.json"); [ "$c" = 200 ] || fail "regrow http=$c ($(head -c 200 "$OUT/df_regrow.txt"))"
n=$(rs_count); [ "$n" = 2 ] || fail "regrow right-size lines=$n want 2"
ssh "$R" "grep 'serial session right-sized' $SRV | tail -1 | grep -q 'live serial state reset'" || \
  fail "regrow missing committed-graph reset marker"
log "leg deep_field PASS"
ssh "$R" "cp $SRV /tmp/srg_deep_field.log"

# ---- leg rightsize_off -----------------------------------------------------
log "=== leg rightsize_off: -c 250000 + DS4_SERVER_SERIAL_RIGHTSIZE=0 ==="
boot 250000 "DS4_SERVER_SERIAL_RIGHTSIZE=0"
c=$(req_anthropic off_anth); [ "$c" = 500 ] || fail "rightsize_off http=$c want 500"
grep -q 'lazy session graph alloc failed' "$OUT/off_anth.txt" || fail "rightsize_off wrong error body"
log "leg rightsize_off PASS (escape reproduces the pre-fix 500)"
ssh "$R" "cp $SRV /tmp/srg_rightsize_off.log"

# ---- leg refuse_503 --------------------------------------------------------
log "=== leg refuse_503: -c 16384 + DS4_SESSION_GRAPH_HEADROOM_MB=200000 ==="
boot 16384 "DS4_SESSION_GRAPH_HEADROOM_MB=200000"
c=$(req_stream_ctl rf_ctl); [ "$c" = 200 ] || fail "refuse_503 cont control http=$c"
c=$(req_token_ids rf_ids);  [ "$c" = 503 ] || fail "refuse_503 http=$c want 503"
ssh "$R" "grep -q 'serial right-size: no graph fits' $SRV" || fail "refuse_503 missing refusal line"
curl -s -m 10 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null || fail "refuse_503 server died"
log "leg refuse_503 PASS (cont serves while serial refuses cleanly)"
ssh "$R" "cp $SRV /tmp/srg_refuse_503.log"

# ---- leg shallow_ctl -------------------------------------------------------
log "=== leg shallow_ctl: -c 16384 default ==="
boot 16384
c=$(req_token_ids sh_ids);  [ "$c" = 200 ] || fail "shallow_ctl token-ids http=$c"
c=$(req_anthropic sh_anth); [ "$c" = 200 ] || fail "shallow_ctl anthropic http=$c"
n=$(rs_count); [ "$n" = 0 ] || fail "shallow_ctl right-size lines=$n want 0 (behavior must be unchanged)"
ssh "$R" "grep -q 'session graph allocated lazily (ctx=16384' $SRV" || \
  fail "shallow_ctl expected the full--c lazy alloc"
log "leg shallow_ctl PASS"
ssh "$R" "cp $SRV /tmp/srg_shallow_ctl.log"

kill_server
log "ALL LEGS PASS. Artifacts in $OUT (+ /tmp/srg_*.log on $R)"
