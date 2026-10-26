#!/bin/bash
# gates/legs/memory/serial_reserve.sh — #8 governance inc2: serial-lane
# reservation (field 378855 post 49 emX0r: serial 503s above ~90k on
# v0.5.4 because bank growth eats the lazy serial graph's room).
#
# DS4_SERIAL_RESERVE_CTX=<tokens> carves a serial-lane reservation
# (session-graph estimate at that depth + 1 GiB fit headroom, clamped to
# what the box can spare beside the bank floor) out of every bank-growth
# verdict: boot comp budget and the mem-floor verdict.
#
# DEFAULT OFF since 08-05, and this gate documents why: a right-sized
# serial graph costs ~7.3 GiB while deep batch serving at -c 524288
# leaves ~6.8 GiB free, so a STATIC carve cannot fund both lanes -- it
# only picks a loser, and on default boots it must not be the batch
# path (the 240k deep gate caught exactly that regression).  This gate
# therefore proves the LEVER works for deployments that opt in, not a
# default behavior.  The general fix is reclaim (serial sheds bank cache
# pages when it cannot fit) -- see the memory-governance design.
#
# Legs (deterministic; the floor is tuned so the reserve TERM alone
# decides the verdict -- binary-controlled by the env kill switch):
#   1 boot     -c 130000 (the field ctx), reserve opted in at ENT tokens
#              (default 65536 = the serial fallback guard's own depth
#              limit): boot ledger prints "serial reserve=R GiB", R > 0,
#              and the #15 plan-budget arithmetic holds.  Record R, F.
#   2 engage   --mem-floor-gb ceil(F-R)+1 (usable beyond floor+reserve
#              < 0): a fresh streaming cont admission REJECTS on the
#              floor line WITH the "serial reserve" term in the message;
#              a 15k /v1/messages serial request then SERVES (200,
#              'prompt start' marker) -- the lane the reserve protects.
#   3 control  same floor, DS4_SERIAL_RESERVE_CTX=0: boot prints
#              reserve=0.00; the SAME admission is ACCEPTED and serves
#              -- the reserve term alone flipped the verdict.
#   4 promise  commons pinned tiny + reserve on: a request just under
#              the guard depth SERVES (the opted-in deployment's win).
#
# Runs FROM the Mac over SSH; end state: box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
CTX=${CTX:-130000}
SRV=/tmp/serial_reserve_gate.log
OUT=${OUT:-/tmp/serial_reserve_gate_$$}
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; ssh "$R" "tail -4 $SRV" 2>/dev/null; kill_all; exit 1; }
mem_gib(){ ssh "$R" "awk '/MemAvailable/{printf \"%.2f\", \$2/1048576}' /proc/meminfo"; }
srv_count(){ local c; c=$(ssh "$R" "grep -c \"$1\" $SRV 2>/dev/null || true" 2>/dev/null | tail -1); echo "${c:-0}"; }

wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable never reached 100G"; sleep 5
  done }

boot(){ # $1 = env prefix, $2 = extra args
  kill_all; wait_mem
  ssh "$R" ": > $SRV; cd $BINDIR; env $1 setsid nohup ./$BIN -c $CTX $2 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
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

# settle: two consecutive free readings within 0.3 GiB
settle_free(){ local a b n=0
  a=$(mem_gib)
  while :; do
    sleep 8; b=$(mem_gib)
    python3 -c "import sys; sys.exit(0 if abs($a-$b) < 0.3 else 1)" && { echo "$b"; return; }
    a=$b; n=$((n+1)); [ $n -ge 15 ] && { echo "$b"; return; }
  done }

serial_body(){ # ~15k-token /v1/messages prompt (the emX0r shape)
  python3 - <<'PY'
import json
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 1500).strip()
print(json.dumps({"model": "ds4", "max_tokens": 128, "messages": [
  {"role": "user", "content": words + "\nReply with exactly: SERIAL-OK"}]}))
PY
}

cont_body(){ # streaming chat = continuous path (W6); ~60k-token admission.
  # Sized to BEAT the eviction ladder: at COALESCE_MAX=2 at most one free
  # bank's ~97 MiB floor is trimmable, and a 60k admission projects
  # ~225 MiB -- so a zero-usable verdict must REJECT, not trim-salvage
  # (leg 2's first run at 12k/8-banks was silently saved by trim: correct
  # governance, wrong leg sizing).
  python3 - <<'PY'
import json
words = ("kilo lima mike november oscar papa quebec romeo sierra tango " * 6000).strip()
print(json.dumps({"messages": [{"role": "user", "content":
  words + "\nSummarize in one word."}], "max_tokens": 32, "stream": True}))
PY
}

# The reservation is DEFAULT OFF since 08-05 (a static carve cannot be
# funded beside deep batch serving -- see the ds4.c comment and the
# governance design's reclaim section). Every leg that wants it opts in
# explicitly at the entitlement the serial fallback guard accepts.
ENT=${ENT:-65536}
log "== leg 1: boot receipt (-c $CTX, reserve opted in at $ENT tok) =="
boot "DS4_SERIAL_RESERVE_CTX=$ENT" ""
BOOTLINE=$(ssh "$R" "grep 'batch vmm' $SRV | head -1")
log "boot: $BOOTLINE"
RES=$(echo "$BOOTLINE" | grep -oE 'serial reserve=[0-9.]+' | cut -d= -f2)
[ -n "$RES" ] || fail "boot ledger has no serial reserve field"
python3 -c "import sys; sys.exit(0 if float('$RES') > 0.5 else 1)" \
  || fail "serial reserve $RES GiB implausibly small"
# #15 receipt: the budget is the FIT PLAN's allowance (max_seq x full
# per-bank virtual extent), not a vfree snapshot -- pure arithmetic on
# the boot line, deterministic across boots regardless of memory state.
python3 - "$BOOTLINE" <<'PY' || fail "#15: budget != plan allowance (banks x virtual/bank)"
import re, sys
line = sys.argv[1]
vpb   = float(re.search(r"virtual ([0-9.]+) MiB/bank", line).group(1))
banks = int(re.search(r"x (\d+) banks", line).group(1))
bud   = float(re.search(r"budget=([0-9.]+) GiB", line).group(1))
plan  = banks * vpb / 1024.0
print(f"plan check: {banks} x {vpb} MiB = {plan:.2f} GiB vs budget {bud:.2f} GiB")
sys.exit(0 if abs(plan - bud) < 0.05 * max(plan, 1.0) else 1)
PY
log "#15 receipt: budget == plan allowance"
log "== leg 2 prep: re-measure free under the leg 2/3 config (2 banks) =="
# Legs 2/3 boot COALESCE_MAX=2 (minimal trimmable slack, see cont_body);
# the floor must be computed from THAT config's own settled free, and
# +1 GiB over the carve line so post-boot page-cache settling cannot
# drift usable back above zero.
boot "DS4_SERVER_COALESCE_MAX=2 DS4_SERIAL_RESERVE_CTX=$ENT" ""
F=$(settle_free)
log "reserve R=$RES GiB, settled free (2-bank config) F=$F GiB"
FLOOR=$(python3 -c "import math; print(max(1, math.ceil($F - $RES) + 1))")
log "leg 2/3 floor: --mem-floor-gb $FLOOR (usable beyond floor+reserve < 0)"

log "== leg 2: verdict engagement (reserve ON, floor $FLOOR) =="
boot "DS4_SERVER_COALESCE_MAX=2 DS4_SERIAL_RESERVE_CTX=$ENT" "--mem-floor-gb $FLOOR"
REJ0=$(srv_count "rejected on memory floor")
cont_body > "$OUT/cont.json"
curl -s -N -m 240 "http://127.0.0.1:$TUNNEL/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @"$OUT/cont.json" > "$OUT/leg2_cont.out" 2>&1 || true
REJ1=$(srv_count "rejected on memory floor")
[ "$REJ1" -gt "$REJ0" ] || fail "leg 2: admission was not floor-rejected (REJ $REJ0 -> $REJ1)"
ssh "$R" "grep 'rejected on memory floor' $SRV | tail -1" | grep -q "serial reserve" \
  || fail "leg 2: floor reject line lacks the serial reserve term"
log "leg 2: floor reject fired WITH the reserve term"
serial_body > "$OUT/serial.json"
HTTP=$(curl -s -o "$OUT/leg2_serial.out" -w '%{http_code}' -m 600 \
  "http://127.0.0.1:$TUNNEL/v1/messages" \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d @"$OUT/serial.json")
[ "$HTTP" = "200" ] || fail "leg 2: serial request got $HTTP (want 200): $(head -c 200 "$OUT/leg2_serial.out")"
grep -q "SERIAL-OK" "$OUT/leg2_serial.out" || log "note: serial reply lacks echo (acceptable; 200 + completion is the gate)"
[ "$(srv_count 'prompt start')" -ge 1 ] || fail "leg 2: no serial 'prompt start' marker"
log "leg 2: serial lane SERVED beside the floor-rejected growth"

log "== leg 3: binary control (reserve OFF, same floor) =="
boot "DS4_SERVER_COALESCE_MAX=2 DS4_SERIAL_RESERVE_CTX=0" "--mem-floor-gb $FLOOR"
ssh "$R" "grep 'batch vmm' $SRV | head -1" | grep -qE "serial reserve=0.00" \
  || fail "leg 3: reserve not disabled by DS4_SERIAL_RESERVE_CTX=0"
REJ0=$(srv_count "rejected on memory floor")
curl -s -N -m 240 "http://127.0.0.1:$TUNNEL/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @"$OUT/cont.json" > "$OUT/leg3_cont.out" 2>&1 || true
REJ1=$(srv_count "rejected on memory floor")
[ "$REJ1" -eq "$REJ0" ] || fail "leg 3: admission floor-rejected even with reserve off (floor tuning wrong?)"
grep -q "data:" "$OUT/leg3_cont.out" || fail "leg 3: cont request produced no stream output"
log "leg 3: same admission ACCEPTED with the reserve off (term is decisive)"

log "== leg 4: the FUNDED PROMISE (commons pinned tiny, serial serves at the guard depth) =="
# Guard/reservation agreement: the serial fallback guard admits up to
# DS4_SERVER_SERIAL_MAX_TOKENS (65536 default), so a just-under-that
# request must SERVE with the reservation opted in and the commons
# budget pinned tiny (the outage shape dissolves for that deployment).
boot "DS4_BATCH_VMM_BUDGET_MB=256 DS4_SERIAL_RESERVE_CTX=$ENT" ""
DEEP_TOK=$(( ENT - 4000 ))   # just under the guard depth, margin for template
python3 - "$DEEP_TOK" > "$OUT/leg4_serial.json" <<'PY'
import json, sys
n = int(sys.argv[1])
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet " * ((n // 10) + 1)).split()
body = " ".join(words[:n - 40])
print(json.dumps({"model": "ds4", "max_tokens": 64, "messages": [
  {"role": "user", "content": body + "\nReply with exactly: DEEP-SERIAL-OK"}]}))
PY
HTTP=$(curl -s -o "$OUT/leg4_serial.out" -w '%{http_code}' -m 1800 \
  "http://127.0.0.1:$TUNNEL/v1/messages" \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d @"$OUT/leg4_serial.json")
[ "$HTTP" = "200" ] || fail "leg 4: ~${ENT}-tok serial request got $HTTP (the promise is unfunded): $(head -c 200 "$OUT/leg4_serial.out")"
log "leg 4: ~${ENT}-tok serial request SERVED with the commons pinned (promise funded)"

kill_all
log "SERIAL-RESERVE-GATE PASS (R=$RES GiB, F=$F, floor=$FLOOR; verdict binary-controlled; serial served under pressure incl. the funded guard-depth promise)"
