#!/bin/bash
# gates/legs/memory/deep_capacity_limit.sh — TRUE-LIMIT probe: keep admitting the
# deepest fundable prompts until the ENGINE refuses (or the box-side 1 Hz
# MemAvailable sampler sees < SAFETY_MIB, which would mean the governor's
# floor promise broke — that is a FAIL, not a stop).
#
# Shape truth this probe respects: zero-config -c 786432 has 4 banks and a
# 5th admission EVICTS the LRU warm bank instead of refusing, so residency
# beyond 4 x prompt only comes from DEEPER prompts, never more of them.
# Depth per admit = 750000 (decode budget 32768 rides on top; flen must
# stay under seq_cap 786432).  On refusal the probe halves the ask and
# retries (a refused admission consumes no bank), converging on the
# frontier; it stops at 4 admitted banks (geometric cap) or ask < 50k.
#
# Stop/verdict:
#   REFUSAL-CONVERGED  total = sum of admitted prompt_tokens, all resident
#   GEOMETRIC-CAP      4 banks admitted with no refusal (funding surprise)
#   SAFETY-ABORT       sampler saw MemAvailable < SAFETY_MIB -> FAIL LOUDLY
# Also asserts: every admitted needle answered exactly, floor intact at
# end, loaded decode stamp <= 1.30, sampler min recorded in the receipt.
#
# ~75-100 min (deep prefills).  Runs FROM the Mac.  End: server killed,
# sampler stopped, box free.  FLOOR_GB overrides the shipped 4 GiB floor
# (e.g. FLOOR_GB=1 probes the box edge; default = shipped).
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL:-18003}
CTX=${CTX:-786432}
# Depth ask: tokenizer inflation is LENGTH-DEPENDENT (measured 1.0068 at
# a 450k ask but 1.0238 at 772k, where the ask 400'd at 790,369 tokens),
# so the margin is sized on the worst measured ratio: ask*1.024 + MAX_TOK
# must stay under seq_cap 786432 or the ask dies on the exceeds-context
# path instead of probing admission.  A small explicit max_tokens (USER
# call: response << 10k; needle answers measured ~110 completion tokens
# uncapped) keeps the budget honest and lets each bank fill ~28k deeper
# than the 32768 agent-shape default.
DEPTH=${DEPTH:-740000}
MAX_TOK=${MAX_TOK:-4096}
MIN_ASK=${MIN_ASK:-50000}
MAX_BANKS=${MAX_BANKS:-4}
SAFETY_MIB=${SAFETY_MIB:-1024}
FLOOR_GB=${FLOOR_GB:-}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${OUT:-/tmp/deep_capacity_limit_$$}
SRV=/tmp/dcl_srv.log
CSV=/tmp/dcl_mem.csv
URL="http://127.0.0.1:$TUNNEL/v1/chat/completions"
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
wd(){ perl -e 'alarm shift @ARGV; exec @ARGV or die' "$@"; }
cleanup(){ ssh "$R" "touch /tmp/dcl_stop; P=\$(pgrep -x ${BIN:0:15}); [ -n \"\$P\" ] && kill \$P; exit 0" 2>/dev/null
           pkill -f "[s]sh -f -N -L $TUNNEL:" 2>/dev/null; }
fail(){ log "FAIL: $*"; scp -q "$R:$SRV" "$OUT/srv_fail.log" 2>/dev/null
        scp -q "$R:$CSV" "$OUT/mem.csv" 2>/dev/null; cleanup; exit 1; }
metric(){ curl -s -m 5 "http://127.0.0.1:$TUNNEL/metrics" | grep "^$1" | awk '{print $2}' | head -1; }
avail_mib(){ ssh "$R" "awk '/MemAvailable/{printf \"%d\", \$2/1024}' /proc/meminfo"; }
abort_hit(){ ssh "$R" "[ -f /tmp/dcl_abort ]" 2>/dev/null; }
decode_stamp(){
  wd 300 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/PR.json" "$URL" "$1" > "$OUT/$1.out" 2>&1 || \
    fail "$1 decode probe transport"
  sed -n 's/.*total=\([0-9.]*\)s.*/\1/p' "$OUT/$1.out" | head -1; }

CODES=("7413-DELTA-ONYX" "5527-IRON-MESA" "9082-CEDAR-VAULT" "2201-BRASS-FJORD" "6640-SLATE-HARBOR" "3318-COPPER-GLEN")
cat > "$OUT/PR.json" <<'EOF'
{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Count from one to two hundred as English words, comma separated, no other text."}],"max_tokens":128}
EOF

log "boot: fresh $BIN -c $CTX (zero-config${FLOOR_GB:+, DS4_MEM_FLOOR_GB=$FLOOR_GB})"
ssh "$R" "P=\$(pgrep -x ${BIN:0:15}); [ -n \"\$P\" ] && kill \$P; sleep 2; \
          P=\$(pgrep -x ${BIN:0:15}); [ -n \"\$P\" ] && kill -9 \$P; \
          rm -f /tmp/ds4.lock /tmp/dcl_stop /tmp/dcl_abort $CSV; exit 0"
ssh "$R" ": > $SRV; cd $BINDIR; env DS4_SERVER_DEFAULT_TEMP=0 DS4_ADMIT_DEBUG=1 \
    ${FLOOR_GB:+DS4_MEM_FLOOR_GB=$FLOOR_GB} ${ENGINE_ENV:-} \
    setsid --fork nohup ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null; exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || \
    fail "BOOT-DIED: $(ssh "$R" "tail -3 $SRV" 2>/dev/null | tr '\n' ' ')"
  sleep 10; n=$((n+10)); [ $n -ge 1500 ] && fail "boot timeout"
done
# 1 Hz sampler with the safety tell (independent of the engine's own books)
ssh "$R" "cat > /tmp/dcl_sampler.sh <<'EOS'
#!/bin/bash
while :; do
  [ -f /tmp/dcl_stop ] && exit 0
  m=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo)
  echo \"\$(date +%s),\$m\" >> /tmp/dcl_mem.csv
  if [ \"\$m\" -lt $SAFETY_MIB ]; then
    echo \"\$(date +%s),ABORT,\$m\" >> /tmp/dcl_mem.csv; touch /tmp/dcl_abort; exit 0
  fi
  sleep 1
done
EOS
chmod +x /tmp/dcl_sampler.sh; setsid --fork /tmp/dcl_sampler.sh < /dev/null > /dev/null 2>&1; exit 0"
pkill -f "[s]sh -f -N -L $TUNNEL:" 2>/dev/null; sleep 1
ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" || fail "tunnel"
sleep 2
AV0=$(avail_mib)
log "boot up; MemAvailable ${AV0} MiB; sampler live (safety ${SAFETY_MIB} MiB)"
AT_REST=$(decode_stamp d0)
[ -n "$AT_REST" ] || fail "no at-rest stamp"
log "at-rest decode probe ${AT_REST}s / 128 tok"

TOTAL=0; ADMITS=0; ASK=$DEPTH; ATTEMPT=0; VERDICT=""
while [ $ADMITS -lt $MAX_BANKS ] && [ $ASK -ge $MIN_ASK ]; do
  abort_hit && fail "SAFETY-ABORT: sampler saw MemAvailable < ${SAFETY_MIB} MiB (floor promise broken); csv in $OUT"
  CODE=${CODES[$((ATTEMPT % 6))]}
  log "attempt $((ATTEMPT+1)): ask ~$ASK tok (admitted so far: $ADMITS banks, $TOTAL tok)"
  python3 "$REPO/gates/lib/needle_prompt.py" "$ASK" "$OUT/P$ATTEMPT.json" 0.5 "$CODE" deepseek-v4-flash "$ATTEMPT" || fail "prompt build"
  python3 - "$OUT/P$ATTEMPT.json" "$MAX_TOK" <<'EOF' || fail "shape cap"
import json, sys
b = json.load(open(sys.argv[1])); b["max_tokens"] = int(sys.argv[2])
json.dump(b, open(sys.argv[1], "w"))
EOF
  RJ_BEFORE=$(metric ds4_cont_admit_rejects_total); RJ_BEFORE=${RJ_BEFORE:-0}
  wd 3600 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/P$ATTEMPT.json" "$URL" "A$ATTEMPT" "$OUT/R$ATTEMPT.json" > "$OUT/A$ATTEMPT.out" 2>&1
  RC=$?
  RJ_AFTER=$(metric ds4_cont_admit_rejects_total); RJ_AFTER=${RJ_AFTER:-0}
  if [ "$RJ_AFTER" != "$RJ_BEFORE" ]; then
    log "REFUSED at ask ~$ASK (rejects $RJ_BEFORE -> $RJ_AFTER); halving"
    ASK=$((ASK / 2)); ATTEMPT=$((ATTEMPT+1)); VERDICT="REFUSAL-SEEN"
    continue
  fi
  [ $RC -eq 0 ] || fail "attempt $((ATTEMPT+1)) rc=$RC with no refusal (transport?): $(tail -2 "$OUT/A$ATTEMPT.out" | tr '\n' ' ')"
  grep -q "$CODE" "$OUT/A$ATTEMPT.out" || fail "attempt $((ATTEMPT+1)): needle not answered exactly"
  PT=$(python3 -c "import json; print(json.load(open('$OUT/R$ATTEMPT.json'))['usage']['prompt_tokens'])") || fail "no usage"
  [ "$PT" -ge $((ASK * 96 / 100)) ] || fail "attempt $((ATTEMPT+1)): prompt_tokens $PT < 96% of $ASK"
  TOTAL=$((TOTAL + PT)); ADMITS=$((ADMITS+1)); ATTEMPT=$((ATTEMPT+1))
  log "ADMITTED: $PT tok (bank $ADMITS; total $TOTAL); MemAvailable $(avail_mib) MiB"
done
[ -z "$VERDICT" ] && VERDICT="GEOMETRIC-CAP"
[ "$VERDICT" = "REFUSAL-SEEN" ] && VERDICT="REFUSAL-CONVERGED"

abort_hit && fail "SAFETY-ABORT after admissions (floor promise broken)"
AV1=$(avail_mib)
LOADED=$(decode_stamp d1)
[ -n "$LOADED" ] || fail "no loaded stamp"
STAMP=$(python3 -c "print(round($LOADED / $AT_REST, 3))")
python3 -c "import sys; sys.exit(0 if $STAMP <= 1.30 else 1)" || \
  fail "loaded decode slowdown x$STAMP at $TOTAL resident"
curl -s -m 10 "http://127.0.0.1:$TUNNEL/metrics" > "$OUT/metrics_end.txt"
ssh "$R" "touch /tmp/dcl_stop"
scp -q "$R:$SRV" "$OUT/srv.log" 2>/dev/null
scp -q "$R:$CSV" "$OUT/mem.csv" 2>/dev/null
MIN_AV=$(awk -F, '$2 ~ /^[0-9]+$/ {if (m=="" || $2<m) m=$2} END{print m}' "$OUT/mem.csv" 2>/dev/null)
cleanup
log "TRUE-LIMIT PROBE $VERDICT — $TOTAL tokens resident across $ADMITS banks; MemAvailable end $AV1 MiB, sampler min ${MIN_AV:-?} MiB (safety ${SAFETY_MIB}); decode x$STAMP vs at-rest (${AT_REST}s -> ${LOADED}s / 128 tok); artifacts in $OUT"
