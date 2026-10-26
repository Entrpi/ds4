#!/bin/bash
# gates/legs/memory/memcal.sh — MT-7 (v0.6.1 memory truth): the STANDING
# memory-calibration battery, promoted from the leg2a driver
# (local/docs/v061/leg2a/leg2a_memcal.sh; offline joins still via
# leg2a_analyze.py on the preserved samples).
#
# One deep zero-config boot (-c 786432) runs the charter event sequence
# through the real API while a box-side 1 Hz sampler logs census classes +
# meminfo:
#   E0 baseline + at-rest probe   E1 250k cold   E2 250k cold beside it
#   E3 warm turn-2 on E1's bank   E4 THE 500k admit   E5 shallow 10k beside
#
# What the gate ASSERTS (leg2a was calibration; this is the battery leg):
#   - every event serves rc=0; zero cont admission rejects across the run
#     (deep shapes admit on actual requirements -- the charter);
#   - the MT-7 rate gauges: observed commit B/tok within [1.0, 1.5]x of the
#     packed shape rate at ~1M resident tokens (page floors amortized;
#     leg2a measured ~1.0-1.1x) and NO commit-rate anomaly warning;
#   - the disclosed admission band in the ready ledger (admit_band=);
#   - E4 all-in steady cost within [3.0, 7.0] KiB/tok of MemAvailable
#     (the leg2a constants 4.1-4.5 with sampler-noise margin -- catches a
#     return of the ~20.3 KiB/tok virtual-extent class and gross leaks).
#
# Runs FROM the Mac over SSH + tunnel (500k bodies).  Release-battery
# cadence (~60 min).  End state: server + sampler killed, box left free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL:-18000}
CTX=${CTX:-786432}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${OUT:-/tmp/memcal_gate_$$}
SRV=/tmp/memcal_gate_srv.log
SAMP=/tmp/memcal_samples.log
URL="http://127.0.0.1:$TUNNEL/v1/chat/completions"
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
mark(){ echo "$(date +%s) $1" >> "$OUT/events.log"; log "EVENT $1"; }
wd(){ perl -e 'alarm shift @ARGV; exec @ARGV or die' "$@"; }
cleanup(){
  ssh "$R" 'pkill -f "[m]emcal_sampler"; exit 0' 2>/dev/null
  ssh "$R" "pkill -x ${BIN:0:15}; exit 0" 2>/dev/null
  scp -q "$R:$SAMP" "$OUT/samples.log" 2>/dev/null
  scp -q "$R:$SRV" "$OUT/srv.log" 2>/dev/null
  pkill -f "[s]sh -f -N -L $TUNNEL:" 2>/dev/null
}
fail(){ log "FAIL: $*"; cleanup; exit 1; }
metric(){ curl -s -m 5 "http://127.0.0.1:$TUNNEL/metrics" | grep "^$1" | awk '{print $2}' | head -1; }
avail_kib(){ ssh "$R" "awk '/MemAvailable/{printf \"%d\", \$2}' /proc/meminfo"; }

# ---------- prompts (distinct corpora rotations, distinct needles) -----------
log "build prompts (500k + 2x250k + shallow 10k)"
python3 "$REPO/gates/lib/needle_prompt.py" 500000 "$OUT/A.json" 0.5 "7413-DELTA-ONYX"  deepseek-v4-flash 0 || fail "prompt A"
python3 "$REPO/gates/lib/needle_prompt.py" 250000 "$OUT/B.json" 0.5 "5527-IRON-MESA"   deepseek-v4-flash 1 || fail "prompt B"
python3 "$REPO/gates/lib/needle_prompt.py" 250000 "$OUT/C.json" 0.5 "9082-CEDAR-VAULT" deepseek-v4-flash 2 || fail "prompt C"
python3 "$REPO/gates/lib/needle_prompt.py" 10000  "$OUT/D.json" 0.5 "2201-BRASS-FJORD" deepseek-v4-flash 3 || fail "prompt D"

# ---------- sampler ----------------------------------------------------------
ssh "$R" "cat > /tmp/memcal_sampler.sh" <<'EOS' || fail "sampler ship"
#!/bin/bash
OUT=/tmp/memcal_samples.log
: > $OUT
while :; do
  ts=$(date +%s)
  eval $(awk '/^MemAvailable/{printf "av=%s;",$2} /^AnonPages/{printf "an=%s;",$2} /^Cached/{printf "ca=%s;",$2}' /proc/meminfo)
  echo "SAMPLE $ts avail=$av anon=$an cached=$ca" >> $OUT
  curl -s -m 2 http://127.0.0.1:8000/metrics 2>/dev/null | \
    grep -E '^ds4_memory_bytes\{.*state="allocated"|^ds4_memory_lease_bytes|^ds4_cont_admit_rejects_total|^ds4_cont_commit_bytes_per_token' | \
    sed "s/^/M $ts /" >> $OUT
  sleep 1
done
EOS

# ---------- boot (zero-config deep) ------------------------------------------
log "boot: fresh $BIN -c $CTX (zero-config launch defaults)"
ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
ssh "$R" 'prev=$(awk "/MemAvailable/{print \$2}" /proc/meminfo); i=0
  while [ $i -lt 24 ]; do sleep 5
    cur=$(awk "/MemAvailable/{print \$2}" /proc/meminfo)
    d=$((cur - prev)); [ $d -lt 0 ] && d=$((-d))
    [ $d -lt 512000 ] && { echo "mem stable: $((cur/1048576)) GiB"; exit 0; }
    prev=$cur; i=$((i+1)); done
  echo "mem NOT stable after 120s (continuing)"'
ssh "$R" ": > $SRV; cd $BINDIR; env DS4_SERVER_DEFAULT_TEMP=0 \
    setsid nohup ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || \
    fail "BOOT-DIED: $(ssh "$R" "tail -3 $SRV" 2>/dev/null | tr '\n' ' ')"
  sleep 10; n=$((n+10)); [ $n -ge 1500 ] && fail "boot timeout"
done
READY=$(ssh "$R" "grep 'persistent batch ctx ready' $SRV" | head -1)
log "ready: $READY"
echo "$READY" | grep -q 'admit_band=' || fail "ready ledger does not disclose admit_band"
ssh "$R" "setsid nohup bash /tmp/memcal_sampler.sh > /dev/null 2>&1 < /dev/null & exit 0" || fail "sampler start"
sleep 2
ssh "$R" "pgrep -f '[m]emcal_sampler' >/dev/null" || fail "sampler not running"

pkill -f "[s]sh -f -N -L $TUNNEL:" 2>/dev/null; sleep 1
ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" || fail "tunnel"
sleep 2

RJ0=$(metric ds4_cont_admit_rejects_total); RJ0=${RJ0:-0}

# ---------- E0: baseline + probe ---------------------------------------------
mark "E0_BASELINE_START"
sleep 60
PROBE=$(curl -s -m 120 -X POST "$URL" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4}')
echo "$PROBE" | grep -q '"choices"' || fail "E0 probe: $PROBE"
mark "E0_DONE"

# ---------- E1/E2: two 250k cold admits --------------------------------------
mark "E1_250K_B_START"
wd 1500 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/B.json" "$URL" E1B "$OUT/B_result.json" > "$OUT/E1.out" 2>&1
RC1=$?; tail -3 "$OUT/E1.out"
[ $RC1 -eq 0 ] || fail "E1 250k cold rc=$RC1"
mark "E1_DONE"
sleep 90
mark "E2_250K_C_START"
wd 1500 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/C.json" "$URL" E2C > "$OUT/E2.out" 2>&1
RC2=$?; tail -3 "$OUT/E2.out"
[ $RC2 -eq 0 ] || fail "E2 250k cold rc=$RC2"
mark "E2_DONE"
sleep 90

# ---------- E3: warm turn-2 on B ---------------------------------------------
python3 - "$OUT/B.json" "$OUT/B_result.json" "$OUT/B_turn2.json" <<'EOF' || fail "turn2 build"
import json, sys
b = json.load(open(sys.argv[1])); r = json.load(open(sys.argv[2]))
b["messages"].append({"role": "assistant", "content": r["text"]})
b["messages"].append({"role": "user", "content":
    "Good. Repeat the keystone code once more, then summarize the archive in 200 words."})
b["max_tokens"] = 400
json.dump(b, open(sys.argv[3], "w"))
EOF
mark "E3_WARM_B_START"
wd 900 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/B_turn2.json" "$URL" E3W > "$OUT/E3.out" 2>&1
RC3=$?; tail -3 "$OUT/E3.out"
[ $RC3 -eq 0 ] || fail "E3 warm turn-2 rc=$RC3"
mark "E3_DONE"
sleep 60

# ---------- E4: THE 500k admit -----------------------------------------------
AV_E4_0=$(avail_kib)
mark "E4_500K_A_START"
wd 3000 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/A.json" "$URL" E4A "$OUT/A_result.json" > "$OUT/E4.out" 2>&1
RC4=$?; tail -3 "$OUT/E4.out"
[ $RC4 -eq 0 ] || fail "E4 500k admit rc=$RC4"
mark "E4_DONE"
sleep 120
AV_E4_1=$(avail_kib)

# E4 all-in steady cost: MemAvailable delta / admitted tokens vs the leg2a band
ATOK=$(python3 -c "import json; print(json.load(open('$OUT/A_result.json'))['usage']['prompt_tokens'])" 2>/dev/null)
[ -n "$ATOK" ] && [ "$ATOK" -ge 480000 ] || fail "E4 prompt_tokens=$ATOK (want >=480k)"
KIBTOK=$(python3 -c "print(round(($AV_E4_0 - $AV_E4_1) / $ATOK, 2))")
log "E4 all-in steady: $KIBTOK KiB/tok over $ATOK tokens (leg2a band 4.1-4.5)"
python3 -c "import sys; k=$KIBTOK; sys.exit(0 if 3.0 <= k <= 7.0 else 1)" || \
  fail "E4 steady $KIBTOK KiB/tok outside [3.0, 7.0] (virtual-extent class or leak)"

# ---------- E5: shallow 10k beside the deep banks ----------------------------
mark "E5_SHALLOW_START"
wd 600 python3 "$REPO/gates/lib/sse_probe_client.py" "$OUT/D.json" "$URL" E5S > "$OUT/E5.out" 2>&1
RC5=$?; tail -3 "$OUT/E5.out"
[ $RC5 -eq 0 ] || fail "E5 shallow rc=$RC5"
mark "E5_DONE"
sleep 30

# ---------- verdicts ---------------------------------------------------------
RJ1=$(metric ds4_cont_admit_rejects_total); RJ1=${RJ1:-0}
[ "$RJ1" = "$RJ0" ] || fail "cont admission rejects grew $RJ0 -> $RJ1 (deep shapes must admit)"
OBS=$(metric 'ds4_cont_commit_bytes_per_token{kind="observed"}')
PHYS=$(metric 'ds4_cont_commit_bytes_per_token{kind="packed"}')
[ -n "$OBS" ] && [ -n "$PHYS" ] && [ "$PHYS" -gt 0 ] || fail "MT-7 rate gauges missing (obs=$OBS phys=$PHYS)"
RATIO=$(python3 -c "print(round($OBS / $PHYS, 3))")
log "commit rate observed=$OBS B/tok packed=$PHYS B/tok ratio=$RATIO"
python3 -c "import sys; r=$RATIO; sys.exit(0 if 1.0 <= r <= 1.5 else 1)" || \
  fail "observed/packed ratio $RATIO outside [1.0, 1.5] at ~1M resident tokens"
WARN=$(ssh "$R" "grep -c 'cont commit rate .* exceeds' $SRV || true" | tail -1)
[ "${WARN:-0}" = "0" ] || fail "commit-rate anomaly warning fired ($WARN)"
curl -s -m 10 "http://127.0.0.1:$TUNNEL/metrics" > "$OUT/metrics_end.txt"
mark "TEARDOWN"
cleanup
log "ALL PASS — E1-E5 served, rejects 0, E4 steady $KIBTOK KiB/tok, rate ratio $RATIO, band disclosed; artifacts in $OUT ($BIN killed, $R left free)"
