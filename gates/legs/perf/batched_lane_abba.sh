#!/bin/bash
# perf/batched_lane_abba.sh — v0.5.6 API Inc 3 CLOSE: the plan §7 perf acceptance.
# ABBA legs on the same configuration, two builds:
#   A = tip (Inc 3a-3d promotions; BINDIR_A)
#   B = baseline `75c7d9c` (Inc 2e, last pre-promotion commit; BINDIR_B)
# Each leg is a FRESH BOOT (ms/tok compares need fresh boots BOTH sides),
# then, in funded-window order: N=16 burst, N=8 burst, N=1, and a ~2.3k-token
# cont prefill probe.  Decode rows are FIXED-LENGTH (256 tokens, thinking
# off, temp 0, budget-cut by design) so throughput compares are
# count-matched; tok/step is reported beside every ms/tok (accept-rate law).
# Lane purity is asserted per burst (floor rejects fall back serial and
# would poison the compare -- that failure is ENVIRONMENT, not a verdict).
#
# Verdict (computed at the end): mean tip-vs-base decode ms/tok per N within
# +2% (plan §7 "roughly 1-2%"), prefill tok/s within -5% (tripwire; §7 says
# "no material regression").  Prints the full table either way.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR_A=${BINDIR_A:-${DS4_BOX_TREE:?}}
BINDIR_B=${BINDIR_B:-${DS4_BOX_TREE_B:?}}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/inc3_perf_abba
OUT=${OUT:-/tmp/inc3_perf_abba_$$}
mkdir -p "$OUT"
BASE="http://127.0.0.1:$TUNNEL_PORT"
CSV="$OUT/results.csv"
echo "leg,build,metric,n,ms_per_tok,tok_per_step,rows" > "$CSV"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; exit 1; }

tunnel_up(){
  curl -s -m 5 "$BASE/v1/models" >/dev/null 2>&1 && return 0
  ssh -f -N -L "$TUNNEL_PORT:127.0.0.1:$PORT" "$R" 2>/dev/null || true
  sleep 2
  curl -s -m 10 "$BASE/v1/models" >/dev/null 2>&1
}

wait_mem(){
  local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge "$1" ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable ${got:-?}G never reached ${1}G"
    sleep 5
  done
}

boot(){ # $1 = BINDIR
  SRV=$RWORK/srv.log
  log "boot($1): killing old ds4-server on $R"
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; mkdir -p $RWORK; rm -f /tmp/ds4.lock; exit 0"
  wait_mem 100
  ssh "$R" ": > $SRV; cd $1; env DS4_MEM_FLOOR_GB=2 setsid nohup ./ds4-server -c $CTX --port $PORT \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
      sleep 3
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "BOOT-DIED($1): $(ssh "$R" "tail -2 $SRV" 2>/dev/null | tr '\n' ' ')"
    fi
    sleep 10; n=$((n+10)); [ $n -ge 1200 ] && fail "boot timeout ($1)"
  done
  tunnel_up || fail "tunnel :$TUNNEL_PORT unreachable"
  log "boot($1): up"
}

m(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$' | head -1; }

DECODE_REQ='{"max_tokens":256,"temperature":0,"thinking":false,"messages":[{"role":"user","content":"Count upward from 1, one number per line, without stopping."}]}'

python3 - > "$OUT/prefill_req.json" <<'PY'
import json
body = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. " * 620
print(json.dumps({"prompt": body, "max_tokens": 8, "temperature": 0}))
PY

burst(){ # $1=leg $2=build $3=N -> appends CSV row; asserts lane purity
  local leg=$1 build=$2 N=$3 i
  local oc0 sl0 oc1 sl1 reqs=$N
  oc0=$(m 'surface="openai_chat",lane="continuous"'); sl0=$(m ds4_requests_serial_total)
  if [ "$N" -gt 1 ]; then
    for i in $(seq 1 "$N"); do
      curl -s -m 300 -o "$OUT/${leg}_n${N}_$i.json" "$BASE/v1/chat/completions" \
           -H 'Content-Type: application/json' -d "$DECODE_REQ" &
    done
    wait
  else
    # N=1 is one sample per request -- a single 256-token request measured
    # 2.4 ms/tok run-to-run spread (10%) across fresh boots, 3x the deltas
    # being adjudicated.  Four SEQUENTIAL singles per leg give the mean the
    # same power the N=8/16 rows get from their own widths.
    reqs=4
    for i in 1 2 3 4; do
      curl -s -m 300 -o "$OUT/${leg}_n1_$i.json" "$BASE/v1/chat/completions" \
           -H 'Content-Type: application/json' -d "$DECODE_REQ"
    done
  fi
  oc1=$(m 'surface="openai_chat",lane="continuous"'); sl1=$(m ds4_requests_serial_total)
  [ $(( ${oc1:-0} - ${oc0:-0} )) -ge "$reqs" ] || fail "ENVIRONMENT $leg N=$N: only $(( ${oc1:-0} - ${oc0:-0} ))/$reqs rode cont"
  [ "${sl1:-0}" -eq "${sl0:-0}" ] || fail "ENVIRONMENT $leg N=$N: serial fallback engaged (${sl0:-?} -> ${sl1:-?})"
  python3 - "$leg" "$build" "$N" "$OUT" >> "$CSV" <<'PY'
import json, sys, glob
leg, build, N, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
# timings exposes decode_tok_s ((tokens-1)/decode_s) + tok_per_step, not the
# raw fields -- ms/tok = 1000/decode_tok_s, same convention both builds.
ms = []; tps = []
for f in glob.glob(f"{out}/{leg}_n{N}_*.json"):
    t = json.load(open(f)).get("timings", {})
    if not t.get("decode_tok_s"):
        sys.exit(f"{f}: no decode_tok_s in timings")
    ms.append(1000.0 / t["decode_tok_s"])
    tps.append(t.get("tok_per_step", 1.0))
print(f"{leg},{build},decode,{N},{sum(ms)/len(ms):.3f},{sum(tps)/len(tps):.3f},{len(ms)}")
PY
  [ $? -eq 0 ] || fail "$leg N=$N: timings extraction failed"
}

prefill_probe(){ # $1=leg $2=build
  local leg=$1 build=$2 oc0 oc1
  oc0=$(m 'surface="openai_completion",lane="continuous"')
  curl -s -m 300 -o "$OUT/${leg}_prefill.json" "$BASE/v1/completions" \
       -H 'Content-Type: application/json' -d @"$OUT/prefill_req.json"
  oc1=$(m 'surface="openai_completion",lane="continuous"')
  [ $(( ${oc1:-0} - ${oc0:-0} )) -ge 1 ] || fail "ENVIRONMENT $leg prefill: probe did not ride cont"
  python3 - "$leg" "$build" "$OUT" >> "$CSV" <<'PY'
import json, sys
leg, build, out = sys.argv[1], sys.argv[2], sys.argv[3]
# timings exposes prefill_tok_s, not prefill_ms (same as the decode fields).
t = json.load(open(f"{out}/{leg}_prefill.json")).get("timings", {})
pf_tok, pf_tok_s = t.get("prefill_tokens"), t.get("prefill_tok_s")
if not pf_tok or not pf_tok_s:
    sys.exit("prefill timings missing")
print(f"{leg},{build},prefill,1,{1000.0/pf_tok_s:.4f},0,{pf_tok}")
PY
  [ $? -eq 0 ] || fail "$leg prefill: timings extraction failed"
}

run_leg(){ # $1=leg-name $2=build-tag $3=BINDIR
  boot "$3"
  burst "$1" "$2" 16
  burst "$1" "$2" 8
  burst "$1" "$2" 1
  prefill_probe "$1" "$2"
  log "$1 ($2) complete"
}

# ---- ABBA ------------------------------------------------------------------
run_leg legA1 tip  "$BINDIR_A"
run_leg legB1 base "$BINDIR_B"
run_leg legB2 base "$BINDIR_B"
run_leg legA2 tip  "$BINDIR_A"

ssh "$R" "pkill -x ds4-server; exit 0"

# ---- verdict ----------------------------------------------------------------
python3 - "$CSV" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
def mean(v): return sum(v) / len(v)
fail = []
print(f"\n{'metric':8} {'N':>3} {'tip ms/tok':>11} {'base ms/tok':>12} {'delta%':>7}  tok/step tip/base")
for n in (16, 8, 1):
    a = [r for r in rows if r["metric"] == "decode" and r["n"] == str(n) and r["build"] == "tip"]
    b = [r for r in rows if r["metric"] == "decode" and r["n"] == str(n) and r["build"] == "base"]
    ma, mb = mean([float(r["ms_per_tok"]) for r in a]), mean([float(r["ms_per_tok"]) for r in b])
    ta, tb = mean([float(r["tok_per_step"]) for r in a]), mean([float(r["tok_per_step"]) for r in b])
    d = (ma - mb) / mb * 100
    print(f"{'decode':8} {n:>3} {ma:>11.3f} {mb:>12.3f} {d:>+6.2f}%  {ta:.3f}/{tb:.3f}")
    if d > 2.0: fail.append(f"decode N={n}: +{d:.2f}% > 2%")
    if abs(ta - tb) / tb > 0.02:
        print(f"  NOTE decode N={n}: tok/step moved {ta:.3f} vs {tb:.3f} -- accept-rate noise, read ms/tok with that in mind")
a = [float(r["ms_per_tok"]) for r in rows if r["metric"] == "prefill" and r["build"] == "tip"]
b = [float(r["ms_per_tok"]) for r in rows if r["metric"] == "prefill" and r["build"] == "base"]
d = (mean(a) - mean(b)) / mean(b) * 100
print(f"{'prefill':8} {'1':>3} {mean(a):>11.4f} {mean(b):>12.4f} {d:>+6.2f}%  (ms/prompt-token)")
if d > 5.0: fail.append(f"prefill: +{d:.2f}% > 5% tripwire")
if fail:
    print("\nVERDICT: FAIL — " + "; ".join(fail)); sys.exit(1)
print("\nVERDICT: PASS — tip within +2% decode at every N, prefill within tripwire")
PY
rc=$?
[ $rc -eq 0 ] || fail "perf acceptance breached (table above)"
log "ABBA PASS — results in $CSV"
