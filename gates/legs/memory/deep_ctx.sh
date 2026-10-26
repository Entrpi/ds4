#!/bin/bash
# gates/legs/memory/deep_ctx.sh — standing deep-context KV-capacity gate
# (v0.2 charter ws4; first PASS 2026-07-14: 766,437 tokens concurrent at
# ctx 524288 on GB10, needles exact to 518K, turn-2 warm in-place TTFT 1.2 s,
# deep decode 146-177 ms/tok).
#
# Drives a GB10 over SSH: boots the ship-config ds4-server FRESH from the
# tree already on the host (deploy/rebuild is a separate step, same as
# agent/teb.sh), then three stages on ONE boot:
#   1. CONCURRENT charter shape — ~ORCH_TOK orchestrator + ~SUB_TOK subagent
#      needle prompts fired 5 s apart; both must be admitted AND served on
#      cont with the needles answered exactly.
#   2. ORCHESTRATOR TURN 2 — the same conversation plus the ACTUAL stage-1
#      answer plus a new user turn. Must be a WARM IN-PLACE admit
#      (fork admit count 0 — the deep-trunk fork guard under test) with
#      cached >= ~ORCH depth; TTFT seconds, not tens of minutes.
#   3. Turn 2 generates ~512 tokens = the deep decode ms/tok stamp.
# PASS = both needles exact, warm in-place turn 2, 0 illegal / 0 cont-fail /
# 0 serial markers ('prompt start') / 0 admit rejects across all stages.
#
# Laws (earned 2026-07-14 — see memory kv-capacity-deep-ctx-gate):
#   * Nothing may route serial at deep ctx (serial session at 524288 takes
#     the managed-KV fallback; lazy-graph fit_ok fails it to a clean 500).
#   * Prompt-size estimates drift ±4% per corpus rotation: keep
#     ORCH_TOK + 6K + max_tokens under CTX.
#   * Self-load pins the GGUF bytes (~90.8 GiB on the ship set): small
#     batch-fit/budget numbers on a self-load box are physics, not staleness.
#   * NOTE: each boot kills any running ds4-server on the host, and the run
#     kills its own server again on exit (PASS or FAIL).
#
# Env overrides (defaults in parens):
#   DC_GATE_HOST (${DS4_BOX33:?})  BINDIR (${DS4_BOX_TREE:?})
#   CTX (524288)  ORCH_TOK (500000)  SUB_TOK (250000)
#   PORT (8000)  TUNNEL_PORT (18000)  HEADROOM_MB (6272)
#   DC_GATE_OUT (/tmp/deep_ctx_gate_<ts>)  GGUF (${DS4_GGUF_DIR:?})
#   EXTRA_ENV ("") — extra server env spliced into the boot, e.g.
#     EXTRA_ENV="DS4_CUDA_FP8_KV=1 DS4_CUDA_FP4_INDEX=1" for the >1M levers
#   SPEC (1) — SPEC=0 boots plain (--no-spec --no-mtp, no DSpark env): the
#     plain-decode cost reference at depth; PASS criteria are unchanged
#   EXTRA_ARGS ("") — extra ds4-server flags spliced after -c (e.g. "--vision FILE")
#   ZEROCONF (0) — ZEROCONF=1 drops the HEADROOM_MB/COALESCE_MAX pins so the
#     boot fits banks with the SHIPPED defaults (deepmem lite charter:
#     "default tested as shipped").  The unpinned fit is the thing under
#     test; PASS criteria are unchanged.  Default 0 = ship-env control.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${DC_GATE_HOST:-${DS4_BOX33:?}}
RT=${BINDIR:-${DS4_BOX_TREE:?}}
CTX=${CTX:-524288}
ORCH_TOK=${ORCH_TOK:-500000}
SUB_TOK=${SUB_TOK:-250000}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
HEADROOM_MB=${HEADROOM_MB:-6272}
OUT=${DC_GATE_OUT:-/tmp/deep_ctx_gate_$(date +%Y%m%d_%H%M%S)}
GGUF=${GGUF:-${DS4_GGUF_DIR:?}}
EXTRA_ENV=${EXTRA_ENV:-}
SPEC=${SPEC:-1}
ZEROCONF=${ZEROCONF:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}   # extra ds4-server flags spliced after -c, e.g. EXTRA_ARGS="--vision $GGUF/DeepSeek-V4-Flash-Vision-Encoder.gguf" (v0.7.0 3b/3c legs)
BASE=$GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MTP=${MTP-$GGUF/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}   # MTP="" boots --no-mtp (MTP-droppable legs; blocks launch-default auto-attach)
DRAFTER=$GGUF/DSpark-drafter-Q2K-Q8.gguf
SRV=/tmp/deep_ctx_gate_srv.log
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT"
DRV=$OUT/driver.log
SENT=$OUT/sentinel.log
rm -f "$SENT"; : > "$DRV"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRV"; }
fail(){ log "FAIL: $*"; echo "RUN_DONE_deepctxgate exit=1" >> "$SENT"; exit 1; }

# Every exit path must kill the server this run booted: an idle gate server
# holds ~90 GiB and /tmp/ds4.lock, blocking later ds4-bench runs on the box.
cleanup(){ log "cleanup: killing ds4-server on $R"
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null; }
trap cleanup EXIT

# v0.2.x counter-based health (PRIMARY): /metrics snapshots + deltas mirror
# the stderr greps in the PASS block, which stay as fallback for one release.
msnap(){ curl -s -m 10 "http://127.0.0.1:$TUNNEL/metrics" > "$1" && [ -s "$1" ]; }
mval(){ awk -v k="$2" '$1 == k {v=$2} END {printf "%.0f", v+0}' "$1" 2>/dev/null; }
mdelta(){ echo $(( $(mval "$2" "$3") - $(mval "$1" "$3") )); }

log "build needle prompts (orch ~$ORCH_TOK @35%, sub ~$SUB_TOK @70%)"
python3 "$DS4_GATES/lib/needle_prompt.py" "$ORCH_TOK" "$OUT/orch.json" 0.35 "8842-OMEGA-KILO" | tee -a "$DRV" || fail "orch prompt"
python3 "$DS4_GATES/lib/needle_prompt.py" "$SUB_TOK"  "$OUT/sub.json"  0.70 "5517-SIGMA-NOVA" | tee -a "$DRV" || fail "sub prompt"

log "boot: killing old ds4-server on $R"
ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0"
# Boot-fit hygiene (law re-learned 2026-07-14): wait for MemAvailable to
# stabilize after the kill -- a transient low reading shrinks the batch fit
# AND silences the live budget refresh at admit (both read the same gauge).
ssh "$R" "prev=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo); i=0
  while [ \$i -lt 24 ]; do sleep 5
    cur=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
    d=\$((cur - prev)); [ \$d -lt 0 ] && d=\$((-d))
    [ \$d -lt 512000 ] && { echo \"mem stable: \$((cur/1048576)) GiB\"; exit 0; }
    prev=\$cur; i=\$((i+1)); done
  echo \"mem NOT stable after 120s: \$((cur/1048576)) GiB (continuing)\"" | tee -a "$DRV"
SPECENV="DS4_CONT_MTP_MODE=2 DS4_CONT_DSPARK=1 DS4_DSPARK_MODEL=$DRAFTER"
SPECARG=""
[ "$SPEC" = 0 ] && { SPECENV=""; SPECARG="--no-spec"; MTP=""; }
PINS="DS4_BATCH_FIT_HEADROOM_MB=$HEADROOM_MB DS4_SERVER_COALESCE_MAX=8"
[ "$ZEROCONF" = 1 ] && PINS=""
log "boot: ctx=$CTX $([ "$ZEROCONF" = 1 ] && echo 'ZEROCONF (shipped-default fit)' || echo coalesce_max=8), lazy serial graph, $([ "$SPEC" = 0 ] && echo plain || echo ship) cont env${EXTRA_ENV:+ + $EXTRA_ENV}${EXTRA_ARGS:+ args: $EXTRA_ARGS}"
ssh "$R" ": > $SRV; cd $RT; env $EXTRA_ENV \
    $PINS \
    DS4_CONT_PREFILL_CHUNK=2048 $SPECENV \
    DS4_CONT_CAPTURE=1 DS4_SERVER_DEFAULT_TEMP=0 \
    setsid nohup ./ds4-server -m $BASE $SPECARG ${MTP:+--mtp} ${MTP:---no-mtp} --cuda -c $CTX $EXTRA_ARGS --port $PORT \
    > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
  if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
    sleep 3   # re-check: a too-eager liveness poll can false-negative mid-exec
    ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
      fail "BOOT-DIED: $(ssh "$R" "tail -3 $SRV" 2>/dev/null | tr '\n' ' ')"
  fi
  sleep 10; n=$((n+10)); [ $n -ge 1500 ] && fail "boot timeout"
done
log "boot up. Ledger:"
ssh "$R" "grep -E 'batch fit|batch vmm|persistent batch ctx' $SRV" | tee -a "$DRV"
curl -s -m 5 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1 || {
  ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true
  sleep 2
  curl -s -m 10 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null || fail "tunnel/:$TUNNEL unreachable"
}

URL="http://127.0.0.1:$TUNNEL/v1/chat/completions"
OFF=$(ssh "$R" "wc -l < $SRV")
msnap "$OUT/m0.txt" || fail "/metrics unreachable at boot"

log "STAGE 1: concurrent orchestrator + subagent needles (long: deep prefill)"
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/orch.json" "$URL" ORCH "$OUT/orch_result.json" > "$OUT/orch.out" 2>&1 &
P1=$!
sleep 5
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/sub.json" "$URL" SUB > "$OUT/sub.out" 2>&1 &
P2=$!
wait $P1; RC1=$?
wait $P2; RC2=$?
cat "$OUT/orch.out" "$OUT/sub.out" | tee -a "$DRV"
[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ] || fail "stage 1 client rc=$RC1/$RC2"

log "STAGE 2+3: orchestrator turn 2 from the ACTUAL stage-1 answer"
python3 - "$OUT/orch.json" "$OUT/orch_result.json" "$OUT/turn2.json" <<'EOF' || fail "turn2 build"
import json, sys
b = json.load(open(sys.argv[1]))
r = json.load(open(sys.argv[2]))
b["messages"].append({"role": "assistant", "content": r["text"]})
b["messages"].append({"role": "user", "content":
    "Good. Now write a detailed summary (at least 300 words) of the main "
    "technical topics covered in the archive."})
b["max_tokens"] = 512
json.dump(b, open(sys.argv[3], "w"))
EOF
OFF2=$(ssh "$R" "wc -l < $SRV")
msnap "$OUT/m_turn2_pre.txt" || fail "/metrics unreachable before turn 2"
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/turn2.json" "$URL" TURN2 > "$OUT/turn2.out" 2>&1
RC3=$?
cat "$OUT/turn2.out" | tee -a "$DRV"
msnap "$OUT/m_end.txt" || fail "/metrics unreachable after turn 2"

ssh "$R" "tail -n +$((OFF+1)) $SRV" > "$OUT/srv_seg.log"
ssh "$R" "tail -n +$((OFF2+1)) $SRV" > "$OUT/turn2_seg.log"
log "server segment health (all stages):"
for pat in 'illegal' 'continuous batch failed' 'prompt start' 'cont admit rejected' 'budget refreshed' 'warm admit' 'fork admit' 'kv-gate' 'DEEP record'; do
  printf '  %-28s %s\n' "$pat:" "$(grep -c "$pat" "$OUT/srv_seg.log")" | tee -a "$DRV"
done
grep -E 'warm admit|fork admit|cont admit bank|rejected|refreshed' "$OUT/srv_seg.log" | head -8 | tee -a "$DRV"

PASS=1
# PRIMARY health: registry counter deltas (whole run, and the turn-2 slice).
MFB=$(mdelta "$OUT/m0.txt" "$OUT/m_end.txt" ds4_cont_batch_failures_total)
MSER=$(mdelta "$OUT/m0.txt" "$OUT/m_end.txt" ds4_requests_serial_total)
MREJ=$(mdelta "$OUT/m0.txt" "$OUT/m_end.txt" ds4_cont_admit_rejects_total)
T2WARM=$(mdelta "$OUT/m_turn2_pre.txt" "$OUT/m_end.txt" 'ds4_admits_total{kind="warm"}')
T2FORK=$(( $(mdelta "$OUT/m_turn2_pre.txt" "$OUT/m_end.txt" 'ds4_admits_total{kind="fork"}') \
         + $(mdelta "$OUT/m_turn2_pre.txt" "$OUT/m_end.txt" 'ds4_admits_total{kind="partial_fork"}') ))
log "counters: fail=+$MFB serial=+$MSER rejects=+$MREJ turn2[warm=+$T2WARM fork=+$T2FORK]"
[ "$MFB" -eq 0 ]  || { log "CONT FAIL (counter +$MFB)"; PASS=0; }
[ "$MSER" -eq 0 ] || { log "SERIAL FALLBACK (counter +$MSER)"; PASS=0; }
[ "$MREJ" -eq 0 ] || { log "ADMIT REJECT (counter +$MREJ)"; PASS=0; }
[ "$T2WARM" -ge 1 ] || { log "TURN2 NOT WARM (counter +$T2WARM)"; PASS=0; }
[ "$T2FORK" -eq 0 ] || { log "TURN2 FORKED (counter +$T2FORK, deep fork guard failed)"; PASS=0; }
# FALLBACK health (one release): the stderr grep twins.
grep -q '8842-OMEGA-KILO' "$OUT/orch.out" || { log "ORCH NEEDLE MISS"; PASS=0; }
grep -q '5517-SIGMA-NOVA' "$OUT/sub.out"  || { log "SUB NEEDLE MISS"; PASS=0; }
[ $RC3 -eq 0 ] || { log "turn2 client rc=$RC3"; PASS=0; }
grep -qE 'warm admit bank=[0-9]+ cached=[0-9]{6}' "$OUT/turn2_seg.log" || { log "TURN2 NOT WARM-IN-PLACE"; PASS=0; }
[ "$(grep -c 'fork admit' "$OUT/turn2_seg.log")" -eq 0 ] || { log "TURN2 FORKED (deep fork guard failed)"; PASS=0; }
[ "$(grep -c 'illegal' "$OUT/srv_seg.log")" -eq 0 ] || { log "ILLEGAL"; PASS=0; }
[ "$(grep -c 'continuous batch failed' "$OUT/srv_seg.log")" -eq 0 ] || { log "CONT FAIL"; PASS=0; }
[ "$(grep -c 'prompt start' "$OUT/srv_seg.log")" -eq 0 ] || { log "SERIAL FALLBACK"; PASS=0; }
[ "$(grep -c 'cont admit rejected' "$OUT/srv_seg.log")" -eq 0 ] || { log "ADMIT REJECT"; PASS=0; }
log "post-gate MemAvailable: $(ssh "$R" "awk '/MemAvailable/{print \$2/1048576 \" GiB\"}' /proc/meminfo")"
if [ $PASS -eq 1 ]; then log "DEEP-CTX GATE: ALL STAGES PASS"; else log "DEEP-CTX GATE: FAIL"; fi
echo "RUN_DONE_deepctxgate exit=$((1-PASS))" >> "$SENT"
exit $((1-PASS))
