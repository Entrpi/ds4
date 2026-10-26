#!/bin/bash
# gates/legs/serving/needle_sweep.sh — formal deep-context needle-retrieval suite
# (v0.2 charter ws4 quality leg: the release claim behind "exact retrieval at
# 250K/500K" — the informal 2026-07-14 single probes graduated to a matrix).
#
# ONE ship-config boot at CTX, then for each size in SIZES (token ESTIMATES;
# actuals land ~+3.6% higher per the needle_prompt.py drift law, so est 250000
# ~= 259K actual and est 500000 ~= 518K actual — both ABOVE the claim values)
# fires one needle probe per depth in DEPTHS, sequentially, all COLD admits
# (shared filler prefixes are fork-guarded at deep committed depths => every
# probe pays full prefill; ~13 min at 250K, ~40 min at 500K on GB10 — the
# full default matrix is an overnight run, ~9 h).
#
# PASS = every probe answers its keystone code exactly, plus 0 illegal /
# 0 'continuous batch failed' / 0 'prompt start' / 0 'cont admit rejected'
# across the run.
#
# The boot runs DS4_SERVER_WARM=0: this is a single-tenant COLD-retrieval
# matrix, and warm retention breaks it (2026-07-15 run 1: retained deep
# records keep their committed pages while placement steers new admits to
# FREE banks — bank slots and the page budget are different currencies —
# so probes 4+ hit 'cont admit rejected on comp-cache budget' with 90+ GiB
# nominally free. Records only yield pages via bank REUSE, which the
# free-bank preference avoids. Post-v0.2: budget-aware eviction retry.)
# With warm off every probe rides the engine-default lowest-free-bank and
# pays honest full prefill, which is exactly what the matrix measures.
#
# LAW: a release claim at ctx N needs a standing leg above the largest tested
# ctx — that is gates/legs/memory/deep_ctx.sh (518K probe + 766K concurrent at
# ctx 524288), which must ALSO be green on the release binary.
#
# Artifacts: $NS_OUT/sweep.csv (size_est,depth,actual_prompt_tokens,ttft_s,
# decode_deltas,memavail_gib,pass) + per-probe outs + server segment.
#
# Env overrides (defaults in parens):
#   NS_HOST (${DS4_BOX33:?})  BINDIR (${DS4_BOX_TREE:?})
#   CTX (524288)  SIZES ("250000 500000")
#   DEPTHS ("0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95")
#   PORT (8000)  TUNNEL_PORT (18000)  HEADROOM_MB (6272)
#   NS_OUT (/tmp/needle_sweep_<ts>)  GGUF (${DS4_GGUF_DIR:?})
#   EXTRA_ENV ("") — extra server env spliced into the boot
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${NS_HOST:-${DS4_BOX33:?}}
RT=${BINDIR:-${DS4_BOX_TREE:?}}
CTX=${CTX:-524288}
SIZES=${SIZES:-"250000 500000"}
DEPTHS=${DEPTHS:-"0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95"}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
HEADROOM_MB=${HEADROOM_MB:-6272}
OUT=${NS_OUT:-/tmp/needle_sweep_$(date +%Y%m%d_%H%M%S)}
GGUF=${GGUF:-${DS4_GGUF_DIR:?}}
EXTRA_ENV=${EXTRA_ENV:-}
BASE=$GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MTP=${MTP-$GGUF/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}   # MTP="" boots --no-mtp (MTP-droppable legs; blocks launch-default auto-attach)
DRAFTER=$GGUF/DSpark-drafter-Q2K-Q8.gguf
SRV=/tmp/needle_sweep_srv.log
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT"
DRV=$OUT/driver.log
SENT=$OUT/sentinel.log
CSV=$OUT/sweep.csv
rm -f "$SENT"; : > "$DRV"
echo "size_est,depth,actual_prompt_tokens,ttft_s,decode_deltas,memavail_gib,pass" > "$CSV"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRV"; }
fail(){ log "FAIL: $*"; echo "RUN_DONE_needlesweep exit=1" >> "$SENT"; exit 1; }

# v0.2.x counter-based health (PRIMARY): /metrics snapshots + deltas mirror
# the stderr greps below, which stay as fallback for one release.
msnap(){ curl -s -m 10 "http://127.0.0.1:$TUNNEL/metrics" > "$1" && [ -s "$1" ]; }
mval(){ awk -v k="$2" '$1 == k {v=$2} END {printf "%.0f", v+0}' "$1" 2>/dev/null; }
mdelta(){ echo $(( $(mval "$2" "$3") - $(mval "$1" "$3") )); }

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
log "boot: ctx=$CTX ship cont env${EXTRA_ENV:+ + $EXTRA_ENV}"
ssh "$R" ": > $SRV; cd $RT; env $EXTRA_ENV \
    DS4_CUDA_NO_HBM_CACHE=1 DS4_SERVER_WARM=0 \
    DS4_BATCH_FIT_HEADROOM_MB=$HEADROOM_MB DS4_SERVER_COALESCE_MAX=8 \
    DS4_CONT_PREFILL_CHUNK=2048 DS4_CONT_MTP_MODE=2 DS4_CONT_DSPARK=1 \
    DS4_DSPARK_MODEL=$DRAFTER DS4_CONT_CAPTURE=1 DS4_SERVER_DEFAULT_TEMP=0 \
    setsid nohup ./ds4-server -m $BASE ${MTP:+--mtp} ${MTP:---no-mtp} --cuda -c $CTX --port $PORT \
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
OFF0=$(ssh "$R" "wc -l < $SRV")
msnap "$OUT/m0.txt" || fail "/metrics unreachable at boot"

MISS=0; NPROBE=0
for SZ in $SIZES; do
  SZK=$((SZ / 1000))
  for D in $DEPTHS; do
    NPROBE=$((NPROBE+1))
    DD=$(python3 -c "print(f'{int(round(float('$D')*100)):02d}')")
    CODE="7${DD}2-SWEEP-${SZK}"
    TAG="s${SZK}d${DD}"
    log "probe $NPROBE: size~$SZ depth=$D code=$CODE"
    python3 "$DS4_GATES/lib/needle_prompt.py" "$SZ" "$OUT/req_$TAG.json" "$D" "$CODE" >> "$DRV" \
      || fail "prompt build $TAG"
    python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/req_$TAG.json" "$URL" "$TAG" "$OUT/res_$TAG.json" \
      > "$OUT/probe_$TAG.out" 2>&1
    rc=$?
    grep -m1 -E 'ttft|TRANSPORT' "$OUT/probe_$TAG.out" | tee -a "$DRV"
    if [ $rc -ne 0 ]; then
      ssh "$R" "pgrep -x ds4-server >/dev/null" || fail "server DIED at probe $TAG: $(ssh "$R" "tail -3 $SRV" | tr '\n' ' ')"
      MISS=$((MISS+1)); log "PROBE $TAG TRANSPORT FAIL"
      echo "$SZ,$D,,,,,0" >> "$CSV"
      continue
    fi
    P=1; grep -q "$CODE" "$OUT/probe_$TAG.out" || { P=0; MISS=$((MISS+1)); log "PROBE $TAG NEEDLE MISS"; }
    M=$(ssh "$R" "awk '/MemAvailable/{printf \"%.1f\", \$2/1048576}' /proc/meminfo")
    python3 - "$OUT/res_$TAG.json" "$SZ" "$D" "$M" "$P" >> "$CSV" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
u = r.get("usage") or {}
print(f"{sys.argv[2]},{sys.argv[3]},{u.get('prompt_tokens','')},{r.get('ttft_s','')},"
      f"{r.get('decode_deltas','')},{sys.argv[4]},{sys.argv[5]}")
EOF
  done
done

ssh "$R" "tail -n +$((OFF0+1)) $SRV" > "$OUT/srv_seg.log"
log "server segment health:"
for pat in 'illegal' 'continuous batch failed' 'prompt start' 'cont admit rejected' \
           'cold admit evicting DEEP record' 'budget refreshed'; do
  printf '  %-34s %s\n' "$pat:" "$(grep -c "$pat" "$OUT/srv_seg.log")" | tee -a "$DRV"
done

PASS=1
# PRIMARY health: registry counter deltas over the whole sweep.
msnap "$OUT/m_end.txt" || { log "/metrics unreachable at end"; PASS=0; }
MFB=$(mdelta "$OUT/m0.txt" "$OUT/m_end.txt" ds4_cont_batch_failures_total)
MSER=$(mdelta "$OUT/m0.txt" "$OUT/m_end.txt" ds4_requests_serial_total)
MREJ=$(mdelta "$OUT/m0.txt" "$OUT/m_end.txt" ds4_cont_admit_rejects_total)
log "counters: fail=+$MFB serial=+$MSER rejects=+$MREJ"
[ "$MFB" -eq 0 ]  || { log "CONT FAIL (counter +$MFB)"; PASS=0; }
[ "$MSER" -eq 0 ] || { log "SERIAL FALLBACK (counter +$MSER)"; PASS=0; }
[ "$MREJ" -eq 0 ] || { log "ADMIT REJECT (counter +$MREJ)"; PASS=0; }
# FALLBACK health (one release): the stderr grep twins.
[ "$MISS" -eq 0 ] || { log "PROBE MISSES=$MISS/$NPROBE"; PASS=0; }
[ "$(grep -c 'illegal' "$OUT/srv_seg.log")" -eq 0 ] || { log "ILLEGAL"; PASS=0; }
[ "$(grep -c 'continuous batch failed' "$OUT/srv_seg.log")" -eq 0 ] || { log "CONT FAIL"; PASS=0; }
[ "$(grep -c 'prompt start' "$OUT/srv_seg.log")" -eq 0 ] || { log "SERIAL FALLBACK"; PASS=0; }
[ "$(grep -c 'cont admit rejected' "$OUT/srv_seg.log")" -eq 0 ] || { log "ADMIT REJECT"; PASS=0; }
log "sweep matrix ($NPROBE probes):"
column -s, -t "$CSV" | tee -a "$DRV"
if [ $PASS -eq 1 ]; then log "NEEDLE SWEEP: ALL $NPROBE PROBES PASS"; else log "NEEDLE SWEEP: FAIL"; fi
echo "RUN_DONE_needlesweep exit=$((1-PASS))" >> "$SENT"
exit $((1-PASS))
