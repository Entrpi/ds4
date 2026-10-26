#!/bin/bash
# gates/legs/memory/bank_churn_soak.sh — mixed-depth multi-tenant churn soak
# (v0.2 charter ws3 extra: live proof of the 32e1dab placement stack —
# DS4_SERVER_PIN_MIN_TOKENS deep tier, comp-budget live refresh, deep-trunk
# fork guard — under sustained bank eviction pressure).
#
# Shape: ONE deep needle trunk (~DEEP_TOK, above the 65536 pin threshold)
# admitted first, then TENANTS distinct shallow needle tenants (~TEN_TOK each,
# far below the pin threshold, rotated corpora => no shared prefixes) fired in
# pairs, round-robin, for DURATION_MIN. Every DEEP_TOUCH_EVERY rounds the deep
# trunk gets a follow-up turn that must be a WARM IN-PLACE admit (6-digit
# cached, 0 forks) with the needle still exact.
#
# PASS gates:
#   * 0 'cold admit evicting DEEP record' lines — shallow candidates always
#     exist here, so a deep eviction = pin-tier failure.
#   * every deep touch warm-in-place (cached 6+ digits, 0 fork admits) with
#     the keystone code exact; every tenant needle exact (3K retrieval = 100%).
#   * 0 illegal / 0 'continuous batch failed' / 0 'prompt start' (serial) /
#     0 'cont admit rejected' across the whole soak.
#   * MemAvailable drift > CREEP_WARN_GIB from the post-trunk baseline is a
#     WARNING, not a failure (known GB10 MemAvailable-decline; see memory
#     gb10-memavailable-decline) — but it is printed loudly.
#
# Env overrides (defaults in parens):
#   CS_HOST (${DS4_BOX33:?})  BINDIR (${DS4_BOX_TREE:?})
#   CTX (262144)  DEEP_TOK (240000)  TENANTS (12)  TEN_TOK (3000)
#   DURATION_MIN (60)  DEEP_TOUCH_EVERY (6)  CREEP_WARN_GIB (4)
#   PORT (8000)  TUNNEL_PORT (18000)  HEADROOM_MB (6272)
#   CS_OUT (/tmp/bank_churn_soak_<ts>)  GGUF (${DS4_GGUF_DIR:?})
#   EXTRA_ENV ("") — extra server env spliced into the boot
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${CS_HOST:-${DS4_BOX33:?}}
RT=${BINDIR:-${DS4_BOX_TREE:?}}
CTX=${CTX:-262144}
DEEP_TOK=${DEEP_TOK:-240000}
TENANTS=${TENANTS:-12}
TEN_TOK=${TEN_TOK:-3000}
DURATION_MIN=${DURATION_MIN:-60}
DEEP_TOUCH_EVERY=${DEEP_TOUCH_EVERY:-6}
CREEP_WARN_GIB=${CREEP_WARN_GIB:-4}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
HEADROOM_MB=${HEADROOM_MB:-6272}
OUT=${CS_OUT:-/tmp/bank_churn_soak_$(date +%Y%m%d_%H%M%S)}
GGUF=${GGUF:-${DS4_GGUF_DIR:?}}
EXTRA_ENV=${EXTRA_ENV:-}
BASE=$GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MTP=${MTP-$GGUF/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}   # MTP="" boots --no-mtp (MTP-droppable legs; blocks launch-default auto-attach)
DRAFTER=$GGUF/DSpark-drafter-Q2K-Q8.gguf
SRV=/tmp/bank_churn_soak_srv.log
HERE=$(cd "$(dirname "$0")" && pwd)
DEEP_CODE="9174-DELTA-ECHO"

mkdir -p "$OUT"
DRV=$OUT/driver.log
SENT=$OUT/sentinel.log
CSV=$OUT/rounds.csv
rm -f "$SENT"; : > "$DRV"
echo "round,elapsed_s,memavail_gib,tenant_miss_cum,touch_fail_cum" > "$CSV"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRV"; }
fail(){ log "FAIL: $*"; echo "RUN_DONE_churnsoak exit=1" >> "$SENT"; exit 1; }
mem_gib(){ ssh "$R" "awk '/MemAvailable/{printf \"%.1f\", \$2/1048576}' /proc/meminfo"; }

# v0.2.x counter-based health (PRIMARY): /metrics snapshots + deltas mirror
# the stderr greps below, which stay as fallback for one release.
msnap(){ curl -s -m 10 "http://127.0.0.1:$TUNNEL/metrics" > "$1" && [ -s "$1" ]; }
mval(){ awk -v k="$2" '$1 == k {v=$2} END {printf "%.0f", v+0}' "$1" 2>/dev/null; }
mdelta(){ echo $(( $(mval "$2" "$3") - $(mval "$1" "$3") )); }

log "build prompts: deep ~$DEEP_TOK @50% + $TENANTS tenants ~$TEN_TOK (rotated)"
python3 "$DS4_GATES/lib/needle_prompt.py" "$DEEP_TOK" "$OUT/deep.json" 0.50 "$DEEP_CODE" | tee -a "$DRV" || fail "deep prompt"
for i in $(seq 1 "$TENANTS"); do
  python3 "$DS4_GATES/lib/needle_prompt.py" "$TEN_TOK" "$OUT/ten_$i.json" 0.50 "TEN$i-KILO-$i$i" deepseek-chat "$i" \
    >> "$DRV" || fail "tenant $i prompt"
done

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
log "boot: ctx=$CTX ship cont env (lazy serial graph, pin tier at default 65536)${EXTRA_ENV:+ + $EXTRA_ENV}"
ssh "$R" ": > $SRV; cd $RT; env $EXTRA_ENV \
    DS4_CUDA_NO_HBM_CACHE=1 \
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
MS=$(ssh "$R" "grep -m1 -oE 'max_seq=[0-9]+' $SRV" | grep -oE '[0-9]+$'); MS=${MS:-0}
log "boot up, max_seq=$MS"
[ "$MS" -ge 4 ] || fail "max_seq=$MS < 4 — need deep trunk + 2 in-flight tenants + idle shallow records"
curl -s -m 5 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1 || {
  ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true
  sleep 2
  curl -s -m 10 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null || fail "tunnel/:$TUNNEL unreachable"
}
URL="http://127.0.0.1:$TUNNEL/v1/chat/completions"
OFF0=$(ssh "$R" "wc -l < $SRV")
msnap "$OUT/m0.txt" || fail "/metrics unreachable at boot"

log "STAGE 1: admit deep trunk (cold ~${DEEP_TOK} tokens — several minutes)"
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/deep.json" "$URL" DEEP "$OUT/deep_result.json" > "$OUT/deep.out" 2>&1 \
  || fail "deep trunk client: $(tail -1 "$OUT/deep.out")"
cat "$OUT/deep.out" | tee -a "$DRV"
grep -q "$DEEP_CODE" "$OUT/deep.out" || fail "deep needle miss"
BASE_MEM=$(mem_gib)
log "deep trunk in. baseline MemAvailable=$BASE_MEM GiB"

python3 - "$OUT/deep.json" "$OUT/deep_result.json" "$OUT/touch.json" <<'EOF' || fail "touch build"
import json, sys
b = json.load(open(sys.argv[1]))
r = json.load(open(sys.argv[2]))
b["messages"].append({"role": "assistant", "content": r["text"]})
b["messages"].append({"role": "user", "content":
    "Confirm you still remember the secret keystone code; reply with just the code."})
b["max_tokens"] = 24
json.dump(b, open(sys.argv[3], "w"))
EOF

TEN_MISS=0; TOUCH_FAIL=0; ROUND=0
deep_touch(){
  local offt seg=$OUT/touch_seg.log
  offt=$(ssh "$R" "wc -l < $SRV")
  msnap "$OUT/m_touch_pre.txt" || { TOUCH_FAIL=$((TOUCH_FAIL+1)); log "DEEP TOUCH FAIL (round $ROUND): /metrics unreachable"; return; }
  python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/touch.json" "$URL" TOUCH > "$OUT/touch.out" 2>&1
  local rc=$?
  ssh "$R" "tail -n +$((offt+1)) $SRV" > "$seg"
  msnap "$OUT/m_touch_post.txt" || { TOUCH_FAIL=$((TOUCH_FAIL+1)); log "DEEP TOUCH FAIL (round $ROUND): /metrics unreachable"; return; }
  # PRIMARY: registry admit-class deltas over the touch; the stderr grep twins
  # stay as fallback for one release.
  local mwarm mtrunc mforks warm forks trunc
  mwarm=$(mdelta "$OUT/m_touch_pre.txt" "$OUT/m_touch_post.txt" 'ds4_admits_total{kind="warm"}')
  mtrunc=$(mdelta "$OUT/m_touch_pre.txt" "$OUT/m_touch_post.txt" 'ds4_admits_total{kind="partial_truncate"}')
  mforks=$(( $(mdelta "$OUT/m_touch_pre.txt" "$OUT/m_touch_post.txt" 'ds4_admits_total{kind="fork"}') \
           + $(mdelta "$OUT/m_touch_pre.txt" "$OUT/m_touch_post.txt" 'ds4_admits_total{kind="partial_fork"}') ))
  warm=$(grep -cE 'warm admit bank=[0-9]+ cached=[0-9]{6}' "$seg")
  forks=$(grep -c 'fork admit' "$seg")
  # Steady-state touch path: after the first touch the record carries the
  # generated answer, so an identical re-touch is a strict PREFIX of the
  # record and the server serves it as an in-place partial truncate
  # (cut = deep trunk, suffix ~1) -- warm-class, cheaper than a re-admit.
  trunc=$(grep -cE 'partial truncate admit bank=[0-9]+ cut=[0-9]{6}' "$seg")
  if [ $rc -ne 0 ] || ! grep -q "$DEEP_CODE" "$OUT/touch.out" || \
     [ $((mwarm + mtrunc)) -lt 1 ] || [ "$mforks" -ne 0 ] || \
     [ $((warm + trunc)) -lt 1 ] || [ "$forks" -ne 0 ]; then
    TOUCH_FAIL=$((TOUCH_FAIL+1))
    log "DEEP TOUCH FAIL (round $ROUND): rc=$rc counters[warm=$mwarm trunc=$mtrunc forks=$mforks] greps[warm=$warm trunc=$trunc forks=$forks] $(grep -m1 ttft "$OUT/touch.out")"
    cp "$seg" "$OUT/touch_seg_fail_r$ROUND.log"
  else
    log "deep touch OK (round $ROUND): warm=+$mwarm trunc=+$mtrunc $(grep -m1 ttft "$OUT/touch.out" | cut -c1-80)"
  fi
}

log "STAGE 2: churn $TENANTS tenants in pairs for ${DURATION_MIN}m (deep touch every $DEEP_TOUCH_EVERY rounds)"
T0=$SECONDS
while [ $(( (SECONDS - T0) / 60 )) -lt "$DURATION_MIN" ]; do
  ROUND=$((ROUND+1))
  i=1
  while [ $i -le "$TENANTS" ]; do
    j=$((i+1))
    python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/ten_$i.json" "$URL" "T$i" > "$OUT/ten_$i.out" 2>&1 &
    p1=$!
    p2=
    if [ $j -le "$TENANTS" ]; then
      python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/ten_$j.json" "$URL" "T$j" > "$OUT/ten_$j.out" 2>&1 &
      p2=$!
    fi
    wait $p1
    [ -n "$p2" ] && wait $p2
    for k in $i $j; do
      [ "$k" -le "$TENANTS" ] || continue
      grep -q "TEN$k-KILO-$k$k" "$OUT/ten_$k.out" || {
        TEN_MISS=$((TEN_MISS+1)); log "TENANT $k NEEDLE MISS/FAIL (round $ROUND): $(tail -1 "$OUT/ten_$k.out" | cut -c1-100)"; }
    done
    i=$((i+2))
  done
  [ $((ROUND % DEEP_TOUCH_EVERY)) -eq 0 ] && deep_touch
  M=$(mem_gib)
  echo "$ROUND,$((SECONDS - T0)),$M,$TEN_MISS,$TOUCH_FAIL" >> "$CSV"
  log "round $ROUND done: elapsed=$(( (SECONDS-T0)/60 ))m mem=${M}GiB miss=$TEN_MISS touch_fail=$TOUCH_FAIL"
done

log "STAGE 3: final deep touch + gate"
deep_touch
ssh "$R" "tail -n +$((OFF0+1)) $SRV" > "$OUT/srv_seg.log"
for pat in 'illegal' 'continuous batch failed' 'prompt start' 'cont admit rejected' \
           'cold admit evicting DEEP record' 'budget refreshed' 'warm admit' 'kv-gate'; do
  printf '  %-34s %s\n' "$pat:" "$(grep -c "$pat" "$OUT/srv_seg.log")" | tee -a "$DRV"
done

PASS=1
# PRIMARY health: registry counter deltas over the whole soak.
msnap "$OUT/m_end.txt" || { log "/metrics unreachable at end"; PASS=0; }
MFB=$(mdelta "$OUT/m0.txt" "$OUT/m_end.txt" ds4_cont_batch_failures_total)
MSER=$(mdelta "$OUT/m0.txt" "$OUT/m_end.txt" ds4_requests_serial_total)
MREJ=$(mdelta "$OUT/m0.txt" "$OUT/m_end.txt" ds4_cont_admit_rejects_total)
log "counters: fail=+$MFB serial=+$MSER rejects=+$MREJ"
[ "$MFB" -eq 0 ]  || { log "CONT FAIL (counter +$MFB)"; PASS=0; }
[ "$MSER" -eq 0 ] || { log "SERIAL FALLBACK (counter +$MSER)"; PASS=0; }
[ "$MREJ" -eq 0 ] || { log "ADMIT REJECT (counter +$MREJ)"; PASS=0; }
# FALLBACK health (one release): the stderr grep twins ('DEEP record evicted'
# and 'illegal' have no counter twin and stay grep-gated).
[ "$(grep -c 'cold admit evicting DEEP record' "$OUT/srv_seg.log")" -eq 0 ] || { log "DEEP RECORD EVICTED (pin tier failed)"; PASS=0; }
[ "$TEN_MISS" -eq 0 ] || { log "TENANT MISSES=$TEN_MISS"; PASS=0; }
[ "$TOUCH_FAIL" -eq 0 ] || { log "DEEP TOUCH FAILS=$TOUCH_FAIL"; PASS=0; }
[ "$(grep -c 'illegal' "$OUT/srv_seg.log")" -eq 0 ] || { log "ILLEGAL"; PASS=0; }
[ "$(grep -c 'continuous batch failed' "$OUT/srv_seg.log")" -eq 0 ] || { log "CONT FAIL"; PASS=0; }
[ "$(grep -c 'prompt start' "$OUT/srv_seg.log")" -eq 0 ] || { log "SERIAL FALLBACK"; PASS=0; }
[ "$(grep -c 'cont admit rejected' "$OUT/srv_seg.log")" -eq 0 ] || { log "ADMIT REJECT"; PASS=0; }
END_MEM=$(mem_gib)
DRIFT=$(python3 -c "print(f'{$BASE_MEM - $END_MEM:.1f}')")
log "MemAvailable: baseline=$BASE_MEM end=$END_MEM drift=${DRIFT} GiB (rounds=$ROUND)"
python3 -c "exit(0 if $BASE_MEM - $END_MEM <= $CREEP_WARN_GIB else 1)" || \
  log "WARNING: MemAvailable creep ${DRIFT} GiB > ${CREEP_WARN_GIB} GiB — gb10-memavailable-decline class, consider reboot before decisive legs"
if [ $PASS -eq 1 ]; then log "BANK-CHURN SOAK: PASS ($ROUND rounds)"; else log "BANK-CHURN SOAK: FAIL"; fi
echo "RUN_DONE_churnsoak exit=$((1-PASS))" >> "$SENT"
exit $((1-PASS))
