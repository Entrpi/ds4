#!/bin/bash
# gates/legs/memory/vmm_trim.sh — trim-on-evict engagement gate (v0.5.1 Inc2,
# field report #31: grow-only comp/index pool squeezes the weight page cache
# under long-running agentic churn).
#
# Shape: NO deep trunk (the pin tier is memory/bank_churn_soak.sh's business).  A
# sequence of ROUNDS x TENANTS DISTINCT cold needle tenants admitted one at a
# time against a deliberately TIGHT pinned budget (DS4_BATCH_VMM_BUDGET_MB),
# so bank churn walks the pool into the budget and admission-time trim must
# shed free banks' pages for every late admission to fit.
#
# PASS (default, trim ON):
#   * every tenant needle exact (trim never corrupts a live/kept bank)
#   * >= 1 'batch vmm: trimmed' server line   (ENGAGEMENT — the point)
#   * 0 'cont admit rejected' lines           (trim absorbed the pressure)
#   * 0 illegal / 0 'continuous batch failed' / 0 'prompt start'
# CONTROL=1 (boots DS4_BATCH_VMM_TRIM=0, same load):
#   * >= 1 'cont admit rejected' line         (pressure was REAL)
#   * 0 'batch vmm: trimmed' lines            (switch works)
#   * needle exact on every ACCEPTED tenant; rejected requests are counted,
#     not failed (the reject IS the expected observable).
#
# If the ON leg sees NEITHER a trim nor a reject, the budget never bound at
# this shape — the gate FAILS loudly; lower BUDGET_MB (the floor line in the
# boot ledger tells you the per-bank baseline) and re-run.
#
# Env overrides (defaults in parens):
#   TG_HOST (${DS4_BOX33:?})  BINDIR (${DS4_BOX_TREE:?})
#   CTX (16384)  TENANTS (10)  ROUNDS (2)  TEN_TOK (3000)  BUDGET_MB (700)
#   CONTROL (0)  PORT (8000)  TUNNEL_PORT (18000)  HEADROOM_MB (6272)
#   TG_OUT (/tmp/vmm_trim_gate_<ts>)  GGUF (${DS4_GGUF_DIR:?})  MTP ("" => --no-mtp)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${TG_HOST:-${DS4_BOX33:?}}
RT=${BINDIR:-${DS4_BOX_TREE:?}}
CTX=${CTX:-16384}
TENANTS=${TENANTS:-10}
ROUNDS=${ROUNDS:-2}
TEN_TOK=${TEN_TOK:-3000}
# Inc 0b (union credits): 700 was sized for the pre-credit per-bank
# projections, which overcharged ~2x (neighbor banks share VMM edge pages;
# measured union(4 full banks) = 339 MiB at this shape vs 4 x 280 virtual).
# 450 binds the union truth: measured 6 trims / 0 rejects / 0 serial at
# 10x2 tenants; 700 never binds (0 trims), 250 starves (16 serial bounces).
BUDGET_MB=${BUDGET_MB:-450}
CONTROL=${CONTROL:-0}
# v0.6.3 Inc 6: FIT=1 is a DIAGNOSTIC mode, NOT a battery leg.  Measured
# on receipt (08-20, three shapes): at small ctx the per-bank slab stride
# barely exceeds the 2 MiB VMM page, so only phase-aligned banks (bank 0)
# ever hold interior pages -- every other bank estimates ~0 and the
# best-fit substitution cannot engage live here.  The policy is pinned by
# the synthetic unit (test_trim_bestfit_pick, --server family; the v0.6.2
# victim-order precedent) and observable in the field via the
# 'best-fit candidate/victim/kept default' disclosure lines.  Run FIT=1
# to watch the scan's census on a live box.
# (The tail-trim predecessor was refuted on receipt: page granularity +
# the raw-ring warm-fork floor cap per-victim tail yield below one VMM
# page at any context -- see the v0.6.3 plan.)
# FITSHAPE env: FIT_BUDGET_MB (280) TEN_TOK_DEEP (15500)
# TEN_TOK_SMALL (3500) TENANTS_FIT (4) TEN_TOK_SHALLOW (1000).
FIT=${FIT:-0}
if [ "$FIT" = 1 ]; then
  BUDGET_MB=${FIT_BUDGET_MB:-280}
  TEN_TOK_DEEP=${TEN_TOK_DEEP:-15500}
  TEN_TOK_SMALL=${TEN_TOK_SMALL:-7000}
  TENANTS_FIT=${TENANTS_FIT:-4}
  TEN_TOK_SHALLOW=${TEN_TOK_SHALLOW:-1000}
  FIT_SMALL_MAX=${FIT_SMALL_MAX:-8500}
fi
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
HEADROOM_MB=${HEADROOM_MB:-6272}
OUT=${TG_OUT:-/tmp/vmm_trim_gate_$(date +%Y%m%d_%H%M%S)}
GGUF=${GGUF:-${DS4_GGUF_DIR:?}}
BASE=$GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MTP=${MTP-}
DRAFTER=$GGUF/DSpark-drafter-Q2K-Q8.gguf
SRV=/tmp/vmm_trim_gate_srv.log
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT"
DRV=$OUT/driver.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRV"; }
fail(){ log "VMM-TRIM GATE: FAIL: $*"; exit 1; }

TRIMENV=""
[ "$CONTROL" = 1 ] && TRIMENV="DS4_BATCH_VMM_TRIM=0"
log "mode=$([ "$CONTROL" = 1 ] && echo CONTROL || echo TRIM-ON) budget=${BUDGET_MB}MB ctx=$CTX tenants=${TENANTS}x${ROUNDS} @${TEN_TOK}tok"

if [ "$FIT" = 1 ]; then
  N=$TENANTS_FIT
  log "build FIT shape: deep @${TEN_TOK_DEEP}tok + small @${TEN_TOK_SMALL}tok + $N shallow @${TEN_TOK_SHALLOW}tok"
  python3 "$DS4_GATES/lib/needle_prompt.py" "$TEN_TOK_DEEP" "$OUT/ten_deep.json" 0.30 "TGDEEP-VMM-9999" deepseek-chat 99 \
    >> "$DRV" || fail "deep tenant prompt"
  python3 "$DS4_GATES/lib/needle_prompt.py" "$TEN_TOK_SMALL" "$OUT/ten_small.json" 0.50 "TGSMALL-VMM-8888" deepseek-chat 88 \
    >> "$DRV" || fail "small tenant prompt"
  for i in $(seq 1 "$N"); do
    python3 "$DS4_GATES/lib/needle_prompt.py" "$TEN_TOK_SHALLOW" "$OUT/ten_$i.json" 0.50 "TG$i-VMM-$((1000+i))" deepseek-chat "$i" \
      >> "$DRV" || fail "tenant $i prompt"
  done
else
N=$((TENANTS * ROUNDS))
log "build $N distinct tenant prompts (rotated corpora, unique codes)"
for i in $(seq 1 "$N"); do
  python3 "$DS4_GATES/lib/needle_prompt.py" "$TEN_TOK" "$OUT/ten_$i.json" 0.50 "TG$i-VMM-$((1000+i))" deepseek-chat "$i" \
    >> "$DRV" || fail "tenant $i prompt"
done
fi

log "boot: killing old ds4-server on $R"
ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0"
ssh "$R" "prev=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo); i=0
  while [ \$i -lt 24 ]; do sleep 5
    cur=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
    d=\$((cur - prev)); [ \$d -lt 0 ] && d=\$((-d))
    [ \$d -lt 512000 ] && { echo \"mem stable: \$((cur/1048576)) GiB\"; exit 0; }
    prev=\$cur; i=\$((i+1)); done
  echo \"mem NOT stable after 120s (continuing)\"" | tee -a "$DRV"
log "boot: ctx=$CTX pinned budget ${BUDGET_MB}MB${TRIMENV:+ + $TRIMENV}"
ssh "$R" ": > $SRV; cd $RT; env DS4_BATCH_VMM_BUDGET_MB=$BUDGET_MB $TRIMENV \
    DS4_CUDA_NO_HBM_CACHE=1 \
    DS4_BATCH_FIT_HEADROOM_MB=$HEADROOM_MB DS4_SERVER_COALESCE_MAX=8 \
    DS4_CONT_PREFILL_CHUNK=2048 DS4_CONT_MTP_MODE=2 DS4_CONT_DSPARK=1 \
    DS4_DSPARK_MODEL=$DRAFTER DS4_CONT_CAPTURE=1 DS4_SERVER_DEFAULT_TEMP=0 \
    setsid nohup ./ds4-server -m $BASE ${MTP:+--mtp} ${MTP:---no-mtp} --cuda -c $CTX --port $PORT \
    > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
  if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
    sleep 3
    ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
      fail "BOOT-DIED: $(ssh "$R" "tail -3 $SRV" 2>/dev/null | tr '\n' ' ')"
  fi
  sleep 10; n=$((n+10)); [ $n -ge 1500 ] && fail "boot timeout"
done
ssh "$R" "grep -E 'batch vmm|batch fit|persistent batch ctx' $SRV" | tee -a "$DRV"
curl -s -m 5 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1 || {
  ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true
  sleep 2
  curl -s -m 10 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null || fail "tunnel/:$TUNNEL unreachable"
}
URL="http://127.0.0.1:$TUNNEL/v1/chat/completions"
OFF0=$(ssh "$R" "wc -l < $SRV")

# ---------------------------- FIT=1 leg --------------------------------------
if [ "$FIT" = 1 ]; then
  log "FIT: admit the deep tenant FIRST (oldest activity = default recency pick)"
  python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/ten_deep.json" "$URL" "TDEEP" > "$OUT/ten_deep.out" 2>&1 \
    || fail "deep tenant rc=$?: $(tail -1 "$OUT/ten_deep.out")"
  grep -q "TGDEEP-VMM-9999" "$OUT/ten_deep.out" || fail "deep tenant needle miss"

  log "FIT: admit the small tenant (the covering victim best-fit must prefer)"
  python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/ten_small.json" "$URL" "TSMALL" > "$OUT/ten_small.out" 2>&1 \
    || fail "small tenant rc=$?: $(tail -1 "$OUT/ten_small.out")"
  grep -q "TGSMALL-VMM-8888" "$OUT/ten_small.out" || fail "small tenant needle miss"

  # a just-finished bank sits in the pending->live seed window still holding
  # its admission credit, and credited banks are never trim victims -- give
  # the credit time to lapse or the small bank is invisible to the scan.
  log "FIT: 10s credit-lapse pause before the storm"
  sleep 10

  log "FIT: shallow storm ($N tenants) against the ${BUDGET_MB}MB budget"
  MISS=0
  for i in $(seq 1 "$N"); do
    python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/ten_$i.json" "$URL" "T$i" > "$OUT/ten_$i.out" 2>&1 \
      || { MISS=$((MISS+1)); log "tenant $i: client rc!=0: $(tail -1 "$OUT/ten_$i.out")"; continue; }
    grep -q "TG$i-VMM-$((1000+i))" "$OUT/ten_$i.out" || { MISS=$((MISS+1)); log "tenant $i: NEEDLE MISS"; }
  done

  ssh "$R" "tail -n +$((OFF0+1)) $SRV" > "$OUT/srv_seg.log"
  FITS=$(grep -c 'batch vmm: best-fit victim' "$OUT/srv_seg.log" || true)
  REJL=$(grep -c 'cont admit rejected' "$OUT/srv_seg.log" || true)
  ILL=$(grep -ci 'illegal' "$OUT/srv_seg.log" || true)
  BF=$(grep -c 'continuous batch failed' "$OUT/srv_seg.log" || true)
  FIRSTHIST=$(grep -m1 'batch vmm: trim victim' "$OUT/srv_seg.log" | grep -oE 'hist=[0-9]+' | grep -oE '[0-9]+' || echo 0)
  grep -E 'batch vmm: (best-fit victim|trim victim|trimmed)' "$OUT/srv_seg.log" | head -8 | tee -a "$DRV"

  log "cleanup: killing ds4-server on $R"
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null

  [ "$ILL" -eq 0 ] && [ "$BF" -eq 0 ] || fail "health counters dirty (illegal=$ILL batch-failed=$BF)"
  [ "$MISS" -eq 0 ] || fail "tenant misses: $MISS"
  [ "$REJL" -eq 0 ] || fail "admit-rejects=$REJL with trim ON"
  [ "$FITS" -ge 1 ] || fail "0 'best-fit victim' lines — the substitution never engaged; retune FIT_BUDGET_MB"
  [ "${FIRSTHIST:-0}" -ge 1 ] || fail "no 'trim victim' line found"
  [ "$FIRSTHIST" -lt "$FIT_SMALL_MAX" ] \
    || fail "first victim hist=$FIRSTHIST >= $FIT_SMALL_MAX — the deep trunk died despite a covering small victim"
  log "VMM-TRIM GATE (FIT): ALL PASS — best-fit substitutions=$FITS, first victim hist=$FIRSTHIST (< $FIT_SMALL_MAX), rejects=0, needles exact"
  exit 0
fi
# ------------------------------------------------------------------------------

MISS=0; REJ=0; OKN=0
for i in $(seq 1 "$N"); do
  python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/ten_$i.json" "$URL" "T$i" > "$OUT/ten_$i.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    if [ "$CONTROL" = 1 ]; then REJ=$((REJ+1)); log "tenant $i: rejected/errored (rc=$rc — expected under CONTROL)"; continue; fi
    log "tenant $i: client rc=$rc: $(tail -1 "$OUT/ten_$i.out")"
    MISS=$((MISS+1)); continue
  fi
  if grep -q "TG$i-VMM-$((1000+i))" "$OUT/ten_$i.out"; then OKN=$((OKN+1)); else
    MISS=$((MISS+1)); log "tenant $i: NEEDLE MISS"; fi
done
log "tenants: ok=$OKN miss=$MISS rejected=$REJ of $N"

ssh "$R" "tail -n +$((OFF0+1)) $SRV" > "$OUT/srv_seg.log"
TRIMS=$(grep -c 'batch vmm: trimmed' "$OUT/srv_seg.log" || true)
REJL=$(grep -c 'cont admit rejected' "$OUT/srv_seg.log" || true)
ILL=$(grep -ci 'illegal' "$OUT/srv_seg.log" || true)
BF=$(grep -c 'continuous batch failed' "$OUT/srv_seg.log" || true)
SER=$(grep -c 'prompt start' "$OUT/srv_seg.log" || true)
log "server: trimmed-lines=$TRIMS admit-rejects=$REJL illegal=$ILL batch-failed=$BF serial=$SER"
grep 'batch vmm: trimmed' "$OUT/srv_seg.log" | head -5 | tee -a "$DRV"

log "cleanup: killing ds4-server on $R"
ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null

[ "$ILL" -eq 0 ] && [ "$BF" -eq 0 ] || fail "health counters dirty"
if [ "$CONTROL" = 1 ]; then
  # A rejected admission bounces to the serial path ('prompt start' lines) —
  # that IS the pre-trim degraded-but-correct behavior, so serial lines are
  # the expected observable here, not a health failure.
  [ "$TRIMS" -eq 0 ] || fail "CONTROL leg trimmed ($TRIMS) — kill switch broken"
  [ "$REJL" -ge 1 ] || fail "CONTROL leg saw 0 rejects — budget never bound; lower BUDGET_MB"
  [ "$MISS" -eq 0 ] || fail "tenant needle misses: $MISS"
  log "VMM-TRIM GATE (CONTROL): PASS — rejects=$REJL serial-bounces=$SER trims=0, needles exact"
else
  [ "$SER" -eq 0 ] || fail "serial starts=$SER with trim ON — admissions still bouncing"
  [ "$MISS" -eq 0 ] || fail "tenant needle misses: $MISS"
  [ "$REJL" -eq 0 ] || fail "admit-rejects=$REJL with trim ON — trim did not absorb the pressure"
  [ "$TRIMS" -ge 1 ] || fail "0 trimmed lines AND 0 rejects — budget never bound at this shape; lower BUDGET_MB"
  log "VMM-TRIM GATE: ALL PASS — trims=$TRIMS rejects=0 serial=0 needles $OKN/$N"
fi
