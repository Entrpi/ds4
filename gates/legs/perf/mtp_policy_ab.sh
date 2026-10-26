#!/bin/bash
# gates/legs/perf/mtp_policy_ab.sh — matched-drafter DRAFTING-POLICY A/B.
# CENSUS FACT (reference-0731-checkpoint-census.md): the 0731
# checkpoint's 3 "mtp" blocks ARE the DSpark drafter stages (19.8B,
# shards 46-48; the gridbook bridge's mtp.* names map the same layers).
# Our ship DSpark-drafter-Q2K-Q8-0731.gguf IS the matched head — no
# cross-generation arm exists or is needed.  The EXL3 recipe's edge is
# POLICY, not weights: it drafts unconditionally at width 5 (their
# 4.09 tok/step at 63.8% accept); our ship path quench-throttles
# drafting on confidence (measured ~1.67 drafts/step attempted at 80%
# accept on the throttled subset -> 2.33 tok/step).  The DSpark block
# is already width-5-shaped (D = DS4_DSPARK_BLOCK-1 = 4 drafts + 1
# committed = 5 rows/step).
#
# Arms (each on its OWN boot so the shutdown CONT_MTP_ACCEPT counters
# are per-leg), all C1, temperature 0, reasoning_effort none:
#   A   ship zero-config (quench ON, verify depth 4)   [baseline]
#   Q0  DS4_DSPARK_QUENCH=0 (always-spec: THEIR policy, matched weights)
#   V2  DS4_DSPARK_VERIFY_DEPTH=2 (the code's own small-width dial)
# x tasks: prose rewrite | codegen (their 63.8%/2.4x was codegen).
# 3 reps each (rep 1 = warmup, reported separately), 256 gen tokens.
# NOTE: quench exists for QUALITY protection under memory pressure and
# depth (spec break-even ~2.37 at 240K ctx); a Q0 win at C1-shallow
# does NOT license flipping the default -- it sizes the policy gap.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
GGUF=${GGUF:-${DS4_GGUF_DIR:?}}
MTP=$GGUF/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
OUT=${OUT:-/tmp/mtp_ab_$(date +%Y%m%d_%H%M%S)}
SRV=/tmp/mtp_ab_srv.log
HERE=$(cd "$(dirname "$0")" && pwd)
BASE="http://127.0.0.1:$TUNNEL"
URL="$BASE/v1/chat/completions"
mkdir -p "$OUT"
DRV=$OUT/driver.log
log(){ echo "[$(date +%H:%M:%S)] MTPAB: $*" | tee -a "$DRV"; }
fail(){ log "FAIL: $*"; exit 1; }

NONCE="mtpab_$(date +%s)_$$"
LOCKED=0
cleanup(){ rc=$?
  ssh "$R" "pkill -x ds4-server; sleep 1; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null
  if [ "$LOCKED" = 1 ]; then ssh "$R" "rm -rf /tmp/ds4_box_lock; exit 0" 2>/dev/null; log "box lock freed"; fi
  log "artifacts in $OUT"
  exit $rc
}
trap cleanup EXIT INT TERM
OWN=$(ssh "$R" "cat /tmp/ds4_box_lock/nonce 2>/dev/null; exit 0" 2>/dev/null)
[ -n "$OWN" ] && { log "box lock held by $OWN -- refusing"; exit 2; }
ssh "$R" "mkdir /tmp/ds4_box_lock && echo '$NONCE' > /tmp/ds4_box_lock/nonce" || fail "lock take"
LOCKED=1
log "box lock taken ($NONCE)"

boot(){ # $1=extra args (may be empty)
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0"
  ssh "$R" ": > $SRV; cd $BINDIR; env ${BOOT_ENV:-} setsid nohup ./ds4-server --cuda -c 262144 --port $PORT $1 \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || {
      sleep 3
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" 2>/dev/null | tr '\n' ' ')"; }
    sleep 10; n=$((n+10)); [ $n -ge 1500 ] && fail "boot timeout"
  done
  curl -s -m 5 "$BASE/v1/models" >/dev/null 2>&1 || {
    ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true; sleep 2
    curl -s -m 10 "$BASE/v1/models" >/dev/null || fail "tunnel unreachable"; }
}

mkreq(){ # $1=index $2=task(prose|code) -> $OUT/req_$2_$1.json
  python3 - "$1" "$2" "$OUT/req_$2_$1.json" <<'PY'
import json, sys
i, task, path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
corpus = ("The archive of expedition %d records tides, granite, lichen, "
          "and the slow accounting of weather over the ridge. " % i)
filler = (corpus * 90)[:11250]
if task == "prose":
    q = "Rewrite the passage above in your own words, expansively."
else:
    q = ("Now write a complete, well-commented Python implementation of "
         "an LRU cache class with get/put/eviction and a small unit test.")
json.dump({"max_tokens": 256, "temperature": 0, "reasoning_effort": "none",
           "messages": [{"role": "user", "content": filler + "\n\n" + q}]},
          open(path, "w"))
PY
}

run_leg(){ # $1=leg-name $2=boot-env $3=task
  log "=== leg $1 ($3): env='$2' ==="
  BOOT_ENV="$2" boot ""
  for r in 1 2 3; do
    i=$((RANDOM % 1000 + r))
    mkreq "$i" "$3"
    python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/req_$3_$i.json" "$URL" "$1-$3-r$r" "$OUT/leg_$1_$3_r$r.json" \
      | tee -a "$DRV" || fail "$1/$3 rep $r transport"
  done
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; exit 0"
  sleep 2
  ssh "$R" "grep 'CONT_MTP_ACCEPT' $SRV | tail -2; exit 0" | tee "$OUT/accept_$1_$3.log" | tee -a "$DRV"
}

for task in prose code; do
  run_leg A  ""                          "$task"
  run_leg Q0 "DS4_DSPARK_QUENCH=0"       "$task"
  run_leg V2 "DS4_DSPARK_VERIFY_DEPTH=2" "$task"
done

log "=== A/B summary ==="
python3 - "$OUT" <<'PY' | tee -a "$DRV"
import json, glob, re, sys
out = sys.argv[1]
for leg in ("A", "Q0", "V2"):
    for task in ("prose", "code"):
        fs = sorted(glob.glob(f"{out}/leg_{leg}_{task}_r*.json"))
        tgs = []
        for f in fs:
            d = json.load(open(f))
            nd, dec = d.get("decode_deltas", 0), d.get("decode_s", 0)
            if dec and nd > 1: tgs.append((nd - 1) / dec)
        acc = open(f"{out}/accept_{leg}_{task}.log").read().strip().replace("\n", " | ")
        steady = tgs[1:] if len(tgs) > 1 else tgs
        warm = f"{tgs[0]:.2f}" if tgs else "?"
        mean = sum(steady)/len(steady) if steady else 0
        print(f"{leg}/{task}: tg warm={warm} steady={mean:.2f} tok/s (n={len(steady)})")
        print(f"    accept: {acc if acc else 'NO CONT_MTP_ACCEPT LINE'}")
PY

log "MTP A/B STAMPS: COMPLETE — artifacts in $OUT"
exit 0
