#!/bin/bash
# battery: units_goldens tier=1 stop=1 server=33 driver=33 trigger=always est=7 seq=10
# verdict: ^\[.*\] 3A_LEG_(PASS|FAIL)
# catches: silent numeric drift on the primary box: the full engine-free unit suite plus the recorded top-k/logit goldens on the 0731 text base and the Vision base (parsers, kvstore, prompt builders, span validation, memory units)
# cmd: RD="$RB_OUT" bash gates/legs/vision/units.sh
# Ported into gates/legs/vision/ on 2026-09-04 (release-process §3.0): the Vision-Exp release-battery member.
# Runs ON the GB10 box-side (the tree is the rsync mirror at $T); helper scripts live beside this file.
# V8 step 2 = release-process 3a on .33: full ./ds4_test (0731 base via the ds4flash.gguf link),
# golden 0731 verify, Vision-Exp golden verify. Sentinel: $RD/leg.DONE, verdict line 3A_LEG_PASS/FAIL.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HERE=$(cd "$(dirname "$0")" && pwd)
T=${BINDIR:-${DS4_BOX_TREE:?}}
VBASE=${VBASE:-${DS4_VISION_BASE:?}}
RD=${RD:-${DS4_BOX_SCRATCH:?}/vision_units_$(date +%Y%m%d_%H%M)}; mkdir -p "$RD"
OUT=$RD/leg.log; SENT=$RD/leg.DONE; rm -f "$SENT"; : > "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
export PATH=/usr/local/cuda/bin:$PATH
[ -z "$(pgrep -x ds4-server)" ] || { log "FAIL: a ds4-server is up"; log "3A_LEG_FAIL (precondition)"; touch "$SENT"; exit 1; }
cd "$T"
log "== provenance: $(cuobjdump ds4_cuda.o | grep -m1 arch) $(md5sum ds4.c ds4_cuda.cu | cut -c1-12 | tr "\n" " ")"
fails=0
log "== full ./ds4_test (0731 base)"
timeout 5400 ./ds4_test > "$RD/units_full.log" 2>&1; rc=$?; echo "rc=$rc" >> "$RD/units_full.log"
log "   rc=$rc $(tail -2 "$RD/units_full.log" | tr "\n" " " | cut -c1-200)"; [ $rc = 0 ] || fails=$((fails+1))
log "== golden 0731 (local-golden-0731.vec)"
DS4_TEST_LOCAL_GOLDEN_FILE=tests/test-vectors/local-golden-0731.vec timeout 1800 ./ds4_test --local-golden-vectors > "$RD/golden_0731.log" 2>&1; rc=$?; echo "rc=$rc" >> "$RD/golden_0731.log"
log "   rc=$rc $(grep -o "top1 ref=.*" "$RD/golden_0731.log" | tail -1)"; [ $rc = 0 ] || fails=$((fails+1))
grep -q "top20_max_abs=0" "$RD/golden_0731.log" || { log "   golden 0731 NOT bit-exact"; fails=$((fails+1)); }
log "== golden Vision-Exp (local-golden-vision-exp.vec on the Vision base)"
DS4_TEST_MODEL="$VBASE" DS4_TEST_LOCAL_GOLDEN_FILE=tests/test-vectors/flash-vision-exp/local-golden-vision-exp.vec timeout 1800 ./ds4_test --local-golden-vectors > "$RD/golden_vision.log" 2>&1; rc=$?; echo "rc=$rc" >> "$RD/golden_vision.log"
log "   rc=$rc $(grep -o "top1 ref=.*" "$RD/golden_vision.log" | tail -1)"; [ $rc = 0 ] || fails=$((fails+1))
grep -q "top20_max_abs=0" "$RD/golden_vision.log" || { log "   golden vision NOT bit-exact"; fails=$((fails+1)); }
[ $fails = 0 ] && log "3A_LEG_PASS" || log "3A_LEG_FAIL ($fails)"
log "== LEG_DONE"; touch "$SENT"
[ $fails = 0 ]   # rc follows the verdict (review 09-05, Grok 5)
