#!/bin/bash
# battery: teb tier=2 stop=0 server=33 driver=15 trigger=always est=40 seq=20
# verdict: ^\[.*\] 3D_(PASS|FAIL)
# catches: end-to-end agent scoring on real tool workflows through scaffold + spec-decode + serving in the shipping config (crash/fast/think legs, scores in band, DSpark engaged, counters clean); the only leg that scores agent behaviour
# cmd: RD="$RB_OUT" bash gates/legs/agent/teb_vision.sh
# V8 step 7 = release-process 3d teb restamp on the Vision set, driver ON .15 (detached; teb checkout +
# .venv live here), server ON .33 in the shipping config (Vision base + MarkovQ8 vision drafter via the
# $DS4_GGUF_VISION_DIR symlink dir, MTP= boots --no-mtp) WITH the encoder sidecar (--vision).  Sentinel $RD/leg.DONE.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
T=${BINDIR:-${DS4_BOX_TREE:?}}
RD=${RD:-$HOME/v8_teb_$(date +%Y%m%d_%H%M)}; mkdir -p "$RD"
OUT=$RD/leg.log; SENT=$RD/leg.DONE; rm -f "$SENT"; : > "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
export TEB_GATE_HOST=${DS4_BOX33_LAN:?} GGUF=${DS4_GGUF_VISION_DIR:?} MTP= TEB_SAFETY_KNOWN="TC-58" BINDIR=${DS4_BOX_TREE:?}
export EXTRA_ARGS="--vision ${DS4_VISION_ENCODER:?}"
[ -z "$(ssh $TEB_GATE_HOST pgrep -x ds4-server)" ] || { log "FAIL: a ds4-server is up on .33"; log "3D_FAIL (precondition)"; touch "$SENT"; exit 1; }
log "== teb restamp out=$RD gguf=$GGUF args=$EXTRA_ARGS binary=$(ssh $TEB_GATE_HOST "md5sum $BINDIR/ds4-server | cut -c1-16")"
cd "$T"
TEB_GATE_OUT=$RD/teb bash gates/legs/agent/teb.sh > $RD/teb.out 2>&1; rc=$?
log "   rc=$rc"
grep -E "leg|score|rc=|illegal|serial_starts|admit_rejects|CONT_MTP_ACCEPT|TEB GATE|PASS|FAIL|TC-" $RD/teb/driver.log 2>/dev/null | tail -24 | cut -c1-200 | sed 's/^/   /' | tee -a "$OUT"
ssh $TEB_GATE_HOST "for p in \$(pgrep -x ds4-server); do kill \$p; done; exit 0" 2>/dev/null
log "   servers left on .33: $(ssh $TEB_GATE_HOST pgrep -cx ds4-server)"
[ $rc = 0 ] && log "3D_PASS (rc=0; scores adjudicated against the 81-86 band from the lines above)" || log "3D_FAIL (rc=$rc)"
log "== LEG_DONE"; touch "$SENT"
exit $rc   # rc follows the verdict (review 09-05, Grok 5)
