#!/usr/bin/env bash
# gates/legs/memory/soak_drive.sh — drive memory/soak.sh (the memgov soak) ON the server box from a driver
# box: launch it detached there, poll its driver.log for the verdict, pull the kept logs back.  The two
# battery members soak_lite.sh (4 x 8 min cycles, ~1 h) and soak_24h.sh (the standing 24 h shape) set the
# shape and exec this file.  SOAK SHAPE law: >= 4 cycles (the observe slice sits at CYCLES/2+1 and an
# observe-phase cycle never pulses).  Env: CYCLES (pins the count) or HOURS, CYCLE_MIN, PULSE_EVERY (the soak's own knobs), MODEL_ARGS
# (spliced into the soak's EXTRA_ARGS), POLL_CAP_MIN.
VERDICT_TAG=${VERDICT_TAG:-SOAK}; . "$(dirname "$0")/../../lib/verdict.sh"
R=${RB_SERVER:-${R:-${DS4_BOX33:?}}}; TREE=${RB_TREE:-${BINDIR:-${DS4_BOX_TREE:?}}}; NONCE=${RB_NONCE:-}
LOCK=${RB_LOCK_DIR:-$DS4_BOX_LOCK_DIR}; OUT=${RB_OUT:-${OUT:-/tmp/soak_$$}}; mkdir -p "$OUT"
MODEL_ARGS=${MODEL_ARGS:--m ${RB_VBASE:-${DS4_VISION_BASE:?}} --vision ${RB_ENC:-${DS4_VISION_ENCODER:?}}}
SHAPE="${CYCLES:+CYCLES=$CYCLES }HOURS=${HOURS:-2} CYCLE_MIN=${CYCLE_MIN:-30} PULSE_EVERY=${PULSE_EVERY:-2}${BANKS:+ BANKS=$BANKS}${CTX:+ CTX=$CTX}"; POLL_CAP_MIN=${POLL_CAP_MIN:-200}   # BANKS/CTX: diagnostic shapes (the standing member requests the soak's defaults)
log "soak on .$R: $SHAPE args='$MODEL_ARGS' out=$OUT"
# the soak takes its own box lock; ours is re-asserted when it ends
ssh -o BatchMode=yes "$R" "rm -rf $LOCK /tmp/ds4_soak; cd $TREE && setsid --fork bash -c '$SHAPE EXTRA_ARGS=\"$MODEL_ARGS\" bash gates/legs/memory/soak.sh > /tmp/rb_soak.out 2>&1 < /dev/null' > /dev/null 2>&1 < /dev/null; echo launched" | sed 's/^/   /'
n=0; until ssh -o BatchMode=yes "$R" "grep -qE 'MEMGOV SOAK: (ALL ASSERTS PASS|FAILED)' /tmp/rb_soak.out /tmp/ds4_soak/driver.log 2>/dev/null"; do
  n=$((n+1)); [ $n -gt "$POLL_CAP_MIN" ] && { log "poll cap ${POLL_CAP_MIN} min reached"; break; }; sleep 60; done
ssh -o BatchMode=yes "$R" "cat /tmp/rb_soak.out" > "$OUT/soak.out" 2>/dev/null
mkdir -p "$OUT/keep"; scp -q -o BatchMode=yes "$R:/tmp/ds4_soak/driver.log" "$R:/tmp/ds4_soak/soak_samples.csv" "$R:/tmp/ds4_soak/srv_*.log" "$OUT/keep/" 2>/dev/null
[ -n "$NONCE" ] && ssh -o BatchMode=yes "$R" "mkdir -p $LOCK && echo \"$NONCE\" > $LOCK/nonce" 2>/dev/null
v=$(grep -E "MEMGOV SOAK:" "$OUT/soak.out" "$OUT/keep/driver.log" 2>/dev/null | sed 's/^[^:]*://' | tail -1)
log "${v:-MEMGOV SOAK: FAILED (no verdict line; poll cap or a dead soak)}"
case "$v" in *"ALL ASSERTS PASS"*) exit 0;; *) exit 1;; esac
