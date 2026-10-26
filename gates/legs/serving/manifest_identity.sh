#!/bin/bash
# gates/legs/serving/manifest_identity.sh — rider #48 gate: weight-server
# manifest content identity.  Three legs against ONE live weight server
# (the upload is the expensive part; the manifest file is edited between
# engine boots to simulate the failure modes):
#   A match:  fresh manifest imports with "(content identity verified)".
#   B tamper: one flipped fingerprint hex digit -> the engine boot REFUSES
#             (CONTENT IDENTITY MISMATCH + base import failure + exit),
#             which is the stale-weight-server incident signature inverted.
#   C legacy: content record stripped (pre-#48 manifest) -> imports with
#             the "no content fingerprint" notice and NO verified tag.
# Runs FROM the Mac.  End state: WS + engine killed by PID, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
GGUF=${GGUF:-${DS4_TEXT_BASE:?}}
PORT=${PORT:-8010}
MAN=/tmp/ws48.manifest
WSLOG=/tmp/ws48_ws.log
SRV=/tmp/ws48_srv.log
# LAUNCH LAW (measured on .33, runs 1-3): a daemonizing ssh must be
# PLAIN ssh + `setsid --fork` + a local alarm.  Two hang modes were
# reproduced: `& echo $!` keeps the remote wrapper bash in do_wait on
# the live daemon (runs 1-2, one 5.5 h wedge), and ssh -o ServerAlive*
# options hang the client even with `& exit 0` (run 3, repro exp5 vs
# exp3/4: same launch line, plain ssh returns, optioned ssh hangs).
# setsid --fork removes the wrapper's live child entirely; the alarm
# turns any residual wedge into a loud per-call failure.
SSH(){ perl -e 'alarm 90; exec @ARGV or die' ssh "$@"; }
# REUSE_WS=1 adopts an already-running weight server + manifest instead
# of re-uploading (the upload is the expensive leg-independent part).
REUSE_WS=${REUSE_WS:-0}

log(){ echo "[$(date +%H:%M:%S)] $*"; }
WS_PID=""; ENG_PID=""
cleanup(){
  [ -n "$ENG_PID" ] && SSH "$R" "kill $ENG_PID 2>/dev/null; exit 0"
  [ -n "$WS_PID" ] && SSH "$R" "kill $WS_PID 2>/dev/null; exit 0"
}
fail(){ log "FAIL: $*"; cleanup; exit 1; }

# --- weight server up (once) -------------------------------------------------
if [ "$REUSE_WS" = "1" ]; then
  log "REUSE_WS=1: adopting the running weight server + manifest"
  WS_PID=$(SSH "$R" "pgrep -x ds4_weight_serv | head -1")
  [ -n "$WS_PID" ] || fail "REUSE_WS=1 but no ds4_weight_server running"
  SSH "$R" "grep -q 'ready manifest=' $WSLOG" || fail "adopted WS never reached ready"
else
  log "leg A: start ds4_weight_server (base scope) and wait for the manifest"
  SSH "$R" "rm -f $MAN $WSLOG; cd $BINDIR && setsid --fork nohup ./ds4_weight_server \
      --base $GGUF --scope base --manifest $MAN --no-lock \
      > $WSLOG 2>&1 < /dev/null; exit 0" || fail "ws launch"
  sleep 2
  WS_PID=$(SSH "$R" "pgrep -x ds4_weight_serv | head -1")
  [ -n "$WS_PID" ] || fail "no ws pid"
  n=0
  until SSH "$R" "grep -q 'ready manifest=' $WSLOG" 2>/dev/null; do
    SSH "$R" "kill -0 $WS_PID" 2>/dev/null || \
      fail "WS died: $(SSH "$R" "tail -3 $WSLOG" | tr '\n' ' ')"
    sleep 15; n=$((n+15)); [ $n -ge 3600 ] && fail "ws upload timeout"
  done
fi
SSH "$R" "grep 'content fingerprint base' $WSLOG" || fail "WS logged no fingerprint"
SSH "$R" "grep -q '^content base .* fnv1a-p16m-v1 ' $MAN" || fail "manifest has no content record"
SSH "$R" "cp $MAN ${MAN}.orig"
log "WS up (pid $WS_PID), manifest carries the content record"

launch_engine(){ # $1 = leg tag; fire-and-forget (a refused boot may die in
  # under a second, so PID capture happens only in legs that expect life)
  SSH "$R" ": > $SRV; cd $BINDIR && env DS4_CUDA_WEIGHT_IPC_MANIFEST=$MAN \
      DS4_CUDA_WEIGHT_IPC_SCOPE=base DS4_CUDA_WEIGHT_IPC_NO_DRAFTER=1 \
      setsid --fork nohup ./ds4-server --cuda -m $GGUF -c 4096 --port $PORT \
      > $SRV 2>&1 < /dev/null; exit 0" || fail "$1 engine launch"
}
wait_listening(){ # $1 = leg tag; requires the boot to reach listening, then
  # captures ENG_PID (liveness by name during the wait; kills stay by PID)
  local n=0
  until SSH "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    SSH "$R" "pgrep -x ds4-server >/dev/null" 2>/dev/null || \
      fail "$1 engine died: $(SSH "$R" "tail -4 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "$1 boot timeout"
  done
  ENG_PID=$(SSH "$R" "pgrep -x ds4-server | head -1")
  [ -n "$ENG_PID" ] || fail "$1 listening but no pid"
}
kill_engine(){
  SSH "$R" "kill $ENG_PID 2>/dev/null; sleep 1; kill -9 $ENG_PID 2>/dev/null; exit 0"
  ENG_PID=""
}

# --- leg A: match ------------------------------------------------------------
launch_engine A; wait_listening A
SSH "$R" "grep 'imported shared' $SRV" || fail "A: no import line"
SSH "$R" "grep -q 'content identity verified' $SRV" || fail "A: import not verified"
kill_engine
log "leg A PASS: import verified"

# --- leg B: tampered fingerprint -> refuse -----------------------------------
log "leg B: flip one fingerprint hex digit; the boot must refuse"
SSH "$R" "python3 - $MAN <<'EOF'
import sys
p = sys.argv[1]
out = []
for ln in open(p):
    if ln.startswith('content base '):
        f = ln.split()
        f[4] = f[4][:-1] + ('0' if f[4][-1] != '0' else '1')
        ln = ' '.join(f) + '\n'
    out.append(ln)
open(p, 'w').writelines(out)
EOF" || fail "B tamper edit"
SSH "$R" "cmp -s $MAN ${MAN}.orig" && fail "B: tamper edit changed nothing"
launch_engine B
# Verdict loop on EVIDENCE, not a PID: the refused boot can die faster
# than any capture window (run 4: gone within 2 s of launch).
n=0
while :; do
  SSH "$R" "grep -q 'listening on http' $SRV" 2>/dev/null && \
    fail "B: engine reached listening on a tampered manifest"
  if SSH "$R" "grep -q 'CONTENT IDENTITY MISMATCH for base' $SRV" 2>/dev/null && \
     ! SSH "$R" "pgrep -x ds4-server >/dev/null" 2>/dev/null; then
    break
  fi
  sleep 5; n=$((n+5)); [ $n -ge 600 ] && \
    fail "B: no refusal verdict in 600s: $(SSH "$R" "tail -4 $SRV" | tr '\n' ' ')"
done
SSH "$R" "grep -q 'failed to import shared base weight cache' $SRV" || fail "B: no abort tell"
log "leg B PASS: tampered manifest refused, boot aborted with the tell"

# --- leg C: legacy manifest (no content record) -> notice + import -----------
log "leg C: strip the content record; the boot must import with the notice"
SSH "$R" "grep -v '^# content' ${MAN}.orig | grep -v '^content ' > $MAN"
launch_engine C; wait_listening C
SSH "$R" "grep -q 'carries no content fingerprint' $SRV" || fail "C: no legacy notice"
SSH "$R" "grep 'imported shared' $SRV | grep -q 'content identity verified'" && \
  fail "C: verified tag on an unverified import"
SSH "$R" "grep -q 'imported shared' $SRV" || fail "C: no import line"
kill_engine
log "leg C PASS: legacy manifest imports with the notice, no verified tag"

cleanup
log "MANIFEST IDENTITY GATE PASS — A verified import, B tamper refused at boot, C legacy manifest compatible"
