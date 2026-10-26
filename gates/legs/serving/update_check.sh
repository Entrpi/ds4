#!/bin/bash
# serving/update_check.sh — v0.5.3 inc4: version + once-daily update check.
#
#   leg stamp   (local, seconds): the Makefile version derivation yields a
#       parseable release version in every distribution tree shape — most
#       importantly an installer clone WITHOUT tags, where a bare-hash stamp
#       made the update check nag release builds daily (378855 post 30).
#   leg flags   (seconds, no model): --version prints the build version and
#       exits 0; --check-update against a file:// LATEST reports "update
#       available" for a newer tag and "up to date" for an older one;
#       --upgrade prints the installer one-liner.
#   leg boot    (-c 2048): with a newer file:// LATEST and a cleared stamp,
#       the post-listen async check prints the loud update line; a second
#       boot inside the 24 h stamp window prints NOTHING; a third boot with
#       DS4_NO_UPDATE_CHECK=1 and a cleared stamp prints NOTHING (and the
#       stamp file stays absent — opt-out never touches the network or disk).
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
SRV=/tmp/update_check_gate.log
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server 2>/dev/null; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; kill_all; exit 1; }

log "== leg stamp (local Makefile derivation) =="
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
VER=$(cat "$REPO/VERSION")
T=$(mktemp -d)
cp "$REPO/Makefile" "$REPO/VERSION" "$T/"
( cd "$T" && git init -q && git add -A \
  && git -c user.email=g@g -c user.name=g commit -qm x )
s=$(make -s -C "$T" print-version)
[[ "$s" == "$VER" ]] || fail "tag-less clone stamp: got '$s', want '$VER' (bare-hash regression)"
( cd "$T" && git tag v9.9.9 )
s=$(make -s -C "$T" print-version)
[[ "$s" == v9.9.9 ]] || fail "tagged stamp: got '$s', want v9.9.9"
rm -rf "$T/.git"
s=$(make -s -C "$T" print-version)
[[ "$s" == "$VER" ]] || fail "gitless tree stamp: got '$s', want '$VER'"
rm -rf "$T"
s=$(make -s -C "$REPO" print-version)
[[ "$s" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]] || fail "checkout stamp unparseable: '$s'"
log "stamp PASS (clone=$VER checkout=$s)"

log "== leg flags =="
v=$(ssh "$R" "cd $BINDIR && ./$BIN --version")
[[ "$v" =~ ^ds4-server\ v[0-9]+\.[0-9]+\.[0-9]+ ]] || fail "--version not a release version (bare hash?): $v"
ssh "$R" "printf 'v9.9.9\n' > /tmp/LATEST_hi; printf 'v0.0.1\n' > /tmp/LATEST_lo"
hi=$(ssh "$R" "cd $BINDIR && DS4_UPDATE_URL=file:///tmp/LATEST_hi ./$BIN --check-update")
grep -q "update available: v9.9.9" <<<"$hi" || fail "newer LATEST not reported: $hi"
lo=$(ssh "$R" "cd $BINDIR && DS4_UPDATE_URL=file:///tmp/LATEST_lo ./$BIN --check-update")
grep -q "up to date" <<<"$lo" || fail "older LATEST mis-reported: $lo"
up=$(ssh "$R" "cd $BINDIR && ./$BIN --upgrade")
grep -q "install.sh | bash" <<<"$up" || fail "--upgrade output: $up"
log "flags PASS ($v)"

boot(){ # $1 = extra env
  ssh "$R" ": > $SRV; cd $BINDIR; env $1 setsid nohup ./$BIN -c 2048 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ds4-server >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
  sleep 8   # let the detached check thread run
}

log "== leg boot (async check fires) =="
kill_all
ssh "$R" "rm -f ~/.cache/ds4/update-check"
boot "DS4_UPDATE_URL=file:///tmp/LATEST_hi"
ssh "$R" "grep -q 'update available: v9.9.9' $SRV" || fail "boot check did not fire: $(ssh "$R" "tail -3 $SRV")"
kill_all
log "boot leg 1 PASS (loud line present)"

boot "DS4_UPDATE_URL=file:///tmp/LATEST_hi"
ssh "$R" "grep -q 'update available' $SRV" && fail "stamp window not honored (second boot checked again)"
kill_all
log "boot leg 2 PASS (24 h stamp honored)"

ssh "$R" "rm -f ~/.cache/ds4/update-check"
boot "DS4_UPDATE_URL=file:///tmp/LATEST_hi DS4_NO_UPDATE_CHECK=1"
ssh "$R" "grep -q 'update available' $SRV" && fail "opt-out ignored"
ssh "$R" "test -f ~/.cache/ds4/update-check" && fail "opt-out still wrote the stamp"
kill_all
log "boot leg 3 PASS (opt-out quiet, no stamp)"
log "update_check_gate PASS"
