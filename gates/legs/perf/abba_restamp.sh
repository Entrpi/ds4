#!/bin/bash
# perf/abba_restamp.sh — standalone ABBA decode re-stamp: the d0a_census_gate
# (d2) leg, extracted verbatim so it can run at a NEW tip without the
# gate's decision-oracle legs (whose byte-diff families predate later
# log lines and would false-fail — oracle staleness, not engine defect).
#
# Chartered 2026-08-17 (task #83): the #32 top-k band fix changes the
# indexer topk dispatcher on the decode hot path; the no-new-cost
# argument (scan scales with LIVE rows, not the band) must be MEASURED
# at the tagged tree per release-process 3e.
#
# Method (identical to (d2), lessons preserved): per-request
# timings.decode_tok_s (never the 60s window gauge), both sides
# --no-spec --no-mtp (accept-rate variance busts a 5% band), one warmup
# request per boot, stable-start settle guard before every boot, T C C T
# order, subshell-sentinel validation in the PARENT, clocks logged.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HOST=${HOST:-${DS4_BOX33:?}}
TEST_TREE=${DS4_BOX_TREE:?}
CTRL_TREE=${CTRL_TREE:-'~/code/ds4-cebaff7base'}
CTX=${CTX:-32768}
PORT=8000
TS=$(date +%s); NONCE="abbarestamp-$TS-$$"
SPID=""
R(){ ssh -o ConnectTimeout=15 "$HOST" "$@"; }
log(){ echo "[$(date +%H:%M:%S)] abba_restamp: $*"; }
unlock(){ R "rm -rf /tmp/ds4_box_lock" 2>/dev/null; }
die(){ log "FAIL: $*"; kill_server; unlock; exit 1; }

boot_tree() { # $1 tree  $2 env  $3 ctx  $4 log
    ssh -o ConnectTimeout=10 "$HOST" \
        "cd $1 && setsid nohup env $2 ./ds4-server -c $3 $BOOT_FLAGS > $4 2>&1 < /dev/null & exit 0" &
    local lp=$!; sleep 5; kill $lp 2>/dev/null || true
    for i in $(seq 240); do
        R "grep -q 'listening on http' $4" && { SPID=$(R "pgrep -x ds4-server | head -1"); return 0; }
        R "ps -eo comm | grep -q '^ds4-server'" || { R "tail -5 $4"; return 1; }
        sleep 2
    done
    return 1
}
kill_server() {
    [ -n "$SPID" ] && R "kill $SPID" 2>/dev/null
    for i in $(seq 20); do R "ps -eo comm | grep -q '^ds4-server'" || { SPID=""; return 0; }; sleep 2; done
    R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill -9'; SPID=""
}
avail_mb() { R "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo"; }
abba_settle() {
    for i in $(seq 60); do
        [ "$(avail_mb)" -ge 90000 ] && return 0
        sleep 5
    done
    return 1
}
BOOT_FLAGS=""
abba_ms_tok() { # $1 tree -> echoes decode_tok_s of a measured 256-token request
    kill_server
    abba_settle || { echo UNSETTLED; return 1; }
    BOOT_FLAGS="--no-spec --no-mtp"
    if ! boot_tree "$1" "" "$CTX" /tmp/abba_restamp_srv.log; then
        BOOT_FLAGS=""
        echo BOOTFAIL
        return 1
    fi
    BOOT_FLAGS=""
    R "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null || echo clocks=n/a" >&2
    R "curl -s -m 300 localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Warm up.\"}],\"max_tokens\":128}'" > /dev/null
    local out
    out=$(R "curl -s -m 300 localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Write a 400-word story about a ledger.\"}],\"max_tokens\":256}'")
    echo "$out" | grep -q finish_reason || { echo REQFAIL; return 1; }
    echo "$out" | python3 -c "import json,sys; print(json.load(sys.stdin)['timings']['decode_tok_s'])"
}
abba_num() {
    echo "$2" | grep -qE '^[0-9]+(\.[0-9]+)?$' \
        || die "ABBA $1 measurement invalid ($2)"
}

log "ownership + lock"
R "pgrep -x ds4-server; exit 0" | grep -q . && die "stray ds4-server on box"
R "mkdir /tmp/ds4_box_lock 2>/dev/null && echo $NONCE > /tmp/ds4_box_lock/nonce" \
    || die "box lock held: $(R cat /tmp/ds4_box_lock/nonce 2>/dev/null || echo unknown)"

log "ABBA decode re-stamp (T C C T, per-request timings, plain decode; test=$TEST_TREE ctrl=$CTRL_TREE)"
T1=$(abba_ms_tok "$TEST_TREE");  abba_num T1 "$T1"; log "  T1=$T1 tok/s"
C1=$(abba_ms_tok "$CTRL_TREE");  abba_num C1 "$C1"; log "  C1=$C1 tok/s"
C2=$(abba_ms_tok "$CTRL_TREE");  abba_num C2 "$C2"; log "  C2=$C2 tok/s"
T2=$(abba_ms_tok "$TEST_TREE");  abba_num T2 "$T2"; log "  T2=$T2 tok/s"
kill_server
ABBA=$(python3 -c "
t=($T1+$T2)/2; c=($C1+$C2)/2
d=abs(t-c)/c*100
print('test=%.1f ctrl=%.1f delta=%.1f%% -> %s' % (t, c, d, 'OK' if d < 5.0 else 'FAIL'))")
log "ABBA: $ABBA"
echo "$ABBA" | grep -q 'OK$' || die "ABBA decode delta >= 5%"
unlock
log "ABBA RESTAMP: PASS"
