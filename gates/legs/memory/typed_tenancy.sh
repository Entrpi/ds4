#!/bin/bash
# memory/typed_tenancy.sh — memgov D3 stage close (typed residency tenancy).
#
# The REPEATABLE close: proves the typed policy surface end to end on
# the current tip.  Legs:
#   (0) d33_getenv_gate — zero getenv in the weight-resolve call graph
#   (1) rsync (dirty-tree guarded) + clean cuda-spark build + warn
#       SIGNATURE-SET parity vs /tmp/abba_rebase_build.log + sm_121a
#   (2) CUDA unit suite on the box
#   (3) goldens x3 policy shapes (eager default / NO_HBM lazy alias /
#       explicit mapped), BIT-EXACT + materialize-shape + residency line
#   (4) conflict probe: typed knob beside a legacy lever = rc 1
#   (5) SERVER explicit-mapped leg: mapped-policy engaged, real tokens,
#       ZERO device weight_arena/span through serving (rent-gate assert),
#       and the D3-4 tripwires SILENT (mapped census violation / mapped
#       policy bypassed / POST-FREEZE all absent)
#   (6) SERVER eager leg (release default): full funding, failed 0,
#       real tokens, tripwires silent
# Heavy perf legs (ABBA, deep, N=8) are D3-1's rent gate + the standing
# release battery — this gate is correctness + engagement only.
#
# Usage: bash gates/legs/memory/typed_tenancy.sh   (repo root, Mac side)
# Env: HOST TEST_TREE PORT (defaults below).  Box lock is the CALLER's
# responsibility (two-session law).
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HOST=${HOST:-${DS4_BOX33:?}}
TEST_TREE=${TEST_TREE:-${DS4_BOX_TREE:?}}
PORT=${PORT:-18033}
GOLDEN_GGUF=${DS4_TEXT_BASE:?}
GOLDEN_VEC=tests/test-vectors/local-golden-0731.vec
REF_BUILD_LOG=/tmp/abba_rebase_build.log     # on the box
BUILD_LOG=/tmp/d3close_build.log             # on the box
SLOG=/tmp/d3close_srv.log                    # on the box
SPID=""

log(){ echo "[$(date +%H:%M:%S)] d3_close: $*"; }
die(){ log "FAIL: $*"; cleanup; exit 1; }
R(){ ssh -o ConnectTimeout=10 "$HOST" "$@"; }

kill_server(){
    [ -n "$SPID" ] && R "kill $SPID" 2>/dev/null
    SPID=""
    R 'p=$(pgrep -x ds4-server); [ -n "$p" ] && kill $p; sleep 2; p=$(pgrep -x ds4-server); [ -n "$p" ] && kill -9 $p; exit 0' 2>/dev/null
}
cleanup(){ kill_server; }
trap cleanup EXIT

settle(){   # stable-start guard
    for i in $(seq 60); do
        local a; a=$(R "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo")
        echo "$a" | grep -qE '^[0-9]+$' && [ "$a" -ge 90000 ] && return 0
        sleep 5
    done
    return 1
}

boot_leg(){ # $1 leg-env  $2 ctx  $3 seats
    settle || die "memory did not settle before boot (stable-start guard)"
    local n=0
    R "cd $TEST_TREE && env $1 \
        DS4_SERVER_COALESCE_MAX=$3 \
        DS4_BATCH_FIT_HEADROOM_MB=8192 DS4_BATCH_VMM_BUDGET_MB=8192 \
        setsid nohup ./ds4-server -c $2 --no-spec --no-mtp --port $PORT \
        > $SLOG 2>&1 < /dev/null & exit 0" &
    local lp=$!; sleep 5; kill $lp 2>/dev/null || true
    until R "grep -q 'listening on http' $SLOG 2>/dev/null; exit \$?" 2>/dev/null; do
        R "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || {
            sleep 3
            R "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
                die "BOOT-DIED: $(R "tail -3 $SLOG" 2>/dev/null | tr '\n' ' ')"
        }
        sleep 3; n=$((n+3)); [ $n -ge 1200 ] && die "boot timeout"
    done
    SPID=$(R "pgrep -x ds4-server | head -1")
    [ -n "$SPID" ] || die "no server pid after listening"
}

assert_tripwires_silent(){ # $1 tag — the D3-4 asserts must NOT fire
    for bad in 'mapped census violation' 'mapped policy bypassed' \
               'POST-FREEZE arena range' 'RANGE-PUBLISH REFUSED'; do
        R "grep -q '$bad' $SLOG" && die "($1) tripwire fired: $bad"
    done
    return 0
}

serve_one(){ # $1 tag — one real completion, nonempty text
    local out
    out=$(R "curl -s -m 900 localhost:$PORT/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Write a 150-word note on tides.\"}],\"max_tokens\":128}'")
    echo "$out" | grep -q finish_reason || die "($1) no finish_reason"
    echo "$out" | python3 -c '
import json,sys
d=json.load(sys.stdin)
t=d["choices"][0]["message"]["content"]
assert t and len(t.split())>=30, "empty/short serve"
print("served %d words" % len(t.split()))' || die "($1) empty serve"
}

assert_mapped_zero_device(){ # $1 tag (rent-gate M-leg assert, reused)
    local v
    v=$(R "curl -s -m 20 localhost:$PORT/metrics" | python3 -c '
import sys,re
tot=0
for l in sys.stdin:
    m=re.match(r"ds4_memory_bytes\{domain=\"unified_device\",class=\"weight_(arena|span)\",state=\"allocated\"\} (\d+)",l)
    if m: tot+=int(m.group(2))
print(tot)')
    echo "$v" | grep -qE '^[0-9]+$' || die "($1) device-span-bytes not numeric ($v)"
    [ "$v" = 0 ] || die "($1) mapped leg allocated $v device span/arena bytes mid-serving"
}

# ---- (0) getenv gate ----------------------------------------------------
log "(0) resolve-graph getenv gate"
bash gates/legs/memory/weight_resolve_noenv.sh || die "getenv gate"

# ---- (1) sync + build + warn parity + arch ------------------------------
git diff --quiet && git diff --cached --quiet || die "dirty tracked tree (the gate measures the committed tip)"
TIP=$(git rev-parse --short HEAD)
log "(1) rsync + clean build at $TIP"
rsync -a --files-from=<(git ls-files) . "$HOST:$TEST_TREE/" || die "rsync"
R "cd $TEST_TREE && make clean >/dev/null 2>&1; make -j6 cuda-spark > $BUILD_LOG 2>&1 && make ds4_test >> $BUILD_LOG 2>&1" \
    || die "build: $(R "tail -3 $BUILD_LOG" | tr '\n' ' ')"
# v0.6.0 quiet-build law: the tip builds with ZERO warnings (the old
# signature-set parity vs the 14-warning control baseline is retired --
# any warning at all is now the failure).
R "! grep -E ': warning|warning #' $BUILD_LOG | grep -q ." \
    || { R "grep -E ': warning|warning #' $BUILD_LOG | head -5"; die "quiet-build law: warnings in tip build"; }
for b in ds4-server ds4_test; do
    R "/usr/local/cuda/bin/cuobjdump $TEST_TREE/$b 2>/dev/null | grep -q 'arch = sm_121a'" || die "$b not sm_121a"
done
log "(1) build clean, warn parity, sm_121a"

# ---- (2) units ----------------------------------------------------------
log "(2) CUDA unit suite"
R "cd $TEST_TREE && ./ds4_test --server 2>&1 | tail -1" | grep -q 'ds4 tests: ok' || die "units"

# ---- (3) goldens x3 policy shapes --------------------------------------
golden_shape(){ # $1 name  $2 env  $3 mat-regex  $4 residency-regex
    local glog=/tmp/d3close_golden_$1.log rc line
    rc=$(R "cd $TEST_TREE && env DS4_TEST_MODEL=$GOLDEN_GGUF DS4_TEST_LOCAL_GOLDEN_FILE=$GOLDEN_VEC $2 ./ds4_test --local-golden-vectors > $glog 2>&1; echo \$?")
    [ "$rc" = 0 ] || die "(golden:$1) rc=$rc: $(R "tail -3 $glog" | tr '\n' ' ')"
    line=$(R "grep 'ds4-test: local golden' $glog | tail -1")
    echo "$line" | grep -q 'top20_max_abs=0' || die "(golden:$1) not bit-exact: $line"
    R "grep -m1 'materialize base:' $glog" | grep -qE "$3" || die "(golden:$1) materialize shape: $(R "grep -m1 'materialize base:' $glog")"
    R "grep -m1 'weight residency:' $glog" | grep -qE "$4" || die "(golden:$1) residency line: $(R "grep -m1 'weight residency:' $glog")"
    log "(3) golden:$1 BIT-EXACT, shape OK"
}
golden_shape eager  ""                          'funded ([1-9][0-9]*)/\1[^0-9].*failed 0' 'base=eager '
golden_shape lazy   'DS4_CUDA_NO_HBM_CACHE=1'   'lazy-policy [1-9].*failed 0'       'base=lazy .*\(legacy-alias\)'
golden_shape mapped 'DS4_WEIGHT_RESIDENCY=mapped' 'mapped-policy [1-9].*failed 0'   'base=mapped .*\(explicit\)'

# ---- (4) conflict probe -------------------------------------------------
log "(4) conflict probe"
rc=$(R "cd $TEST_TREE && env DS4_TEST_MODEL=$GOLDEN_GGUF DS4_TEST_LOCAL_GOLDEN_FILE=$GOLDEN_VEC \
        DS4_WEIGHT_RESIDENCY=mapped DS4_CUDA_NO_HBM_CACHE=1 ./ds4_test --local-golden-vectors > /tmp/d3close_conflict.log 2>&1; echo \$?")
[ "$rc" = 1 ] || die "(conflict) rc=$rc want 1"
R "grep -q 'residency policy conflict' /tmp/d3close_conflict.log" || die "(conflict) no typed message"

# ---- (5) SERVER explicit-mapped leg ------------------------------------
log "(5) server explicit-mapped leg"
boot_leg 'DS4_WEIGHT_RESIDENCY=mapped' 32768 4
R "grep -m1 'weight residency:' $SLOG" | grep -qE 'base=mapped .*\(explicit\)' || die "(srv-mapped) residency line"
R "grep -m1 'materialize base:' $SLOG" | grep -qE 'funded 0/.*mapped-policy [1-9].*failed 0' || die "(srv-mapped) materialize shape"
serve_one srv-mapped
assert_mapped_zero_device srv-mapped
assert_tripwires_silent srv-mapped
kill_server
log "(5) mapped serving, zero device span/arena, tripwires silent"

# ---- (6) SERVER eager leg (release default) ----------------------------
log "(6) server eager leg"
boot_leg '' 32768 4
R "grep -m1 'weight residency:' $SLOG" | grep -q 'base=eager' || die "(srv-eager) residency line"
R "grep -m1 'materialize base:' $SLOG" | grep -qE 'funded ([1-9][0-9]*)/\1[^0-9].*failed 0' || die "(srv-eager) materialize shape"
serve_one srv-eager
assert_tripwires_silent srv-eager
kill_server
log "(6) eager serving, full funding, tripwires silent"

log "D3 CLOSE GATE: ALL LEGS PASS at $TIP"
