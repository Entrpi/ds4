#!/bin/bash
# memory/residual_experiment.sh — task #81: attribute the cross-boot
# MemAvailable residual (~350 MiB per boot/kill cycle, only a reboot
# recovers it) to a kernel/driver BUCKET and a teardown MODE.
#
# RUN POST-SOAK, BEFORE ANY BUILD OR RSYNC touches the box: the aged
# pre-reboot state is the specimen.  Uses the EXISTING test-tree binary
# (46297a1-era) deliberately — the residual is driver-class and must not
# wait on the new build.  ~15-20 min (two boots + settles).
#
# Design (receipt + plan doc local/docs/v06/memgaps_recon_plan_2026-08-16.md):
#   S0 idle-settled snapshot (aged box)
#   boot -> clean SIGTERM kill (full teardown: server_close_resources)
#     -> settle -> S1
#   boot -> kill -9 (no teardown) -> settle -> S2
#   (S1-S0) = clean-exit residual; (S2-S1) = kill-9 residual.
#   Equal residuals  => pure driver/kernel (release-note + upstream);
#   clean < kill9    => teardown ordering matters (engine fix candidate);
#   either way the BUCKET (SUnreclaim / VmallocUsed / SecPageTables /
#   Shmem / PageTables) names the suspect.
# Snapshots capture FULL /proc/meminfo + /proc/vmstat + nvidia-smi -q
# memory section; the report prints headline deltas, adjudication is the
# operator's.  No asserts — this is an experiment, not a gate.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HOST=${HOST:-${DS4_BOX33:?}}
TEST_TREE=${TEST_TREE:-${DS4_BOX_TREE:?}}
CTX=${CTX:-131072}
OUT=/tmp/ds4_memresid
TS=$(date +%s); NONCE="memresid-$TS-$$"
R(){ ssh -o ConnectTimeout=15 "$HOST" "$@"; }
log(){ echo "[$(date +%H:%M:%S)] memresid: $*"; }
unlock(){ R "rm -rf /tmp/ds4_box_lock" 2>/dev/null; }
die(){ log "FAIL: $*"; R 'p=$(pgrep -x ds4-server); [ -n "$p" ] && kill -9 $p; exit 0' 2>/dev/null; unlock; exit 1; }

log "ownership + lock"
R "pgrep -x ds4-server; exit 0" | grep -q . && die "stray ds4-server on box"
R "mkdir /tmp/ds4_box_lock 2>/dev/null && echo $NONCE > /tmp/ds4_box_lock/nonce" \
    || die "box lock held: $(R cat /tmp/ds4_box_lock/nonce 2>/dev/null || echo unknown)"
R "mkdir -p $OUT"

avail_mb(){ R "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo"; }

# Aged box may idle well below the 90G gate law's bracket -- stability,
# not a level, is the settle criterion here (3 samples 20s apart within
# 150 MiB of each other; cap 10 min).
settle_stable(){
    local a b c
    for i in $(seq 30); do
        a=$(avail_mb); sleep 20; b=$(avail_mb); sleep 20; c=$(avail_mb)
        local lo=$a hi=$a
        for v in $b $c; do [ "$v" -lt "$lo" ] && lo=$v; [ "$v" -gt "$hi" ] && hi=$v; done
        if [ $((hi - lo)) -le 150 ]; then log "settled: ${c}M (spread $((hi-lo))M)"; return 0; fi
        log "settling: $a/$b/$c M"
    done
    log "WARN: never fully stable; proceeding with ${c}M"
}

snap(){ # $1 label
    R "{ date; echo '=== meminfo'; cat /proc/meminfo; echo '=== vmstat'; cat /proc/vmstat; \
         echo '=== nvidia'; nvidia-smi -q 2>/dev/null | sed -n '/FB Memory Usage/,/^ *$/p'; } > $OUT/snap_$1.txt"
    log "snapshot $1: MemAvailable $(avail_mb)M"
}

boot(){ # srvlog
    R ": > $1; cd $TEST_TREE; setsid nohup ./ds4-server -c $CTX --port 8000 > $1 2>&1 < /dev/null & exit 0"
    local n=0
    until R "grep -q 'listening on http' $1 2>/dev/null; exit \$?" 2>/dev/null; do
        R "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null \
            || die "boot died: $(R "tail -3 $1" | tr '\n' ' ')"
        sleep 5; n=$((n+5)); [ $n -ge 1200 ] && die "boot timeout"
    done
    log "boot OK"
}

kill_mode(){ # $1 clean|kill9
    if [ "$1" = clean ]; then
        R 'p=$(pgrep -x ds4-server); [ -n "$p" ] && kill $p; exit 0'
        for i in $(seq 60); do
            R "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || { log "clean exit done"; return 0; }
            sleep 2
        done
        die "clean exit never finished (drain hung)"
    else
        R 'p=$(pgrep -x ds4-server); [ -n "$p" ] && kill -9 $p; exit 0'
        sleep 3; log "kill -9 done"
    fi
}

field(){ # $1 snap-label $2 meminfo-field
    R "awk '/^$2:/{print \$2}' $OUT/snap_$1.txt"
}

log "=== S0: idle aged baseline ==="
settle_stable; snap s0
log "=== leg 1: boot -> CLEAN exit ==="
boot /tmp/memresid_srv1.log; sleep 30; kill_mode clean
settle_stable; snap s1
log "=== leg 2: boot -> kill -9 ==="
boot /tmp/memresid_srv2.log; sleep 30; kill_mode kill9
settle_stable; snap s2

log "=== report (deltas in kB; negative MemAvailable delta = residual) ==="
printf '%-16s %12s %12s\n' field 'S1-S0(clean)' 'S2-S1(kill9)'
for f in MemAvailable MemFree Cached Shmem Slab SReclaimable SUnreclaim \
         KernelStack PageTables SecPageTables VmallocUsed Mlocked Unevictable; do
    a0=$(field s0 $f); a1=$(field s1 $f); a2=$(field s2 $f)
    [ -n "$a0" ] && [ -n "$a1" ] && [ -n "$a2" ] && \
        printf '%-16s %12s %12s\n' "$f" "$((a1-a0))" "$((a2-a1))"
done
log "snapshots on box: $OUT/snap_{s0,s1,s2}.txt -- copy to ~/soak_v06/ for durability:"
log "  ssh $HOST 'cp $OUT/snap_*.txt ~/soak_v06/'"
unlock
log "MEMRESID EXPERIMENT DONE (adjudicate deltas above)"
