#!/bin/bash
# memory/drift_attrib.sh — MG-3c: attribute the within-lifetime MemAvailable
# decline to the engine's graph-pool hoard (gap 2a) vs the kernel
# estimate (gap 2b), AND live-gate MG-1 + MG-2 in one run.
#
# REQUIRES the MG-1/MG-2 build on the box (run AFTER the combined
# #32+MG build; before commits is fine -- rsync ships the dirty tree).
# ~25-35 min (two boots + two churn loops).
#
# A/B design (improves on the receipt's single-leg sketch: with MG-1
# active the churn itself trims at every session replacement, so the
# hoard never accumulates -- the OFF leg is needed to SEE it):
#   leg OFF: boot DS4_MEM_OWN_TRIM=0 -> settle -> avail0 -> churn N
#            (alternating serial sizes force session free/create +
#            right-size churn; small cont waves in between) -> avail1.
#            decline_off = avail0 - avail1.
#   leg ON:  same, default env.  decline_on likewise; also asserts
#            ds4_mem_own_trim_calls_total >= 1 (the WAIT_MEM ladder
#            fires on every session replacement), MG-2 gauges render,
#            all requests 200, zero crash lines.
#   hoard share ~= decline_off - decline_on; recovered bytes counter +
#   disclosure lines corroborate.  recovered ~= decline => gap 2a
#   CLOSED by MG-1; recovered ~= 0 => remainder is gap 2b (stays D6).
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HOST=${HOST:-${DS4_BOX33:?}}
TEST_TREE=${TEST_TREE:-${DS4_BOX_TREE:?}}
CTX=${CTX:-131072}
N=${N:-30}
TS=$(date +%s); NONCE="memdrift-$TS-$$"
R(){ ssh -o ConnectTimeout=15 "$HOST" "$@"; }
log(){ echo "[$(date +%H:%M:%S)] memdrift: $*"; }
unlock(){ R "rm -rf /tmp/ds4_box_lock" 2>/dev/null; }
die(){ log "FAIL: $*"; R 'p=$(pgrep -x ds4-server); [ -n "$p" ] && kill $p; exit 0' 2>/dev/null; unlock; exit 1; }

log "ownership + lock"
R "pgrep -x ds4-server; exit 0" | grep -q . && die "stray ds4-server on box"
R "mkdir /tmp/ds4_box_lock 2>/dev/null && echo $NONCE > /tmp/ds4_box_lock/nonce" \
    || die "box lock held: $(R cat /tmp/ds4_box_lock/nonce 2>/dev/null || echo unknown)"

avail_mb(){ R "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo"; }
settle(){
    local a=0
    for i in $(seq 60); do
        a=$(avail_mb)
        [ "${a:-0}" -ge 92160 ] && { log "settle: $((a/1024))G"; return 0; }
        sleep 5
    done
    log "WARN: settled low ($((a/1024))G); aged box -- proceeding"
}
metric(){ # $1 metric-name -> value or 0
    R "curl -s -m 10 localhost:8000/metrics" | awk -v m="$1" '$1 == m {print $2; f=1} END{if(!f) print 0}'
}

# Serial ticket bodies: alternating prompt sizes force the session-
# replacement path (free -> WAIT_MEM ladder -> create) every iteration.
# Tools array = serial lane ticket; greedy, tiny outputs.
gen_bodies(){
    R "python3 - <<'EOF'
import random, json
random.seed(4242)
words=('ledger','bank','union','credit','floor','tenant','margin','epoch')
for tag, n in (('small', 1800), ('large', 7200)):
    txt='drift '+ ' '.join(random.choice(words) for _ in range(n))+'\nOne word summary.'
    json.dump({'model':'ds4','max_tokens':8,
               'tools':[{'name':'noop','description':'probe',
                         'input_schema':{'type':'object','properties':{}}}],
               'messages':[{'role':'user','content':txt}]},
              open(f'/tmp/drift_{tag}.json','w'))
wave={'model':'ds4','max_tokens':8,'messages':[{'role':'user','content':'ping'}]}
json.dump(wave, open('/tmp/drift_wave.json','w'))
EOF"
}

boot(){ # $1 srvlog  $2 env-prefix ('' or 'DS4_MEM_OWN_TRIM=0')
    settle
    R ": > $1; cd $TEST_TREE; $2 setsid nohup ./ds4-server -c $CTX --port 8000 > $1 2>&1 < /dev/null & exit 0"
    local n=0
    until R "grep -q 'listening on http' $1 2>/dev/null; exit \$?" 2>/dev/null; do
        R "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null \
            || die "boot died: $(R "tail -3 $1" | tr '\n' ' ')"
        sleep 5; n=$((n+5)); [ $n -ge 1200 ] && die "boot timeout"
    done
    log "boot OK ($2)"
}
kill_server(){
    R 'p=$(pgrep -x ds4-server); [ -n "$p" ] && kill $p; for i in $(seq 30); do pgrep -x ds4-server >/dev/null || exit 0; sleep 2; done; p=$(pgrep -x ds4-server); [ -n "$p" ] && kill -9 $p; exit 0'
}

churn(){ # $1 tag -> serves errors via globals CH_OK/CH_ERR
    CH_OK=0; CH_ERR=0
    for i in $(seq "$N"); do
        local tag=$([ $((i % 2)) -eq 0 ] && echo small || echo large)
        local code
        code=$(R "curl -s -m 300 -o /dev/null -w '%{http_code}' localhost:8000/v1/messages \
            -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
            --data-binary @/tmp/drift_$tag.json")
        [ "$code" = "200" ] && CH_OK=$((CH_OK+1)) || CH_ERR=$((CH_ERR+1))
        if [ $((i % 5)) -eq 0 ]; then
            for w in 1 2 3; do
                R "curl -s -m 120 -o /dev/null -w '' localhost:8000/v1/messages \
                    -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
                    --data-binary @/tmp/drift_wave.json" &
            done
            wait
            log "$1 churn $i/$N ok=$CH_OK err=$CH_ERR avail=$(avail_mb)M"
        fi
    done
}

run_leg(){ # $1 tag  $2 env-prefix  $3 srvlog -> prints RESULT line
    log "=== leg $1 ==="
    boot "$3" "$2"
    sleep 20
    local a0 a1 trims rec crash
    a0=$(avail_mb)
    churn "$1"
    sleep 20
    a1=$(avail_mb)
    trims=$(metric ds4_mem_own_trim_calls_total)
    rec=$(metric ds4_mem_own_trim_recovered_bytes_total)
    # MG-2 gate assert: the raw observation pair renders (env-independent)
    local pair
    pair=$(R "curl -s -m 10 localhost:8000/metrics" | grep -c 'observation_bytes{kind="meminfo_available"}')
    [ "${pair:-0}" -ge 1 ] || die "leg $1: MG-2 raw pair absent from /metrics"
    crash=$(R "grep -cE 'illegal|continuous batch failed' $3 || true")
    log "RESULT leg=$1 avail0=${a0}M avail1=${a1}M decline=$((a0-a1))M ok=$CH_OK err=$CH_ERR trims=$trims recovered_b=$rec crash=$crash"
    R "grep -m4 'mem own-trim' $3 || true"
    [ "$CH_ERR" = "0" ] || die "leg $1: $CH_ERR non-200 serves"
    [ "$crash" = "0" ] || die "leg $1: crash lines"
    eval "D_$1=$((a0-a1)); T_$1=$trims; REC_$1=$rec"
    kill_server
}

gen_bodies
run_leg off "DS4_MEM_OWN_TRIM=0" /tmp/drift_off_srv.log
run_leg on  ""                   /tmp/drift_on_srv.log
# MG-1 gate asserts (ON leg only; OFF leg trims=0 is the kill-switch check)
[ "${T_off:-1}" = "0" ] || die "kill switch leaked: OFF leg ticked $T_off trims"
[ "${T_on:-0}" -ge 1 ] || die "MG-1 never fired on the ON leg (session churn must hit serial_wait_mem)"
log "ADJUDICATION: hoard_share ~= $((D_off - D_on))M (decline_off=$((D_off))M decline_on=$((D_on))M); recovered=$((REC_on / 1048576))MiB over $T_on trims"
log "recovered ~= decline_off => gap-2a CLOSED by MG-1; ~=0 => remainder is gap-2b (stays D6)"
unlock
log "MEMDRIFT ATTRIB: ALL LEGS DONE"
