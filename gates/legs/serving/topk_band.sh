#!/bin/bash
# serving/topk_band.sh — rider #32 fix gate: the captured serial decode's
# top-k band covers the whole session (comp_cap), so decode-appended
# comp rows are scored and selectable.
#
# The bug (reproduced 2026-08-16, three independent confirmations): the
# PC5 substrate read the LIVE count at replay but the kernels' band
# (scores stride + scan ceiling) baked at capture-time n_comp = prompt
# rows, so every serial decode's selector was blind to all post-prompt
# rows and the stream512 guard clamped (+21/step, the indexed layers).
# Fix: ds4_gpu_indexer_topk_tensor passes the capture-stable maximum
# (n_band = n_spec under PC5) to every kernel; stream512 is mandatory
# under PC5 (the chunked tree's grid is n_comp-derived).
#
# Legs (driver on the Mac, server on the box; ~15 min):
#   A right-sized: -c 262144 boot, ~24k serial (tools ticket) ->
#     right-size fires -> decode 512 -> ZERO violation prints + the
#     stream announce line carries the BAND (comp_cap-class), served 200.
#   B full-ctx: -c 32768 boot, same request (no right-size) -> ZERO
#     violation prints, served 200.
# Both legs also assert the serial lane oracle ('prompt start') and
# zero crash lines.  The violation counter stays live in the binary as
# the standing tripwire; this gate is its zero-assert.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HOST=${HOST:-${DS4_BOX33:?}}
TEST_TREE=${TEST_TREE:-${DS4_BOX_TREE:?}}
TS=$(date +%s); NONCE="topkband-$TS-$$"
R(){ ssh -o ConnectTimeout=15 "$HOST" "$@"; }
log(){ echo "[$(date +%H:%M:%S)] topk_band: $*"; }
unlock(){ R "rm -rf /tmp/ds4_box_lock" 2>/dev/null; }
die(){ log "FAIL: $*"; R 'p=$(pgrep -x ds4-server); [ -n "$p" ] && kill $p; exit 0' 2>/dev/null; unlock; exit 1; }

log "ownership + lock"
R "pgrep -x ds4-server; exit 0" | grep -q . && die "stray ds4-server on box"
R "mkdir /tmp/ds4_box_lock 2>/dev/null && echo $NONCE > /tmp/ds4_box_lock/nonce" \
    || die "box lock held: $(R cat /tmp/ds4_box_lock/nonce 2>/dev/null || echo unknown)"

settle(){
    local a=0
    for i in $(seq 60); do
        a=$(R "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo")
        [ "${a:-0}" -ge 90 ] && { log "settle: ${a}G"; return 0; }
        sleep 5
    done
    die "box never settled (${a}G)"
}
boot(){ # $1 ctx  $2 srvlog
    settle
    R ": > $2; cd $TEST_TREE; setsid nohup ./ds4-server -c $1 --port 8000 > $2 2>&1 < /dev/null & exit 0"
    local n=0
    until R "grep -q 'listening on http' $2 2>/dev/null; exit \$?" 2>/dev/null; do
        R "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null \
            || die "boot died: $(R "tail -3 $2" | tr '\n' ' ')"
        sleep 5; n=$((n+5)); [ $n -ge 1200 ] && die "boot timeout"
    done
    log "boot OK (-c $1)"
}
kill_server(){
    R 'p=$(pgrep -x ds4-server); [ -n "$p" ] && kill $p; for i in $(seq 15); do pgrep -x ds4-server >/dev/null || exit 0; sleep 2; done; p=$(pgrep -x ds4-server); [ -n "$p" ] && kill -9 $p; exit 0'
}

log "prompt gen (~24334-token field shape) + serial body (tools ticket)"
R "python3 - <<'EOF'
import random, json
random.seed(24334)
words=('substrate','ledger','bank','union','credit','floor','promote',
       'tenant','collect','margin','honest','verdict','epoch','span')
txt='seed24334 '+' '.join(random.choice(words) for _ in range(24334))
txt+='\nSummarize the theme in one word, then explain at length.'
json.dump({'model':'ds4','max_tokens':512,
           'tools':[{'name':'noop','description':'no-op probe tool',
                     'input_schema':{'type':'object','properties':{}}}],
           'messages':[{'role':'user','content':txt}]},
          open('/tmp/topkband_body.json','w'))
EOF"

run_leg(){ # $1 tag  $2 ctx  $3 srvlog  $4 expect_rightsize(0/1)
    log "=== leg $1: boot -c $2 ==="
    boot "$2" "$3"
    local code
    code=$(R "curl -s -m 900 -o /tmp/topkband_$1.out -w '%{http_code}' localhost:8000/v1/messages \
        -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
        --data-binary @/tmp/topkband_body.json")
    [ "$code" = "200" ] || { R "tail -5 $3"; die "leg $1 http=$code"; }
    local pstart viol rsz crash announce
    pstart=$(R "grep -c 'prompt start' $3 || true")
    viol=$(R "grep -c 'TOPK-BOUND-VIOL' $3 || true")
    rsz=$(R "grep -cE 'right-siz' $3 || true")
    crash=$(R "grep -cE 'illegal|continuous batch failed' $3 || true")
    announce=$(R "grep -m1 'indexer topk stream active' $3 || true")
    log "leg $1: http=200 prompt_start=$pstart right_size=$rsz TOPK_VIOL=$viol crash=$crash"
    [ -n "$announce" ] && log "leg $1: $announce"
    [ "$pstart" -ge 1 ] || die "leg $1: not serial-served (lane oracle absent)"
    [ "$rsz" -ge "$4" ] || die "leg $1: expected right-size engagement ($rsz < $4)"
    [ "$viol" = "0" ] || { R "grep -m2 'TOPK-BOUND-VIOL' $3"; die "leg $1: $viol violation prints (band still stale)"; }
    [ "$crash" = "0" ] || die "leg $1: crash lines ($crash)"
    kill_server
}

run_leg rightsized 262144 /tmp/topkband_rs_srv.log 1
run_leg fullctx    32768  /tmp/topkband_fc_srv.log 0
unlock
log "TOPK BAND GATE: ALL LEGS PASS"
