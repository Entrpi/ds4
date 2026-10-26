#!/bin/bash
# memory/serial_reclaim.sh — deepmem lite-4: the serial lane COLLECTS from the
# commons instead of 503ing beside idle cache.
#
# Chartered by ds4-memory-governance-design-2026-08-03.md (RECLAIM section);
# ships with commit 33ac89f.  Mechanism: serial_session_ensure_fit's refusal
# path calls ds4_batch_ctx_trim_free (context-owned, NULL-tolerant victims),
# re-runs the settle poll, refuses only if the graph still cannot fit.
#
# LANE (updated 2026-08-11 for the v0.5.6 API arc): plain Anthropic chat
# now rides the CONTINUOUS lane (route_decide; the old API-flavor rule
# died with Inc 3).  What still routes /v1/messages SERIAL is NON-STREAMING
# TOOL GENERATION -- request_compute_needs sets DS4_NEED_CORRECTIVE_RECOVERY
# (the serial corrective-retry contract, Inc 6 adjudication) -- so the deep
# request below carries a trivial tools array.  NOT prompt length: at
# -c 32768 every prompt that exceeds seq_cap also exceeds the serial
# session, so there is no oversized-but-servable length (first-run lesson).
# Lane oracle: the serial 'prompt start' log marker.
#
# SIZING (measured 08-06 on .33, zero-config -c 32768 boot, MTP+DSpark
# armed -- the fit estimate includes their scaffolding and the probe path
# is hard-quiet loud=false, so the threshold was bracketed BEHAVIORALLY
# with SETTLED hot-hog probes): need_min(28K prompt) threshold T is in
# (6.02, 6.87) GiB -- refused at settled 6.02, served at settled 6.87.
# Hog settle drift is +-0.5 GiB, so the window is sized to be drift-proof
# with 24 saturated banks (~1.9 GiB trimmable, measured 78.75 MiB/bank at
# ~31K-token tenants):
#     free-after = NEED_EST_MB - RECLAIM_MARGIN_MB = 5.5 GiB
#     drift band (5.0, 6.0)  -> always BELOW T  -> fit fails
#     post-trim  (6.9, 7.9)  -> always ABOVE T  -> re-poll serves
# For the fit probe to FAIL the sign is:
#     hog = MemAvailable - NEED_EST_MB + RECLAIM_MARGIN_MB
# start_hog WAITS FOR SETTLE (two stable reads) before returning -- the
# box has 16 GiB swap and MemAvailable keeps moving for ~30 s after the
# pin; firing unsettled makes the verdict a coin flip (runs 3-5).
# Re-bracket with two settled probes on a measurement boot if the config
# changes (the 503 refusal line carries need_min in TOKENS only).
#
# Legs:
#   A (reclaim): Anthropic request under hog pressure SERVES (200) with
#      engagement proven: +1 'prompt start' (lane), +1 'serial fit reclaim',
#      >=1 line carrying 'for the serial lane (reclaim)'.
#   C (recovery): a cont chat request then admits clean on trimmed banks.
#   B (control): DS4_BATCH_VMM_TRIM=0, same shape -> the old 503 +
#      'no graph fits', ZERO reclaim lines.  Causality.
#   D (deep-pin, memgov D4-2): DS4_SERVER_PIN_MIN_TOKENS=25000 makes every
#      tenant DEEP -- the server ranking hard-excludes all of them, the
#      request refuses 503 with ZERO reclaim lines, and a post-hog replay
#      of tenant 1 still WARM-HITS (ds4_admits_total{kind="warm"} moves):
#      the D4 gate's "deep-pinned rows remain intact", proven by serving.
#
# memgov D0a-4 injection mode: INJECT=unmap:N (or release:N) arms
# DS4_CUDA_TRIM_INJECT on leg A's boot only.  Leg A then ADDITIONALLY
# asserts the armed line and exactly N forced-failure lines on top of the
# unchanged serve-via-reclaim verdicts -- the reclaim path surviving a
# partial trim (the burst is a few pages against ~1.9 GiB trimmable, far
# inside the drift-proof threshold margins above).  Leg B stays
# injection-free and asserts ZERO inject lines (hygiene).
#
# Remote-wait discipline: every detached remote command goes through
# fire_remote() (local-background ssh + grace kill -- the bare
# `ssh '... & exit 0'` pattern hangs the local ssh nondeterministically).
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HOST=${HOST:-${DS4_BOX33:?}}
PORT=${PORT:-8000}
NEED_EST_MB=${NEED_EST_MB:-5900}
RECLAIM_MARGIN_MB=${RECLAIM_MARGIN_MB:-400}
TENANT_TOKENS=${TENANT_TOKENS:-31000}
PROMPT_TOKENS=${PROMPT_TOKENS:-28000}
INJECT=${INJECT:-}                 # D0a-4: unmap:N | release:N on leg A
INJECT_N=${INJECT##*:}
CTX=32768
BANKS=24
TS=$(date +%s)
LOCK="/tmp/ds4_box_lock"
NONCE="reclaim_${TS}_$$"
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "FAIL: $*"; cleanup; exit 1; }

remote() { ssh -o ConnectTimeout=10 "$HOST" "$@"; }

# Detached remote command that cannot wedge the driver: fire the ssh in
# LOCAL background, give it a grace window, then kill the local ssh (the
# remote setsid child survives -- selfload law).
fire_remote() { # $1 remote command string
    ssh -o ConnectTimeout=10 "$HOST" "$1" > /dev/null 2>&1 &
    local pid=$!
    for i in $(seq 12); do kill -0 $pid 2>/dev/null || return 0; sleep 1; done
    kill $pid 2>/dev/null
    return 0
}

cleanup() {
    remote 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill' 2>/dev/null
    remote '[ -f /tmp/reclaim_hog.pid ] && kill $(cat /tmp/reclaim_hog.pid) 2>/dev/null; rm -f /tmp/reclaim_hog.pid' 2>/dev/null
    remote "rm -rf $LOCK" 2>/dev/null
    log "cleanup: server + hog killed on $HOST, lock removed"
}

# ---- box ownership (two-session law) ----
remote "mkdir $LOCK 2>/dev/null && echo $NONCE > $LOCK/nonce" \
    || die "box lock held ($(remote cat $LOCK/nonce 2>/dev/null || echo unknown))"
remote 'ps -eo comm | grep -q "^ds4-server" && exit 1 || exit 0' \
    || die "a ds4-server is already running on $HOST"
trap cleanup EXIT

gen_prompt() { # $1 tokens  $2 seed  $3 outfile(remote)  [$4 chat-json outfile]
    # Also pre-writes the full chat JSON payload when $4 is given: detached
    # curls must post @file -- a process substitution @<(python3 ...) dies
    # with the remote bash session at teardown and the half-fed curl aborts
    # mid-prefill (run-11 lesson: tenants vanish with no .out and the
    # server logs client-gone aborts).
    remote "python3 - <<EOF
import random, json
random.seed($2)
words=('substrate','ledger','bank','union','credit','floor','promote',
       'tenant','collect','margin','honest','verdict','epoch','span')
txt='seed$2 '+' '.join(random.choice(words) for _ in range(int($1)))
txt+='\nSummarize the theme in one word.'
open('$3','w').write(txt)
if '${4:-}':
    json.dump({'messages':[{'role':'user','content':txt}],'max_tokens':8},
              open('${4:-}','w'))
EOF"
}

boot() { # $1 extra_env  $2 log
    fire_remote "cd ${DS4_BOX_TREE:?} && setsid nohup env DS4_SERVER_COALESCE_MAX=$BANKS $1 ./ds4-server -c $CTX > $2 2>&1 < /dev/null & exit 0"
    for i in $(seq 60); do
        remote "grep -q 'persistent batch ctx ready' $2" && return 0
        remote 'ps -eo comm | grep -q "^ds4-server"' || { remote "tail -3 $2"; die "boot crashed"; }
        sleep 2
    done
    die "boot deadline ($2)"
}

kill_server() {
    remote 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill'
    for i in $(seq 30); do
        remote 'ps -eo comm | grep -q "^ds4-server"' || return 0
        sleep 2
    done
    log "WARN: server still draining after 60s"
}

start_hog() { # $1 log-for-sizing-message  [$2 free-after-MB override]
    local avail hog target
    # Default free-after = the leg-A bracket (fail-then-serve needs the
    # drift-proof two-sided window).  A leg whose requirement is ONE-SIDED
    # (fail, full stop -- leg D) passes a LOWER override and overshoots:
    # determinism is free when there is no post-trim recovery to protect.
    target=${2:-$((NEED_EST_MB - RECLAIM_MARGIN_MB))}
    avail=$(remote "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo")
    hog=$((avail - target))
    # A churned box can arrive at hog time ALREADY near the free-after
    # target (measured 08-11: avail=6006 after 24 saturated banks; 08-15
    # battery: 5628 then 5528 -- the arrival SAGS across consecutive
    # saturations, the gb10 MemAvailable-decline shape) -- the tenants
    # themselves supply the pressure and only a minimal hot pin is needed
    # to stop kswapd creeping MemAvailable back over the threshold (run-4
    # lesson).  An arrival within 256M ABOVE the target pins the minimum
    # 256M hot floor anyway: the exact-want reclaim (D4-1) re-fits to
    # deficit + a full headroom above the line, so a sub-256M overshoot
    # spends serve margin it can afford, while dying here cost two battery
    # runs.  Only an arrival BELOW the target is genuinely unreachable.
    if [ "$hog" -lt 256 ]; then
        [ "$avail" -ge "$target" ] || die "avail=$avail already below free-after target (target=$target) -- reboot the box or re-bracket"
        hog=256
    fi
    log "MemAvailable=${avail}M -> hog=${hog}M (free-after target ${target}M)"
    # The box carries 16 GiB of swap and kswapd evicts a cold slept-on hog
    # within ~1 min (MemAvailable creeps back over the threshold and the
    # fit probe passes -- run-4 lesson).  Keep the pages HOT: continuous
    # re-touch loop, ~1-2 s/pass at 4K stride, so swap-out never sticks.
    fire_remote "setsid nohup python3 -c \"
import time,os
b=bytearray($hog*1048576)
for i in range(0,len(b),4096): b[i]=1
open('/tmp/reclaim_hog.pid','w').write(str(os.getpid()))
while True:
    for i in range(0,len(b),4096): b[i]^=1
    time.sleep(0.5)\" > /tmp/reclaim_hog.log 2>&1 < /dev/null & exit 0"
    for i in $(seq 30); do
        remote '[ -f /tmp/reclaim_hog.pid ]' && break
        sleep 2
    done
    remote '[ -f /tmp/reclaim_hog.pid ]' || die "hog failed to pin"
    # settle: fire only once MemAvailable is stable (two reads +-100 MiB)
    for i in $(seq 25); do
        local a b2 d
        a=$(remote "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo"); sleep 5
        b2=$(remote "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo")
        d=$((a - b2))
        [ ${d#-} -lt 100 ] && { log "hog settled at ${b2}M free"; return 0; }
    done
    die "hog never settled"
}

stop_hog() {
    remote '[ -f /tmp/reclaim_hog.pid ] && kill $(cat /tmp/reclaim_hog.pid) 2>/dev/null; rm -f /tmp/reclaim_hog.pid'
}

fire_tenant() { # $1 outfile-prefix  $2 tenant-index
    # THREE rules: fire_remote protects the LOCAL ssh from wedging;
    # setsid nohup detaches the REMOTE curl from session teardown
    # (0b-4 law); the payload is a pre-written FILE because a
    # process substitution dies with the remote bash and the
    # half-fed curl aborts mid-prefill (run-11 lesson).
    fire_remote "setsid nohup curl -s -m 600 localhost:$PORT/v1/chat/completions \
      -H 'Content-Type: application/json' \
      --data-binary @/tmp/reclaim_t$2.json \
      -o $1$2.out > /dev/null 2>&1 < /dev/null & exit 0"
}

saturate() { # $1 outfile-prefix  [$2 mode: strict|terminal, default strict]
    # Waves of 8: prefill aggregation past width 8 hits the small16-tier
    # cliff (knobs-k5 law; measured here as ~350 tok/s at width 24 vs
    # ~2400 at width 8) -- 24-wide saturation runs 3x slower than three
    # 8-wide waves reaching the same final bank state.
    #
    # terminal mode (leg B): with trim DISABLED a churned box exhausts the
    # comp-cache budget mid-saturation (08-11: admission refused at bank
    # 16, the rest 503'd 'no graph fits') -- which IS the control behavior.
    # A refused tenant is TERMINAL; strict counting waits 20 min for
    # completions that already refused and deadlocks the leg.
    local mode=${2:-strict} MARK='finish_reason'
    [ "$mode" = "terminal" ] && MARK='finish_reason|"error"'
    remote "rm -f $1*.out"   # stale outs from a prior run corrupt arrival counting
    local wave_sz=8 fired=0 done_n=0
    while [ "$fired" -lt "$BANKS" ]; do
        local wave_end=$((fired + wave_sz))
        [ "$wave_end" -gt "$BANKS" ] && wave_end=$BANKS
        for b in $(seq $((fired + 1)) $wave_end); do
            fire_tenant "$1" "$b"
        done
        fired=$wave_end
        # Waiter with ONE refire: a detached curl can die client-side
        # mid-run (run 13: one vanished ~4 min into decode -- no OOM, no
        # .out, server logged the client-gone abort; ~1/32 firings).  The
        # bank keeps the prompt as a warm record, so a refire warm-hits
        # and completes in seconds.  Refire when the box runs out of live
        # curls early (>=60 s in, launches are long past) or at the old
        # 600 s deadline, whichever comes first.
        local refired=0 live
        for i in $(seq 240); do
            done_n=$(remote "ls $1*.out 2>/dev/null | xargs -r grep -lE '$MARK' | wc -l")
            [ "$done_n" -ge "$fired" ] && break
            if [ "$refired" -eq 0 ] && [ "$i" -ge 12 ]; then
                live=$(remote "ps -eo comm | grep -c '^curl'" || true)
                if [ "${live:-1}" -eq 0 ] || [ "$i" -eq 120 ]; then
                    refired=1
                    for b in $(seq 1 $fired); do
                        remote "grep -qE '$MARK' $1$b.out 2>/dev/null" && continue
                        log "refire tenant $b ($done_n/$fired terminal, live_curls=${live:-?})"
                        fire_tenant "$1" "$b"
                    done
                fi
            fi
            sleep 5
        done
        [ "$done_n" -ge "$fired" ] || die "wave stalled: $done_n/$fired tenants terminal (mode=$mode, refires=$refired)"
        log "wave done ($done_n/$BANKS)"
    done
    if [ "$mode" = "terminal" ]; then
        local comp
        comp=$(remote "ls $1*.out 2>/dev/null | xargs -r grep -l finish_reason | wc -l")
        [ "$comp" -ge "$wave_sz" ] || die "terminal saturation: only $comp completions (banks would be empty; pressure not real)"
        log "tenants terminal ($done_n/$BANKS; $comp completed)"
    else
        log "tenants done ($done_n/$BANKS)"
    fi
}

anthropic_deep() { # $1 outfile -> echoes http code
    # The tools array is the SERIAL LANE TICKET (see LANE note): buffered
    # anthropic tool generation keeps the corrective-retry contract and
    # routes serial; without it the cont lane admits the prompt into a
    # bank and the serial fit (the machinery under test) never runs.
    remote "curl -s -m 600 -o $1 -w '%{http_code}' localhost:$PORT/v1/messages \
      -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
      --data-binary @<(python3 -c \"import json;print(json.dumps({'model':'ds4','max_tokens':16,'tools':[{'name':'noop','description':'no-op probe tool','input_schema':{'type':'object','properties':{}}}],'messages':[{'role':'user','content':open('/tmp/reclaim_deep.txt').read()}]}))\")"
}

count() { remote "grep -c \"$1\" $2 || true"; }

for b in $(seq 1 $BANKS); do
    gen_prompt $TENANT_TOKENS $((TS+b)) /tmp/reclaim_t$b.txt /tmp/reclaim_t$b.json
done
gen_prompt $PROMPT_TOKENS 999 /tmp/reclaim_deep.txt

# ---- stable-start guard (memgov D2-1 amendment; plan sec 13: "a failure
# to reach a stable starting memory state must abort the experiment").
# Booting into a predecessor's un-reclaimed teardown (driver-lag WAIT_MEM
# class) silently shrinks the fit -- observed 2026-08-13: boot 16s after a
# shadow-gate teardown got free=8.61 GiB -> max_seq 2 (requested 24) and
# the whole saturate/hog/serve choreography ran off its design point.
# The fit needs headroom (8 GiB) + BANKS x ~0.5 GiB, plus the wave's page
# budget; require an idle-shaped box before EVERY boot and DIE if it
# never settles (legs A and B both boot fresh servers).
SETTLE_MIN_MB=${SETTLE_MIN_MB:-90000}
settle_wait() {
    local avail=0
    for i in $(seq 60); do
        avail=$(( $(remote "awk '/MemAvailable/{print \$2}' /proc/meminfo") / 1024 ))
        [ "$avail" -ge "$SETTLE_MIN_MB" ] && { \
            log "stable start: MemAvailable=${avail}M >= ${SETTLE_MIN_MB}M"; return 0; }
        sleep 5
    done
    die "starting memory never settled (MemAvailable ${avail}M < ${SETTLE_MIN_MB}M after 300s) -- aborting per plan sec 13, not continuing degraded"
}
settle_wait

# ---- LEG A ----
log "boot A (stock, $BANKS banks pinned${INJECT:+, trim-inject $INJECT})"
boot "${INJECT:+DS4_CUDA_TRIM_INJECT=$INJECT}" /tmp/reclaim_A.log
log "saturating $BANKS banks with ${TENANT_TOKENS}-token tenants"
saturate /tmp/reclaim_ta
start_hog A
log "firing Anthropic serial request (${PROMPT_TOKENS} tokens; lane oracle = 'prompt start')"
CODE=$(anthropic_deep /tmp/reclaim_deep.out)
PSTART=$(count "prompt start" /tmp/reclaim_A.log)
RECL=$(count "serial fit reclaim" /tmp/reclaim_A.log)
TRIM=$(count "for the serial lane (reclaim)" /tmp/reclaim_A.log)
log "leg A: http=$CODE prompt_start=$PSTART reclaim_lines=$RECL trim_lines=$TRIM"
# Assert ORDER matters (run-5 lesson): reclaim engagement first.  'prompt
# start' prints only inside generate_job AFTER a successful fit, so it is
# the SUCCESS-side lane oracle, not a routing check -- a reclaim line
# already proves the serial path ran ensure_fit.
[ "$RECL" -ge 1 ] || die "leg A: fit never failed (no reclaim line) -- lower NEED_EST_MB / raise the hog (lane proof: a cont-served request leaves reclaim impossible AND prompt_start=0)"
[ "$TRIM" -ge 1 ] || die "leg A: reclaim line without trim receipt"
[ "$CODE" = "200" ] || die "leg A: reclaim fired (trim receipt in the A log shows released MiB) but the re-poll still missed -- post-trim free must exceed the need_min threshold; raise TENANT_TOKENS (more trimmable) or NEED_EST_MB (higher free-after)"
[ "$PSTART" -ge 1 ] || die "leg A: served without the serial success marker (unexpected lane)"
log "A PASS (served via reclaim: prompt_start=$PSTART reclaim=$RECL trim=$TRIM)"
# memgov D2-4: under the SERIAL enforce default this serve is
# quote-ADMITTED (margin calibration: the quote evaluates the fit
# probe's own inequality).  ANY DISAGREE on this boot means quote and
# probe drifted -- churn(24 tenants) + trim + serve is exactly the
# composition this asserts (the D2-1 row-end refresh feeding the
# quote's cross-lane terms).  Pre-margin evidence: the 08-14 re-stamp
# logged ONE serial_rightsize SHADOW_STRICTER here (deficit 2.37 GiB,
# the operator-floor term); the margin calibration retired it.
DIS_A=$(remote "grep -c 'memgov shadow DISAGREE' /tmp/reclaim_A.log" || true)
[ "${DIS_A:-0}" = "0" ] || die "leg A: memgov DISAGREE on the reclaim serve ($DIS_A)"
if [ -n "$INJECT" ]; then
    ARMED=$(count "trim inject: armed" /tmp/reclaim_A.log)
    FIRED=$(count "trim inject: forced" /tmp/reclaim_A.log)
    [ "$ARMED" = "1" ] || die "leg A: injection requested but not armed (armed=$ARMED; bad spec?)"
    [ "$FIRED" = "$INJECT_N" ] || die "leg A: inject fired $FIRED != $INJECT_N (burst must exhaust inside the reclaim trim)"
    log "A INJECT PASS (armed=1 fired=$FIRED/$INJECT_N; reclaim survived partial trim)"
fi

# ---- LEG C ----
gen_prompt 200 777 /tmp/reclaim_c.txt
CODE=$(remote "curl -s -m 120 -o /tmp/reclaim_c.out -w '%{http_code}' localhost:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @<(python3 -c \"import json;print(json.dumps({'messages':[{'role':'user','content':open('/tmp/reclaim_c.txt').read()}],'max_tokens':8}))\")")
[ "$CODE" = "200" ] || die "leg C: post-reclaim cont request failed (http $CODE)"
ILL=$(count "illegal\|continuous batch failed" /tmp/reclaim_A.log)
[ "$ILL" = "0" ] || die "leg C: crash lines present ($ILL)"
log "C PASS (cont admits clean on trimmed banks)"
stop_hog
kill_server

# ---- LEG B (control) ----
settle_wait   # leg A's teardown must reclaim before the control boots
log "boot B (DS4_BATCH_VMM_TRIM=0 control)"
boot "DS4_BATCH_VMM_TRIM=0" /tmp/reclaim_B.log
saturate /tmp/reclaim_tb terminal
start_hog B
CODE=$(anthropic_deep /tmp/reclaim_bdeep.out)
NOFIT=$(count "no graph fits" /tmp/reclaim_B.log)
RECL=$(count "serial fit reclaim" /tmp/reclaim_B.log)
log "leg B: http=$CODE nofit=$NOFIT reclaim_lines=$RECL"
[ "$CODE" = "503" ] || die "leg B control: expected 503, got $CODE"
[ "$NOFIT" -ge 1 ] || die "leg B control: no 'no graph fits' line"
[ "$RECL" = "0" ] || die "leg B control: reclaim fired with trim disabled"
INJ_B=$(count "trim inject" /tmp/reclaim_B.log)
[ "$INJ_B" = "0" ] || die "leg B control: inject lines leaked into an injection-free boot ($INJ_B)"
# memgov D5-3: the typed rejection family under a REAL refusal.  The
# right-size refusal site ticks serial/live_headroom at the SAME point
# that prints the 'no graph fits' line (probe refusals; an enforce
# divergence would tick without the line, which this physical-hog leg
# never takes), so typed == NOFIT on this boot.  Run-2 lesson: the
# legacy graph_fit_refusals metric does NOT tick on the D4 quote path
# (its loud engine gate never runs here) -- the log line is the
# apples-to-apples twin, not that metric.
TYP_B=$(remote "curl -s -m 20 localhost:$PORT/metrics | grep 'ds4_requests_rejected_total{lane=\"serial\",reason=\"live_headroom\"}' | awk '{print \$2}'")
[ -n "$TYP_B" ] && [ "$TYP_B" -ge 1 ] || die "leg B: typed serial/live_headroom cell missing or zero after a refusal (got '$TYP_B')"
[ "$TYP_B" = "$NOFIT" ] || die "leg B: typed serial/live_headroom ($TYP_B) != 'no graph fits' lines ($NOFIT) -- tick site diverged from the refusal line"
log "B PASS (control reproduces the old 503, zero reclaim lines, typed=nofit=$TYP_B)"
stop_hog
kill_server

# ---- LEG D (deep-pin protection; memgov D4-2) ----
# Injection is leg A's concern (armed on its boot only) and leg B owns the
# hygiene assert; rerunning D under INJECT would duplicate a 13-minute
# saturation that deepens the arrival sag above for zero new evidence.
# The BASE run owns leg D.
if [ -n "$INJECT" ]; then
    log "leg D skipped (INJECT mode; the BASE run owns deep-pin protection)"
    log "SERIAL-RECLAIM GATE: ALL LEGS PASS"
    exit 0
fi
settle_wait
log "boot D (DS4_SERVER_PIN_MIN_TOKENS=25000: every ${TENANT_TOKENS}-token tenant is DEEP)"
boot "DS4_SERVER_PIN_MIN_TOKENS=25000" /tmp/reclaim_D.log
saturate /tmp/reclaim_td
# One-sided leg: only FIT-FAIL must hold, so overshoot the pin to a
# free-after a full GiB below the bracketed threshold's low edge
# ((6.02, 6.87) GiB) -- deterministic refusal under +-0.5 GiB drift, and
# a wrongful reclaim still fails loudly on the RECL==0 causality assert.
# Also rescues the churned-box arrival band (5.6-6.3 GiB) that starved
# the leg-A-calibrated pin floor on the 08-15 first run.
start_hog D 4500
CODE=$(anthropic_deep /tmp/reclaim_ddeep.out)
RECL=$(count "serial fit reclaim" /tmp/reclaim_D.log)
NOFIT=$(count "no graph fits" /tmp/reclaim_D.log)
log "leg D: http=$CODE nofit=$NOFIT reclaim_lines=$RECL"
[ "$CODE" = "503" ] || die "leg D: expected 503 (deep-pinned commons must not be spent), got $CODE"
[ "$NOFIT" -ge 1 ] || die "leg D: refused without the 'no graph fits' line"
[ "$RECL" = "0" ] || die "leg D: reclaim fired against deep-pinned banks ($RECL lines)"
stop_hog
# The promise is INTACT, proven by serving: a replay of tenant 1 must
# reuse its untouched deep record's cached prefix (metrics, not log grep
# -- cached prefill tokens are the admission path's own truth, and they
# tick on the warm, fork, and partial-fork serve shapes alike; a trimmed
# bank would cold-prefill and leave the counter flat).
W0=$(remote "curl -s -m 20 localhost:$PORT/metrics | grep 'prefilled_total{kind=\"cached\"}' | awk '{print \$2}'")
CODE=$(remote "curl -s -m 300 -o /tmp/reclaim_dreplay.out -w '%{http_code}' localhost:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/reclaim_t1.json")
[ "$CODE" = "200" ] || die "leg D: post-hog replay failed (http $CODE)"
remote "grep -q finish_reason /tmp/reclaim_dreplay.out" || die "leg D: replay has no finish_reason"
W1=$(remote "curl -s -m 20 localhost:$PORT/metrics | grep 'prefilled_total{kind=\"cached\"}' | awk '{print \$2}'")
[ "${W1:-0}" -gt "${W0:-0}" ] || die "leg D: replay cold-prefilled (cached $W0 -> $W1; deep record lost)"
log "D PASS (deep-pinned commons intact: refused without spending, replay reused $((W1 - W0)) cached tokens)"

log "SERIAL-RECLAIM GATE: ALL LEGS PASS"
