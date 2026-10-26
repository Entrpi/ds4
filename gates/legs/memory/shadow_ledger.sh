#!/bin/bash
# memory/shadow_ledger.sh — memgov D0b close gate (spec: local/docs/v057/
# d0b_scoping_2026-08-11.md §4).  Runs the D0b-SPECIFIC legs; the stage
# close ALSO requires memory/census.sh (decision oracle + reconciliation
# + no-hot-path + ABBA vs the cebaff7 control) and memory/serial_reclaim.sh
# (INJECT + trim-off control) green on the same tree.
#
# WHAT IT PROVES
#   (t) TORN-SNAPSHOT HUNT: a concurrent admission wave with a /metrics +
#       /v1/stats hammer polling THROUGH the wave -- zero torn fallbacks,
#       zero census faults (includes the seqlock single-writer tripwire),
#       zero governor faults, ENGINE_OTHER still 0, epoch even+advancing.
#   (r) SHADOW-CLAIM RECONCILIATION, two snapshots: at BOOT SETTLE the
#       ledger is exact (ENGINE_BOOT == recomputed non-bank non-prewarm
#       census total; bank lease == bank census, D0b-4 absolute basis);
#       at WAVE SETTLE the bank lease sandwiches the census (resident <=
#       census live <= intent), PREWARM is bounded by its classes, the
#       boot leases never moved, and EVERY census class that grew since
#       boot is on the named growable list (batch_bank, scratch_sticky,
#       kernel_partials) -- zero unexplained gaps, drift is attributed
#       or the gate fails with the class named.
#   (d) DECISION AUDIT: every admission ticked exactly one shadow cell
#       (sum over batch_bank_plan cells == admits + rejects) and NO cell
#       carries a real disagreement (reasons beyond agree all zero; on
#       CUDA the observation is OK so obs_policy must be zero too).
#
# Usage: bash gates/legs/memory/shadow_ledger.sh   (from the repo root)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HOST=${HOST:-${DS4_BOX33:?}}
TEST_TREE=${TEST_TREE:-${DS4_BOX_TREE:?}}   # override for branch-tree verification runs
CTX=${CTX:-32768}
PORT=8000
WAVE=${WAVE:-8}
TS=$(date +%s); NONCE="d0bshadow-$TS-$$"
LOCK=/tmp/ds4_box_lock
R() { ssh -o ConnectTimeout=10 "$HOST" "$@"; }
SPID=""
log() { echo "[$(date +%H:%M:%S)] $*"; }
cleanup() {
    [ -n "$SPID" ] && R "kill $SPID" 2>/dev/null
    R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill' 2>/dev/null
    R "rm -rf $LOCK" 2>/dev/null
    rm -f "${SNAP0:-}" "${SNAP1:-}" 2>/dev/null
}
die() { log "FAIL: $*"; cleanup; exit 1; }

R "ps -eo comm | grep -q '^ds4-server' && echo BUSY || echo IDLE" | grep -q IDLE \
    || die "a ds4-server is already running on $HOST"
R "mkdir $LOCK 2>/dev/null && echo $NONCE > $LOCK/nonce" \
    || die "box lock held: $(R cat $LOCK/nonce 2>/dev/null || echo unknown)"
trap cleanup EXIT
log "lock acquired ($NONCE)"

boot_tree() { # $1 log
    ssh -o ConnectTimeout=10 "$HOST" \
        "cd $TEST_TREE && setsid nohup ./ds4-server -c $CTX > $1 2>&1 < /dev/null & exit 0" &
    local lp=$!; sleep 5; kill $lp 2>/dev/null || true
    for i in $(seq 240); do
        R "grep -q 'listening on http' $1" && { SPID=$(R "pgrep -x ds4-server | head -1"); return 0; }
        R "ps -eo comm | grep -q '^ds4-server'" || { R "tail -5 $1"; return 1; }
        sleep 2
    done
    return 1
}
met() { # $1 metric line prefix (exact $1 match on field 1)
    R "curl -s -m 20 localhost:$PORT/metrics" | awk -v k="$1" '$1==k{print $2}'
}

log "(t) torn-snapshot hunt: boot + ${WAVE}-wide admission wave under poll hammer"
boot_tree /tmp/d0bshadow_boot.log || die "boot failed"
# memgov D2-1: the governance mode board rides every boot; a stock boot
# shows the tree defaults (update this expectation WITH each ratchet
# commit).  D2-5 completed the board: ALL FIVE FAMILIES ENFORCE.
R "grep -q 'memgov modes: boot=enforce prewarm=enforce bank=enforce serial=enforce static=enforce' /tmp/d0bshadow_boot.log" \
    || die "memgov mode board missing or not at tree defaults"
grep -q . /dev/null   # no-op keeps set -u happy on empty locals below
# SNAP0: the boot-settle ledger, BEFORE any request (leg (r) asserts the
# boot leases exactly; a request would legitimately grow sticky scratch).
SNAP0=$(mktemp); SNAP1=$(mktemp)
R "curl -s -m 20 localhost:$PORT/metrics" > "$SNAP0"
[ -s "$SNAP0" ] || die "empty boot-settle /metrics snapshot"
E0=$(awk '$1=="ds4_memory_census_epoch"{print $2}' "$SNAP0")
[ -n "$E0" ] || die "no census epoch in /metrics"
# hammer: poll both porcelains continuously for the wave's whole duration
R "for i in \$(seq 2000); do curl -s -m 5 localhost:$PORT/metrics -o /dev/null; curl -s -m 5 localhost:$PORT/v1/stats -o /dev/null; done" &
HAMMER=$!
# the wave: distinct prompts (no warm hits -> cold admissions), real decode
WPIDS=()
for i in $(seq "$WAVE"); do
    R "curl -s -m 600 -X POST localhost:$PORT/v1/messages -H 'content-type: application/json' \
        -d '{\"model\":\"ds4\",\"max_tokens\":96,\"messages\":[{\"role\":\"user\",\"content\":\"Wave tenant $i $TS: write $i short sentences about ledgers.\"}]}' -o /tmp/d0bshadow_w$i.json" &
    WPIDS+=($!)
done
for p in "${WPIDS[@]}"; do wait "$p" 2>/dev/null || true; done
wait "$HAMMER" 2>/dev/null || true
OKS=$(R "grep -l '\"type\":\"message\"' /tmp/d0bshadow_w*.json 2>/dev/null | wc -l")
[ "$OKS" -ge "$WAVE" ] || die "wave incomplete ($OKS/$WAVE terminal messages)"
TORN=$(met ds4_memory_census_torn_fallbacks_total)
CFAULTS=$(met ds4_memory_census_faults_total)
GFAULTS=$(met ds4_memory_governor_faults_total)
E1=$(met ds4_memory_census_epoch)
EO=$(R "curl -s -m 20 localhost:$PORT/metrics" | grep 'class="engine_other",state="allocated"' | awk '{s+=$NF} END{print s}')
[ "$TORN" = "0" ] || die "torn fallbacks=$TORN"
[ "$CFAULTS" = "0" ] || die "census faults=$CFAULTS (includes seqlock tripwire)"
[ "$GFAULTS" = "0" ] || die "governor faults=$GFAULTS"
[ "$EO" = "0" ] || die "ENGINE_OTHER=$EO after wave"
python3 -c "exit(0 if int($E1)%2==0 and int($E1)>int($E0) else 1)" \
    || die "epoch not even+advancing ($E0 -> $E1)"
log "(t) PASS (wave=$OKS/$WAVE torn=0 census_faults=0 gov_faults=0 engine_other=0 epoch $E0->$E1)"

log "(r) shadow-claim reconciliation: boot-settle exact + wave-settle attributed"
sleep 15   # settle: wave drained, idle
R "curl -s -m 20 localhost:$PORT/metrics" > "$SNAP1"
[ -s "$SNAP1" ] || die "empty wave-settle /metrics snapshot"
MET=$(cat "$SNAP1")
python3 - "$SNAP0" "$SNAP1" <<'EOF' || die "reconciliation failed"
import re, sys
def parse(path):
    lease, cens = {}, {}
    for ln in open(path):
        m = re.match(r'ds4_memory_lease_bytes\{consumer="([^"]+)",field="([^"]+)"\} (\d+)', ln)
        if m: lease[(m.group(1), m.group(2))] = int(m.group(3)); continue
        m = re.match(r'ds4_memory_bytes\{domain="([^"]+)",class="([^"]+)",state="allocated"\} (\d+)', ln)
        if m: cens[(m.group(1), m.group(2))] = int(m.group(3))
    return lease, cens
l0, c0 = parse(sys.argv[1]); l1, c1 = parse(sys.argv[2])
DEV = "unified_device"
# scalars_mirror (both domains): lazy first-use ensure of the decode/row
# scalar mirror pairs (ds4_gpu_decode_scalars_ensure + batch-row kin) --
# boot prewarm is serial-side, so the first batched wave establishes the
# cont-path mirrors once (observed +3480 dev / +6920 host, engine-
# lifetime, census-noted alloc/free).  Adjudicated 2026-08-11, run 2.
GROWABLE = {(DEV, "batch_bank"), (DEV, "scratch_sticky"), (DEV, "kernel_partials"),
            (DEV, "scalars_mirror"), ("pinned_host", "scalars_mirror")}
ok = True
def chk(name, cond, detail):
    global ok
    print("%-38s %s  %s" % (name, "OK" if cond else "FAIL", detail))
    ok = ok and cond
def tot(c): return sum(c.values())
# BOOT SETTLE: the ledger is exact at its own definition point.
boot_i = l0[("engine_boot", "intent")]; pre_i = l0[("prewarm", "intent")]
bank_i0, bank_r0 = l0[("batch_bank_plan", "intent")], l0[("batch_bank_plan", "resident")]
bank_c0 = c0[(DEV, "batch_bank")]
chk("boot: engine == total-bank-prewarm", boot_i == tot(c0) - bank_c0 - pre_i,
    "lease=%d recomputed=%d" % (boot_i, tot(c0) - bank_c0 - pre_i))
chk("boot: bank lease == bank census", bank_r0 == bank_i0 == bank_c0,
    "resident=%d intent=%d census=%d" % (bank_r0, bank_i0, bank_c0))
# WAVE SETTLE: boot leases never move; the bank lease sandwiches census.
chk("settle: boot leases unmoved",
    l1[("engine_boot", "intent")] == boot_i and l1[("prewarm", "intent")] == pre_i,
    "engine %d->%d prewarm %d->%d" % (boot_i, l1[("engine_boot", "intent")],
                                      pre_i, l1[("prewarm", "intent")]))
scr, kp = c1.get((DEV, "scratch_sticky"), 0), c1.get((DEV, "kernel_partials"), 0)
chk("settle: prewarm bounded by classes", 0 < pre_i <= scr + kp,
    "prewarm=%d scratch+partials=%d" % (pre_i, scr + kp))
bank_i1, bank_r1 = l1[("batch_bank_plan", "intent")], l1[("batch_bank_plan", "resident")]
bank_c1 = c1[(DEV, "batch_bank")]
chk("settle: bank r <= census <= intent", bank_r1 <= bank_c1 <= bank_i1,
    "resident=%d census=%d intent=%d" % (bank_r1, bank_c1, bank_i1))
# GROWTH ATTRIBUTION: every changed class is named-growable, nothing else
# moved -- the zero-unexplained-gaps line.  (Shrink is also a change:
# trim does not run in this leg, so any delta off the list is a finding.)
stray = []
for key in sorted(set(c0) | set(c1)):
    d = c1.get(key, 0) - c0.get(key, 0)
    if d != 0 and key not in GROWABLE:
        stray.append("%s/%s %+d" % (key[0], key[1], d))
chk("settle: growth fully attributed", not stray,
    "; ".join(stray) if stray else
    "bank %+d scratch %+d partials %+d mirror %+d" % (
        bank_c1 - bank_c0,
        scr - c0.get((DEV, "scratch_sticky"), 0),
        kp - c0.get((DEV, "kernel_partials"), 0),
        sum(c1.get((d, "scalars_mirror"), 0) - c0.get((d, "scalars_mirror"), 0)
            for d in (DEV, "pinned_host"))))
sys.exit(0 if ok else 1)
EOF
log "(r) PASS"

log "(d) decision audit"
ADMITS=$(R "curl -s -m 20 localhost:$PORT/v1/stats -H 'Accept: application/json'" | python3 -c "
import json,sys; b=json.load(sys.stdin)['cache']
print(sum(int(b.get(k,0)) for k in ('admits_cold','admits_warm','admits_fork','admits_partial_fork','admits_partial_truncate','cont_admit_rejects')))" 2>/dev/null || echo "")
BANK_CELLS=$(echo "$MET" | grep 'ds4_memory_decisions_total{consumer="batch_bank_plan"' | awk '{s+=$NF} END{print s}')
DISAGREE=$(echo "$MET" | grep 'ds4_memory_decisions_total{' | grep -v 'reason="agree"' | awk '{s+=$NF} END{print s}')
if [ -n "$ADMITS" ] && [ "$ADMITS" != "0" ]; then
    # memgov D2-1: cells count admissions that TOOK a memory verdict --
    # a zero-growth warm admit (mneed==0) spends nothing, so neither live
    # nor shadow quotes it (the epoch-58 phantom-unfunded fix).  This
    # wave is 8-wide COLD on a fresh boot: every admission projects
    # growth, so exact equality stands.  A future wave that adds warm
    # reuse must subtract its zero-growth admits before asserting.
    # D2-2b: +1 = the boot bank-fit quote (one governed check per
    # successful persistent-ctx create; a healthy boot creates once and
    # never descends).
    [ "$BANK_CELLS" = "$((ADMITS + 1))" ] || die "decision-cell sum ($BANK_CELLS) != growth admissions+rejects + boot fit ($ADMITS + 1)"
else
    [ -n "$BANK_CELLS" ] && [ "$BANK_CELLS" -ge "$WAVE" ] || die "bank decision cells ($BANK_CELLS) < wave ($WAVE)"
fi
[ "$DISAGREE" = "0" ] || { echo "$MET" | grep 'ds4_memory_decisions_total{' | grep -v 'reason="agree"' | awk '$NF!=0'; die "real disagreements present ($DISAGREE)"; }
DIS_LOG=$(R "grep -c 'memgov shadow DISAGREE' /tmp/d0bshadow_boot.log" || true)
[ "${DIS_LOG:-0}" = "0" ] || die "DISAGREE disclosures in server log ($DIS_LOG)"
log "(d) PASS (bank cells=$BANK_CELLS admissions=$ADMITS disagreements=0)"

log "(e) D2-2b deterministic fit refusal (pinned headroom -> serial-only boot)"
# The enforce leg the ratchet demands: a headroom pinned above any real
# free makes budget 0 DETERMINISTICALLY (no settle dependence), so the
# governed fit must refuse the batch ctx at every fit-or-reduce halving
# and the server must land on the per-call path and still serve.  This
# boot's log (/tmp/d0bfit_boot.log) contains INTENTIONAL
# SHADOW_STRICTER DISAGREE disclosures (the quote refusing what legacy
# would have poked) -- it is EXCLUDED from zero-DISAGREE audits BY PATH;
# never point a battery audit at it.
[ -n "$SPID" ] && R "kill $SPID" 2>/dev/null; SPID=""
R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill' 2>/dev/null
sleep 10
ssh -o ConnectTimeout=10 "$HOST" \
    "cd $TEST_TREE && DS4_BATCH_FIT_HEADROOM_MB=1000000 setsid nohup ./ds4-server -c $CTX > /tmp/d0bfit_boot.log 2>&1 < /dev/null & exit 0" &
LP=$!; sleep 5; kill $LP 2>/dev/null || true
BOOTED=0
for i in $(seq 240); do
    R "grep -q 'listening on http' /tmp/d0bfit_boot.log" && { BOOTED=1; break; }
    R "ps -eo comm | grep -q '^ds4-server'" || break
    sleep 2
done
[ "$BOOTED" = "1" ] || { R "tail -5 /tmp/d0bfit_boot.log"; die "(e) refused boot did not reach listening"; }
SPID=$(R "pgrep -x ds4-server | head -1")
R "grep -q 'memgov modes: boot=enforce prewarm=enforce bank=enforce serial=enforce static=enforce' /tmp/d0bfit_boot.log" \
    || die "(e) mode board not at enforce defaults"
ENF=$(R "grep -c 'memgov ENFORCE refuse site=boot_bank_fit' /tmp/d0bfit_boot.log" || true)
[ "${ENF:-0}" -ge 1 ] || die "(e) no boot_bank_fit ENFORCE refusal in log"
R "grep -q 'memgov refused the bank plan' /tmp/d0bfit_boot.log" \
    || die "(e) typed refusal text missing from the shell's unavailable line"
R "grep -q 'persistent batch ctx unavailable' /tmp/d0bfit_boot.log" \
    || die "(e) server did not fall back to the per-call path"
R "grep -q 'persistent batch ctx ready' /tmp/d0bfit_boot.log" \
    && die "(e) a batch ctx was created despite the refusal"
STRAY=$(R "grep 'memgov shadow DISAGREE' /tmp/d0bfit_boot.log | grep -vc 'site=boot_bank_fit'" || true)
[ "${STRAY:-0}" = "0" ] || die "(e) DISAGREE disclosures beyond the fit site ($STRAY)"
# The refused shape still serves (per-call batch path, real decode).
R "curl -s -m 300 -X POST localhost:$PORT/v1/messages -H 'content-type: application/json' \
    -d '{\"model\":\"ds4\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}]}'" \
    | grep -q '\"type\":\"message\"' || die "(e) refused-shape boot does not serve"
log "(e) PASS (enforce_refusals=$ENF, per-call fallback serves)"

log "(f) D2-5 full-board composite: every family's enforce refusal in one boot"
# Pinned fit headroom (budget 0 -> BANK refuses the batch ctx at every
# halving) + pinned operator floor (PREWARM's quote refuses -> prewarm
# skipped; STATIC's per-call quote refuses -> single-path fallback).
# The SERIAL lane still SERVES through it all: its protected term is
# the session MARGIN, not the floor (the D2-4 law made visible).  Two
# concurrent batchable requests coalesce, hit the per-call refusal,
# and both still complete via the fallback.  This log
# (/tmp/d0bboard_boot.log) carries INTENTIONAL refusal disclosures --
# EXCLUDED from zero-DISAGREE audits BY PATH.
# COALESCE_WAIT_MS=1000 (second-run finding): the wait defaults to 0,
# so the gather only groups jobs ALREADY queued -- concurrent curls
# arrive with ms skew and each collapsed to run_job_single (served,
# but never touching the per-call path this leg exists to refuse).
# The knob exists exactly for burst-arrival skew.
# BOOT UN-PINNED (08-14, memgov-mapped-serve-hardening `0998f5a`): this
# leg once pinned DS4_MEMGOV_BOOT=observe because the materializer's
# claims also carry the floor, and on this box geometry (85 GiB model,
# ~34 GiB slack) any floor big enough to refuse prewarm (>~30 GiB) also
# starves eager materialization -- the model boots fully COVERED, and
# the degraded serve path was BROKEN under total refusal ("cuda prefill
# failed" + a 4 KiB unclaimed post-freeze arena range).  Both gaps are
# fixed: a governed refusal now degrades to the coherent mapped pointer
# (raw ATS mapped-serve -- slow, correct), and the cold device-copy
# tier takes a governed "cold_materialize" claim under the funnel
# marker.  The composite therefore runs the TRUE FULL BOARD: the
# materializer's total refusal (funded 0/10) is now a SUBJECT of this
# leg alongside PREWARM/STATIC/SERIAL; BANK's refusals are leg (e)'s.
# Materializer refusals disclose as bounded DISAGREE lines plus the
# funded-0 porcelain (no per-site ENFORCE-refuse fprintf today); the
# stray-refusal allowlist below carries the three materializer sites
# for when they grow ENFORCE disclosures of their own.
[ -n "$SPID" ] && R "kill $SPID" 2>/dev/null; SPID=""
R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill' 2>/dev/null
sleep 10
ssh -o ConnectTimeout=10 "$HOST" \
    "cd $TEST_TREE && DS4_BATCH_FIT_HEADROOM_MB=1000000 DS4_MEM_FLOOR_GB=600 DS4_SERVER_COALESCE_WAIT_MS=1000 setsid nohup ./ds4-server -c $CTX > /tmp/d0bboard_boot.log 2>&1 < /dev/null & exit 0" &
LP=$!; sleep 5; kill $LP 2>/dev/null || true
BOOTED=0
for i in $(seq 240); do
    R "grep -q 'listening on http' /tmp/d0bboard_boot.log" && { BOOTED=1; break; }
    R "ps -eo comm | grep -q '^ds4-server'" || break
    sleep 2
done
[ "$BOOTED" = "1" ] || { R "tail -5 /tmp/d0bboard_boot.log"; die "(f) pinned-board boot did not reach listening"; }
SPID=$(R "pgrep -x ds4-server | head -1")
R "grep -q 'memgov modes: boot=enforce prewarm=enforce bank=enforce serial=enforce static=enforce' /tmp/d0bboard_boot.log" \
    || die "(f) mode board not at full-enforce defaults (BOOT re-pinned?)"
R "grep -q 'materialize base: funded 0/10' /tmp/d0bboard_boot.log" \
    || die "(f) materializer not fully refused (expected 'materialize base: funded 0/10' under the pinned floor)"
R "grep -q 'memgov ENFORCE refuse site=boot_bank_fit' /tmp/d0bboard_boot.log" \
    || die "(f) no BANK fit refusal"
R "grep -q 'memgov ENFORCE refuse site=boot_prewarm' /tmp/d0bboard_boot.log" \
    || die "(f) no PREWARM refusal"
R "grep -q 'skipping prewarm' /tmp/d0bboard_boot.log" \
    || die "(f) prewarm not skipped on refusal"
R "grep -q 'persistent batch ctx ready' /tmp/d0bboard_boot.log" \
    && die "(f) a batch ctx was created despite the pinned headroom"
# Two concurrent batchable requests -> coalesce -> per-call STATIC
# refusal -> single-path fallback serves BOTH via the serial lane.
# OpenAI surface + GREEDY + NON-THINKING (findings 3-5, the last one
# measured via route_decisions on a probe boot: cont_unavailable=2):
# the static lane qualifies unconditionally only on its home surface
# and only for NEEDS-FREE shapes -- buffered, greedy, non-thinking.
# The server's default reasoning effort is LOW (thinking ON), which
# sets DS4_NEED_THINKING; reasoning_effort:none + temperature:0 make
# the pair needs-free.
FP=()
for i in 1 2; do
    R "curl -s -m 300 -X POST localhost:$PORT/v1/chat/completions -H 'content-type: application/json' \
        -d '{\"max_tokens\":16,\"temperature\":0,\"reasoning_effort\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"Board leg request $i: say OK.\"}]}' -o /tmp/d0bboard_r$i.json" &
    FP+=($!)
done
for p in "${FP[@]}"; do wait "$p" 2>/dev/null || true; done
OKS=$(R "grep -l 'finish_reason' /tmp/d0bboard_r*.json 2>/dev/null | wc -l")
[ "$OKS" = "2" ] || { R "cat /tmp/d0bboard_r1.json /tmp/d0bboard_r2.json 2>/dev/null | head -4"; die "(f) fallback served $OKS/2"; }
R "grep -q 'memgov ENFORCE refuse site=static_batch_percall' /tmp/d0bboard_boot.log" \
    || die "(f) no STATIC per-call refusal (requests did not coalesce onto the per-call path?)"
R "grep -q 'memgov refused the per-call graph' /tmp/d0bboard_boot.log" \
    || die "(f) fallback reason line missing"
STRAYF=$(R "grep 'memgov ENFORCE refuse' /tmp/d0bboard_boot.log | grep -vcE 'site=(boot_bank_fit|boot_prewarm|static_batch_percall|boot_materialize|lazy_materialize|cold_materialize)'" || true)
[ "${STRAYF:-0}" = "0" ] || die "(f) unexpected enforce refusals beyond the pinned families ($STRAYF)"
# The hardening's degraded-path contract: a fully-refused (COVERED)
# model must SERVE from the coherent mapped tier -- zero prefill
# failures, zero unclaimed post-freeze publications (funnel tripwire).
PFAIL=$(R "grep -c 'prefill failed' /tmp/d0bboard_boot.log" || true)
[ "${PFAIL:-0}" = "0" ] || { R "grep 'prefill failed' /tmp/d0bboard_boot.log | head -3"; die "(f) prefill failures under the full board ($PFAIL)"; }
TRIPW=$(R "grep -c 'POST-FREEZE arena range' /tmp/d0bboard_boot.log" || true)
[ "${TRIPW:-0}" = "0" ] || { R "grep 'POST-FREEZE arena range' /tmp/d0bboard_boot.log | head -3"; die "(f) funnel tripwire fired ($TRIPW unclaimed post-freeze publication(s))"; }
log "(f) PASS (full board: BOOT starves the materializer (funded 0/10), BANK+PREWARM+STATIC refuse; mapped-tier fallback serves 2/2, prefill-failed 0, tripwire 0)"

cleanup
trap - EXIT
log "D0B SHADOW GATE: ALL LEGS PASS"
