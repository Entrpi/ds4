#!/bin/bash
# memory/residency.sh — memgov D1a STAGE CLOSE gate (stage spec:
# local/docs/v057/d1a_scoping_2026-08-11.md §4; authority plan §12).
# The equivalence matrix: every supported boot shape must produce a
# reconciled per-source ledger, clean semantic catalogs + unit tables,
# and a fully plan-known range registry (D1a-4 funnel), with boot
# decision families byte-identical (normalized) to the pre-D1a control
# tree and golden decode vectors in band per residency mode.
#
# LEGS
#   (l1) base-only          --no-spec
#   (l2) base+mtp           --no-dspark
#   (l3) base+mtp+drafter   --mtp <gguf>   (launch defaults drop mtp when
#                                           the drafter arms; the flag
#                                           overrides — all 3 roles live)
#   (l4) SLICE boot         --role coordinator --listen 127.0.0.1 9411
#                           --layers 20:40 --no-spec
#        (coordinator, not worker: a worker validates --coordinator HOST
#        PORT before anything runs — l4 first-run finding.  The
#        coordinator opens its own engine slice [20:40]+output with no
#        peer required, which is exactly the boot shape this leg gates.)
#        Boot-shape only (no workers -> no serving): catalog + unit
#        tables + reconcile + zero faults/refusals asserted from the boot
#        log.  memgov D1b-1: planned==total IS asserted here — the eager
#        pass materializes the bound slice plan, so any off-plan
#        promotion (the retired D1a-4 5/10 slice-blindness) dies loudly.
#        This leg is the D1b regression needle.
#   (l5) raw                DS4_CUDA_NO_DERIVED_WEIGHTS=1
#                           DS4_CUDA_NO_HBM_CACHE=1 (+ pinned vmm budget
#                           for the control pair, d0a nohbm law)
#        + golden vectors under the same env in CONTROL mode: raw runs
#        different kernel families than the stock-recorded fixture and
#        sits outside its band BY CONSTRUCTION (control A/B 08-12
#        reproduced 3/5 45/64 max_abs 9.69399 digit-for-digit on the
#        pre-D1a tree) -- the oracle is exact test==control equivalence.
#   (l6) self-load          stock zero-config; asserts the BUILT artifact
#        banner + golden vectors under stock env.
#   (l7) manifest           .33-local ds4_weight_server --scope base,
#        DS4_CUDA_WEIGHT_IPC_MANIFEST import boot --no-spec + golden
#        vectors under the manifest env.  SKIP only with a receipt note.
#
# Each server leg (l1,l2,l3,l5,l6,l7): (a) one serial decode returns a
# terminal message with census faults 0 before AND after; (b) decision-
# family parity vs the control tree (~/code/ds4-cebaff7base) with the
# d0a normalization + fit-jitter adjudication; (c) Σ-sources reconcile;
# (d) zero UNIT-TABLE FAULT / RANGE-PUBLISH REFUSED / MEM-CENSUS FAULT
# lines and fully-planned range-plan lines.
#
# Usage: bash gates/legs/memory/residency.sh   (from the repo root)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HOST=${HOST:-${DS4_BOX33:?}}
TEST_TREE=${DS4_BOX_TREE:?}
CTRL_TREE='~/code/ds4-cebaff7base'
CTX=${CTX:-32768}
PORT=8000
BASE_GGUF=${DS4_GGUF_DIR:?}/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MTP_GGUF=${DS4_GGUF_DIR:?}/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
# Golden legs pair the -0731 gguf with the 0731-recorded fixture (the
# documented-green pairing; GOLDEN-HARNESS LAW).  BASE_GGUF -- what
# zero-config SERVING resolves -- was the May-18 OLD checkpoint at the
# D1a close (run-8 finding, mismatch signature max_abs ~10); the user's
# card session made it a SYMLINK to the -0731 file on 2026-08-12
# (verify: readlink $BASE_GGUF).  Keeping the explicit GOLDEN_GGUF
# pairing anyway: the law binds vectors to a NAMED checkpoint, never to
# whatever serving currently resolves.
GOLDEN_GGUF=${DS4_TEXT_BASE:?}
GOLDEN_VEC=tests/test-vectors/local-golden-0731.vec
WS_MANIFEST=/tmp/d1a_ws.manifest
TS=$(date +%s); NONCE="d1ares-$TS-$$"
LOCK=/tmp/ds4_box_lock
R() { ssh -o ConnectTimeout=10 "$HOST" "$@"; }
SPID=""; WSPID=""
log() { echo "[$(date +%H:%M:%S)] $*"; }
cleanup() {
    [ -n "$SPID" ] && R "kill $SPID" 2>/dev/null
    [ -n "$WSPID" ] && R "kill $WSPID" 2>/dev/null
    R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill' 2>/dev/null
    R 'ps -eo pid,comm | awk '"'"'$2=="ds4_weight_serv"{print $1}'"'"' | xargs -r kill' 2>/dev/null
    R "rm -rf $LOCK" 2>/dev/null
}
die() { log "FAIL: $*"; cleanup; exit 1; }

R "ps -eo comm | grep -q '^ds4-server' && echo BUSY || echo IDLE" | grep -q IDLE \
    || die "a ds4-server is already running on $HOST"
R "mkdir $LOCK 2>/dev/null && echo $NONCE > $LOCK/nonce" \
    || die "box lock held: $(R cat $LOCK/nonce 2>/dev/null || echo unknown)"
trap cleanup EXIT
log "lock acquired ($NONCE)"

boot_tree() { # $1 tree  $2 env  $3 flags  $4 log  [$5 ready-regex, default HTTP]
    local ready=${5:-'listening on http'}
    ssh -o ConnectTimeout=10 "$HOST" \
        "cd $1 && setsid nohup env $2 ./ds4-server -c $CTX $3 > $4 2>&1 < /dev/null & exit 0" &
    local lp=$!; sleep 5; kill $lp 2>/dev/null || true
    for i in $(seq 300); do
        R "grep -q '$ready' $4" && { SPID=$(R "pgrep -x ds4-server | head -1"); return 0; }
        R "ps -eo comm | grep -q '^ds4-server'" || { R "tail -5 $4"; return 1; }
        sleep 2
    done
    return 1
}
kill_server() {
    [ -n "$SPID" ] && R "kill $SPID" 2>/dev/null
    for i in $(seq 20); do R "ps -eo comm | grep -q '^ds4-server'" || { SPID=""; sleep 5; return 0; }; sleep 2; done
    R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill -9'; SPID=""; sleep 5
}
met() { R "curl -s -m 20 localhost:$PORT/metrics" | awk -v k="$1" '$1==k{print $2}'; }
decode_one() { # $1 out json, $2 tag
    R "curl -s -m 600 -X POST localhost:$PORT/v1/messages -H 'content-type: application/json' \
        -d '{\"model\":\"ds4\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"Gate $2 $TS: name three rivers and one ledger.\"}]}' -o $1"
    R "grep -q '\"type\":\"message\"' $1"
}
avail_mb() { R "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo"; }
stable_avail() { # two reads within 150M (d0a ABORT-on-unstable law)
    local a b
    for t in 1 2 3 4 5; do
        a=$(avail_mb); sleep 5; b=$(avail_mb)
        local d=$((a - b)); [ ${d#-} -lt 150 ] && { echo "$b"; return 0; }
    done
    return 1
}
# Decision families, normalized exactly like memory/census.sh: strip
# timestamps, mask the free-derived inputs; formulas + verdict shapes
# must byte-match.  The test tree's D1a lines are excluded by grep
# construction (the control cannot print them) -- the D1a-4 range-plan
# porcelain ends in "refused N" and would otherwise leak in through the
# 'refus' family pattern (l1 first-run finding, 2026-08-12).
extract_decisions() { # $1 remote log  $2 local out
    # memgov D2-2b amendment (08-14): the fit CLAMP line is excluded from
    # the byte-diff -- its PRESENCE is free-input-derived (prints only
    # when n < requested), and on a fresh box whose free sits at the
    # requested-count boundary (~23.2 GiB at 32 banks) a T/C pair can
    # straddle the print threshold (observed: T clamps 31, C funds 32 --
    # presence asymmetry no operand mask can absorb).  The line's
    # shape-derived operands (per_bank, headroom, substrate) are still
    # byte-compared by fit_shape when both sides print; max_seq keeps its
    # separate jitter adjudication, extended with the presence rule.
    R "grep -E 'batch fit|batch vmm|mem floor|boot ledger|refus|reject|no graph fits|admission|serial fit' $1" \
        | grep -v 'batch fit: free=' \
        | grep -v 'ds4: range plan ' \
        | sed -E 's/^[0-9]{4} [0-9:]{8} //; s/free=[0-9.]+ GiB/free=# GiB/g; s/MemAvailable[= ][0-9.]+/MemAvailable=#/g; s/capacity [0-9.]+ GiB/capacity # GiB/g; s/max_seq [0-9]+/max_seq #/g; s/x [0-9]+ banks/x # banks/g; s/budget=[0-9.]+ GiB/budget=# GiB/g; s/\[plan [0-9.]+, capacity [0-9.]+\]/[plan #, capacity #]/g' \
        > "$2"
}
fit_free_gib() { R "grep 'batch fit' $1 | head -1" | grep -o 'free=[0-9.]*' | cut -d= -f2; }
fit_max_seq()  { R "grep 'batch fit' $1 | head -1" | grep -o 'max_seq [0-9]*' | awk '{print $2}'; }
fit_requested() { R "grep -m1 'batch fit: free=' $1" | grep -o '(requested [0-9]*' | grep -o '[0-9]*'; }
# The clamp line with its free-input operands masked: what remains is
# shape-derived (per_bank, headroom, substrate) and must byte-match.
fit_shape() { R "grep -m1 'batch fit: free=' $1" | sed -E 's/free=[0-9.]+ GiB/free=# GiB/; s/max_seq [0-9]+/max_seq #/'; }
# D2-2b v2: formula-oracle adjudication of a T/C fit pair.  args:
# T clamp line (may be empty = unclamped), C clamp line, T banks, C banks
# ('-' = absent).  Prints PASS.../FAIL.../RETRY... -- FAIL means a side's
# printed decision violates its own inputs (real, die); RETRY means
# input noise (band/direction/distance), worth one fresh pair.
fit_adjudicate() {
    python3 - "$1" "$2" "$3" "$4" <<'PYEOF'
import sys, re
def parse(ln):
    m = re.search(r'free=([0-9.]+) GiB headroom=([0-9.]+) GiB substrate outstanding=([0-9.]+) GiB per_bank=([0-9.]+) MiB -> max_seq ([0-9]+) \(requested ([0-9]+)\)', ln)
    if not m: return None
    return dict(free=float(m.group(1)), hr=float(m.group(2)), sub=float(m.group(3)),
                pb=float(m.group(4)), n=int(m.group(5)), req=int(m.group(6)))
t, c = parse(sys.argv[1]), parse(sys.argv[2])
bt, bc = sys.argv[3], sys.argv[4]
if t is None and c is None:
    # Neither side clamped: nothing derived from free -- vmm banks must
    # be exactly equal (also the l5 pinned-budget path).
    print(("PASS unclamped both; banks %s/%s exactly equal" % (bt, bc))
          if bt == bc else ("RETRY unclamped both but banks %s vs %s" % (bt, bc)))
    sys.exit(0)
for side, d in (('T', t), ('C', c)):
    if d is None: continue
    budget = max(d['free'] - d['hr'] - d['sub'], 0.0) * 1024.0
    if abs(int(budget / d['pb']) - d['n']) > 1:      # <=1: print-rounding at a boundary
        print("FAIL %s formula self-check: inputs give %d, printed %d"
              % (side, int(budget / d['pb']), d['n'])); sys.exit(0)
pres = t or c
req = pres['req']
nt = t['n'] if t else req                            # unclamped side funded its request
nc = c['n'] if c else req
band_banks = 1 + int(1024.0 / pres['pb'])            # the 1.0 GiB band, in banks
dist = nt - nc
if t and c:
    if abs(t['free'] - c['free']) >= 1.0:
        print("RETRY free inputs outside the 1.0 GiB band (T=%.2f C=%.2f)"
              % (t['free'], c['free'])); sys.exit(0)
    if dist != 0 and (dist > 0) != (t['free'] > c['free']):
        print("RETRY direction mismatch (n T=%d C=%d, free T=%.2f C=%.2f)"
              % (nt, nc, t['free'], c['free'])); sys.exit(0)
if abs(dist) > band_banks:
    print("RETRY decision distance %d beyond band (%d banks)" % (dist, band_banks)); sys.exit(0)
def bank_ok(b, n): return b in ('-', '') or int(b) == n
if not bank_ok(bt, nt) or not bank_ok(bc, nc):
    print("FAIL vmm banks do not match own plan (T %s/%d, C %s/%d)" % (bt, nt, bc, nc)); sys.exit(0)
print("PASS formula-exact both sides; n T=%d C=%d dist=%d (band %d banks)"
      % (nt, nc, dist, band_banks))
PYEOF
}
vmm_banks()    { R "grep 'batch vmm' $1 | head -1" | grep -o 'x [0-9]* banks' | grep -o '[0-9]*'; }

# ---- shared assert batteries ----
plan_asserts() { # $1 log  $2.. expected source names
    local L=$1; shift
    R "grep -q 'ds4: model-source ledger reconciled:' $L" || die "$L: no reconcile line"
    R "grep -q 'source ledger mismatch' $L" && die "$L: ledger mismatch"
    R "grep -q 'model source UNATTRIBUTED' $L" && die "$L: unattributed weight bytes"
    R "grep -q 'UNIT-TABLE FAULT' $L" && die "$L: unit-table fault"
    R "grep -q 'RANGE-PUBLISH REFUSED' $L" && die "$L: range publication refused"
    R "grep -q 'MEM-CENSUS FAULT' $L" && die "$L: census fault line"
    local s
    for s in "$@"; do
        R "grep -q 'ds4: model source $s \[' $L" || die "$L: no $s source line"
        R "grep -q 'ds4: model catalog $s:' $L" || die "$L: no $s catalog line"
        R "grep -q 'ds4: model units $s: .* (verify=0)' $L" || die "$L: no clean $s unit table"
        R "grep -q 'ds4: range plan $s:' $L" || die "$L: no $s range-plan line"
        # memgov D1b-1: the eager pass runs per source, plan-first, and
        # never fails a funded unit on a healthy boot.
        R "grep -q 'ds4: materialize $s: funded .* failed 0' $L" \
            || die "$L: no clean $s materialize line"
    done
}
range_plan_full() { # $1 log — planned == total + multi 0 + refused 0 on EVERY line
    # memgov D1b-3: multi counts device-copy ranges spanning >1 unit
    # (imports/registered excluded engine-side) — any nonzero is an
    # extent-shaped promotion, a materializer regression.
    local BAD
    BAD=$(R "grep 'ds4: range plan ' $1" \
        | sed -E 's/.*model ([0-9]+)\/([0-9]+) planned, derived ([0-9]+)\/([0-9]+), q8 ([0-9]+)\/([0-9]+), multi ([0-9]+), refused ([0-9]+).*/\1 \2 \3 \4 \5 \6 \7 \8/' \
        | awk '$1!=$2 || $3!=$4 || $5!=$6 || $7!=0 || $8!=0 {print}')
    [ -z "$BAD" ] || die "$1: range plan incomplete/multi/refused: $BAD"
}
faults_zero() { # $1 tag
    local CF; CF=$(met ds4_memory_census_faults_total)
    [ "$CF" = "0" ] || die "$1: census faults=$CF"
}

# ---- paired server leg: test asserts + decode, then control parity ----
pair_leg() { # $1 name  $2 env  $3 flags  $4.. expected sources
    local name=$1 env=$2 flags=$3; shift 3
    local attempt TLOG=/tmp/d1ares_${name}_T.log CLOG=/tmp/d1ares_${name}_C.log
    for attempt in 1 2; do
        log "($name) attempt $attempt: test boot"
        stable_avail >/dev/null || die "$name: MemAvailable unstable pre-boot (ABORT)"
        boot_tree "$TEST_TREE" "$env" "$flags" "$TLOG" || die "$name test boot failed"
        plan_asserts "$TLOG" "$@"
        range_plan_full "$TLOG"
        faults_zero "$name boot"
        decode_one "/tmp/d1ares_${name}_dec.json" "$name" || die "$name decode failed"
        sleep 3
        faults_zero "$name after decode"
        kill_server
        log "($name) attempt $attempt: control boot"
        stable_avail >/dev/null || die "$name: MemAvailable unstable between pair (ABORT)"
        boot_tree "$CTRL_TREE" "$env" "$flags" "$CLOG" || die "$name control boot failed"
        kill_server
        extract_decisions "$TLOG" "/tmp/d1ares_${name}_T.dec"
        extract_decisions "$CLOG" "/tmp/d1ares_${name}_C.dec"
        if ! diff -q "/tmp/d1ares_${name}_T.dec" "/tmp/d1ares_${name}_C.dec" >/dev/null; then
            diff "/tmp/d1ares_${name}_T.dec" "/tmp/d1ares_${name}_C.dec" | head -12
            die "$name: decision families differ (normalized byte-diff)"
        fi
        local ft fc mt mc bt bc mv bv band req pres
        ft=$(fit_free_gib "$TLOG"); fc=$(fit_free_gib "$CLOG")
        mt=$(fit_max_seq  "$TLOG"); mc=$(fit_max_seq  "$CLOG")
        bt=$(vmm_banks    "$TLOG"); bc=$(vmm_banks    "$CLOG")
        log "($name) inputs: free T=$ft C=$fc GiB; max_seq T=$mt C=$mc; vmm_banks T=${bt:--} C=${bc:--}"
        if [ -z "$ft" ] && [ -z "$fc" ]; then
            # Pinned-budget shapes (l5 raw: banks come straight from
            # DS4_BATCH_VMM_BUDGET_MB) print NO free-derived 'batch fit'
            # family -- there is nothing to band-check and no jitter
            # excuse: whatever derived decisions exist must be EXACTLY
            # equal (l5 first-run finding, 2026-08-12).
            if [ "${mt:-}" = "${mc:-}" ] && [ "${bt:-}" = "${bc:-}" ]; then
                log "($name) PASS (no free-derived fit family; derived decisions exactly equal)"
                sleep 10
                return 0
            fi
            log "($name): no fit family and derived decisions unequal -- retrying pair once"
            continue
        fi
        # D2-2b v2 (l2 finding): THE FORMULA IS THE ORACLE.  Both trees
        # draw fit-time free from the same noisy boot-to-boot
        # distribution (observed 21.8-23.5 GiB on BOTH trees at the
        # basemtp shape; mmap-mode MemAvailable accounting), so
        # cross-side output equality was only ever a proxy that breaks
        # when input noise reaches one bank (496 MiB).  The real
        # invariants: (1) each side's printed decision equals the fit
        # formula applied to its OWN inputs; (2) the shape-derived
        # operands byte-match when both print; (3) free inputs sit in
        # the 1.0 GiB band; (4) cross-side decision distance is bounded
        # by band/per_bank+1 with direction tracking the inputs; (5)
        # each side's vmm banks equal its own funded plan.  A formula
        # self-check failure is REAL and dies; input-noise verdicts
        # retry the pair.
        if [ -n "$ft" ] && [ -n "$fc" ] && \
           [ "$(fit_shape "$TLOG")" != "$(fit_shape "$CLOG")" ]; then
            fit_shape "$TLOG"; fit_shape "$CLOG"
            die "$name: fit-line shape mismatch (per_bank/headroom/substrate)"
        fi
        local verdict
        verdict=$(fit_adjudicate "$(R "grep -m1 'batch fit: free=' $TLOG")" \
                                 "$(R "grep -m1 'batch fit: free=' $CLOG")" \
                                 "${bt:--}" "${bc:--}")
        case "$verdict" in
        PASS*)  log "($name) PASS ($verdict)"; sleep 10; return 0 ;;
        FAIL*)  die "$name: $verdict" ;;
        *)      log "($name): $verdict -- retrying pair once" ;;
        esac
    done
    die "$name: derived decisions diverge or inputs drift on both attempts -- NOT jitter"
}

golden_run() { # $1 tree  $2 env  $3 log -> echoes RC
    R "cd $1 && env DS4_TEST_MODEL=$GOLDEN_GGUF DS4_TEST_LOCAL_GOLDEN_FILE=$GOLDEN_VEC $2 ./ds4_test --local-golden-vectors > $3 2>&1; echo \$?"
}
golden_leg() { # $1 name  $2 env  $3 mode: fixture|control
    # NOTE: GLOG must sit in its OWN local statement -- all words of one
    # `local` command expand BEFORE any assignment lands, so a same-line
    # ${name} is unbound under set -u (l5->golden first-run finding).
    local name=$1 env=$2 mode=$3 RC
    local GLOG=/tmp/d1ares_golden_${name}.log
    log "(golden:$name) ds4_test --local-golden-vectors under $name env [$mode]"
    RC=$(golden_run "$TEST_TREE" "$env" "$GLOG")
    R "grep 'ds4-test: local golden' $GLOG" || true
    R "grep -q 'ds4-test: local golden' $GLOG" || die "golden:$name: no golden summary line"
    if [ "$mode" = "fixture" ]; then
        # The fixture band is the oracle ONLY for configs matching the
        # fixture's recorded config (aligned artifacts: self-load and
        # manifest boots).
        [ "$RC" = "0" ] || { R "tail -15 $GLOG"; die "golden:$name RC=$RC"; }
    else
        # Kernel-family modes (raw) drift outside the stock-recorded band
        # BY CONSTRUCTION (control A/B 2026-08-12: the pre-D1a tree fails
        # the fixture band identically).  Raw-mode logits are also NOT
        # run-to-run stable on ONE tree (test tree across runs on
        # identical sources: 45/64 -> 43/64, max_abs 9.69399 -> 9.85727),
        # so byte-equality with control is not a valid oracle either.
        # The evidence-backed oracle: top1 ref==cand on BOTH trees, RCs
        # equal, and gross-drift floors (top5>=2/5 top20>=12/20
        # top64>=35/64 max_abs<=12) on BOTH lines -- comfortably above
        # real breakage (wrong tiling produces near-zero overlaps),
        # comfortably below observed raw-mode noise (3/5, 16/20,
        # 43-45/64, 9.69-9.86 across four runs on two trees).
        local CRC
        CRC=$(golden_run "$CTRL_TREE" "$env" "${GLOG}.ctrl")
        R "grep 'ds4-test: local golden' ${GLOG}.ctrl" | sed 's/^/    ctrl: /' || true
        [ "$RC" = "$CRC" ] || die "golden:$name: RC differs (test=$RC control=$CRC)"
        local side line ref cand nums
        for side in "$GLOG" "${GLOG}.ctrl"; do
            line=$(R "grep 'ds4-test: local golden' $side" | tail -1)
            [ -n "$line" ] || die "golden:$name: no summary line in $side"
            ref=$(echo "$line" | sed -E 's/.*top1 ref=([0-9]+) cand=([0-9]+).*/\1/')
            cand=$(echo "$line" | sed -E 's/.*top1 ref=([0-9]+) cand=([0-9]+).*/\2/')
            [ -n "$ref" ] && [ "$ref" = "$cand" ] \
                || die "golden:$name: top1 moved in $side (ref=$ref cand=$cand)"
            nums=$(echo "$line" | sed -E 's/.*top5_overlap=([0-9]+)\/5 top20_overlap=([0-9]+)\/20 top64_overlap=([0-9]+)\/64 top20_max_abs=([0-9.]+).*/\1 \2 \3 \4/')
            echo "$nums" | awk '$1>=2 && $2>=12 && $3>=35 && $4<=12.0 {exit 0} {exit 1}' \
                || die "golden:$name: gross-drift floor violated in $side ($nums)"
        done
    fi
    log "(golden:$name) PASS ($mode)"
    sleep 10
}

# L7_ONLY=1: completion mode -- run ONLY the manifest leg (used when
# l1..l6 already passed on this exact tree+binaries and the weight
# server needed a knob fix; the receipt cites both runs).
L7_ONLY=${L7_ONLY:-0}
if [ "$L7_ONLY" != "1" ]; then

# ---- (l1..l3) source-combination legs ----
pair_leg l1_baseonly "" "--no-spec" base
pair_leg l2_basemtp  "" "--no-dspark" base mtp
pair_leg l3_allthree "" "--mtp $MTP_GGUF" base mtp drafter

# ---- (l4) slice boot: worker-shape, boot asserts only ----
log "(l4) slice boot: coordinator --layers 20:40 (boot-shape leg)"
SLOG=/tmp/d1ares_l4_slice.log
# -m is EXPLICIT here: zero-config launch defaults do not engage on
# distributed-role boots (l4 finding #2 -- the coordinator fell back to
# the ds4flash.gguf default path).
boot_tree "$TEST_TREE" "" "-m $BASE_GGUF --role coordinator --listen 127.0.0.1 9411 --layers 20:40 --no-spec" \
    "$SLOG" "backend initialized for graph diagnostics" \
    || die "l4 slice coordinator boot failed (never reached backend init)"
plan_asserts "$SLOG" base
SLICELINE=$(R "grep 'ds4: range plan base' $SLOG" | tail -1)
echo "$SLICELINE" | grep -q 'refused 0' || die "l4: slice refused != 0 ($SLICELINE)"
UNITLINE=$(R "grep 'ds4: model units base' $SLOG" | tail -1)
log "(l4) slice unit table: $UNITLINE"
# memgov D1b-1: the eager pass materializes the BOUND slice plan, so a
# slice boot's publications are fully planned BY CONSTRUCTION.  The D1a-4
# 5/10 slice-blindness measurement is retired as a hard assert — any
# regression to off-plan promotion dies here (the regression needle).
if echo "$SLICELINE" | sed -E 's/.*model ([0-9]+)\/([0-9]+) planned.*/\1 \2/' | awk '$1!=$2{exit 0} {exit 1}'; then
    die "l4: SLICE-BLIND publication (off-plan promotion): $SLICELINE"
fi
log "(l4) slice fully planned: $SLICELINE"
MATLINE=$(R "grep 'ds4: materialize base' $SLOG" | tail -1)
echo "$MATLINE" | grep -q 'failed 0' || die "l4: slice materialize unhealthy ($MATLINE)"
log "(l4) slice materialize: $MATLINE"
kill_server
sleep 10
log "(l4) PASS (boot-shape asserts; serving needs a coordinator — receipt notes the scope)"

# ---- (l5) raw + goldens ----
pair_leg l5_raw "DS4_CUDA_NO_DERIVED_WEIGHTS=1 DS4_CUDA_NO_HBM_CACHE=1 DS4_BATCH_VMM_BUDGET_MB=8000" "" base drafter
golden_leg raw "DS4_CUDA_NO_DERIVED_WEIGHTS=1 DS4_CUDA_NO_HBM_CACHE=1" control

# ---- (l6) self-load + goldens ----
pair_leg l6_selfload "" "" base drafter
R "grep -q 'ds4: aligned artifacts built in-process' /tmp/d1ares_l6_selfload_T.log" \
    || die "l6: no BUILT artifact banner on the self-load boot"
# memgov D1b-3 per-source exactness (the D1b-0 F3 restatement): on the
# stock self-load shape, base device bytes decompose EXACTLY into
# artifacts + eager-populated units (the 2-D census SPAN/ARENA
# aggregation ambiguity is gone at source scope).  2-dp porcelain =>
# +/-0.05 GiB band.
L6T=/tmp/d1ares_l6_selfload_T.log
BDEV=$(R "grep 'ds4: model source base' $L6T" | tail -1 | sed -E 's/.*device ([0-9.]+) GiB.*/\1/')
BART=$(R "grep 'ds4: aligned artifacts built in-process' $L6T" | tail -1 | sed -E 's/.*artifacts, ([0-9.]+) GiB.*/\1/')
BPOP=$(R "grep 'ds4: materialize base' $L6T" | tail -1 | sed -E 's/.*populated [0-9]+ \(([0-9.]+) GiB\).*/\1/')
python3 -c "
d, a, p = float('$BDEV'), float('$BART'), float('$BPOP')
gap = d - a - p
assert abs(gap) <= 0.05, 'per-source exactness gap %.3f GiB (dev %.2f != art %.2f + pop %.2f)' % (gap, d, a, p)
print('l6 exactness: base dev %.2f == art %.2f + populated %.2f (gap %+.3f)' % (d, a, p, gap))" \
    || die "l6: per-source exactness failed (dev=$BDEV art=$BART pop=$BPOP)"
golden_leg selfload "" fixture

fi  # L7_ONLY

# ---- (l7) manifest via .33-local weight server + goldens ----
# --reserve-gb 12: the default 32 GiB reserve fails preflight on this
# box (plan 80.76 + 32 > budget ~106 with free ~95 after a night of
# boots); 12 GiB still covers the import client's non-weight footprint
# (the weights are shared IPC memory, and the client's bank fit adapts
# to whatever is free).
log "(l7) starting ds4_weight_server (scope=base) -- repacks take minutes"
R "rm -f $WS_MANIFEST"
ssh -o ConnectTimeout=10 "$HOST" \
    "cd $TEST_TREE && setsid nohup ./ds4_weight_server --base $BASE_GGUF --scope base --reserve-gb 12 --manifest $WS_MANIFEST > /tmp/d1a_ws.log 2>&1 < /dev/null & exit 0" &
WLP=$!; sleep 5; kill $WLP 2>/dev/null || true
WS_OK=0
for i in $(seq 450); do
    R "grep -q 'ds4_weight_server: ready manifest=' /tmp/d1a_ws.log" && { WS_OK=1; break; }
    R "ps -eo comm | grep -q '^ds4_weight_serv'" || break
    sleep 2
done
if [ "$WS_OK" = "1" ]; then
    WSPID=$(R "pgrep -x ds4_weight_serv | head -1")
    log "(l7) weight server ready (pid $WSPID)"
    pair_leg l7_manifest "DS4_CUDA_WEIGHT_IPC_MANIFEST=$WS_MANIFEST" "--no-spec" base
    R "grep -q 'ds4: CUDA imported shared' /tmp/d1ares_l7_manifest_T.log" \
        || die "l7: no import engagement line"
    # The golden phase needs the WS to serve the GOLDEN model: the
    # import validates identity by model_id + FILE SIZE only, and the
    # old/0731 checkpoints are size-identical -- a BASE_GGUF-serving WS
    # feeds old-model bytes to a -0731 engine and reproduces the
    # model-mismatch signature (max_abs 10.0568) digit-for-digit
    # (l7-golden first-run finding; manifest content-hash chartered as
    # a rider).  Restart the WS on GOLDEN_GGUF for the fixture run.
    log "(l7) restarting weight server on the golden gguf for the fixture run"
    R "kill $WSPID" 2>/dev/null; WSPID=""
    for i in $(seq 30); do
        R "ps -eo comm | grep -q '^ds4_weight_serv'" || break
        sleep 2
    done
    R "rm -f $WS_MANIFEST"
    ssh -o ConnectTimeout=10 "$HOST" \
        "cd $TEST_TREE && setsid nohup ./ds4_weight_server --base $GOLDEN_GGUF --scope base --reserve-gb 12 --manifest $WS_MANIFEST > /tmp/d1a_ws_golden.log 2>&1 < /dev/null & exit 0" &
    WLP=$!; sleep 5; kill $WLP 2>/dev/null || true
    WS_OK=0
    for i in $(seq 450); do
        R "grep -q 'ds4_weight_server: ready manifest=' /tmp/d1a_ws_golden.log" && { WS_OK=1; break; }
        R "ps -eo comm | grep -q '^ds4_weight_serv'" || break
        sleep 2
    done
    [ "$WS_OK" = "1" ] || { R "tail -6 /tmp/d1a_ws_golden.log"; die "l7: golden-model weight server never reached ready"; }
    WSPID=$(R "pgrep -x ds4_weight_serv | head -1")
    golden_leg manifest "DS4_CUDA_WEIGHT_IPC_MANIFEST=$WS_MANIFEST" fixture
    R "kill $WSPID" 2>/dev/null; WSPID=""
    sleep 5
else
    R "tail -10 /tmp/d1a_ws.log" || true
    log "(l7) SKIPPED: weight server never reached ready -- RECEIPT NOTE REQUIRED"
fi

# ---- (ld) OPTIONAL discard measured leg (RUN_DISCARD_LEG=1) ----
# memgov D1b-3: REPORT-ONLY (F4).  One extra test-tree pair, stock vs
# DS4_CUDA_EAGER_SOURCE_DISCARD=1 (O_DIRECT + per-chunk discard): logs
# boot wall time, fit line, and settle MemAvailable so the F4 default
# decision accumulates evidence.  Never asserted -- the discard mode
# CHANGES fit inputs by design (the D1b-2 fit-parity law), which is
# exactly why it ships default-OFF.
if [ "${RUN_DISCARD_LEG:-0}" = "1" ]; then
    for mode in stock discard; do
        ENVV=""; [ "$mode" = "discard" ] && ENVV="DS4_CUDA_EAGER_SOURCE_DISCARD=1"
        DLOG=/tmp/d1ares_ld_${mode}.log
        T0=$(date +%s)
        boot_tree "$TEST_TREE" "$ENVV" "--no-spec" "$DLOG" || die "ld: $mode boot failed"
        T1=$(date +%s)
        FITLINE=$(R "grep -m1 'batch fit' $DLOG")
        AVAIL=$(R "grep MemAvailable /proc/meminfo" | awk '{printf "%.2f", $2/1048576}')
        log "(ld) $mode: boot $((T1-T0))s; $FITLINE; settle MemAvailable=${AVAIL} GiB"
        kill_server
    done
    log "(ld) REPORTED (no asserts by design)"
fi

log "D1A RESIDENCY GATE: ALL LEGS COMPLETE"
cleanup
trap - EXIT
