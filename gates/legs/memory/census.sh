#!/bin/bash
# memory/census.sh — memgov D0a close gate (spec: local/docs/v057/
# d0a_scoping_2026-08-08.md §3 D0a-close).  Mac-driven, runs .33 legs under
# the box-ownership lock and Mac legs locally.
#
# WHAT IT PROVES
#   (a) DECISION ORACLE: fit/floor/admission/reject log families identical
#       vs the pre-D0a control build (~/code/ds4-v0563base @ b9c97ad) on
#       three boot shapes: eager default, DS4_CUDA_NO_HBM_CACHE=1 (the
#       tripwire class exercised), and the zeroconf deep leg.  "Identical"
#       is byte-diff on the normalized family lines: timestamps stripped
#       and the free-memory INPUT masked (fit-jitter law: consecutive
#       boots differ in MemAvailable, not in the formula) -- with the raw
#       inputs asserted inside a drift band and the DERIVED decision
#       (max_seq) asserted equal, one retry per pair to absorb a
#       boundary-crossing jitter.
#   (b) RECONCILIATION at settle: census registry totals vs the allocator
#       books (WEIGHT_HOST_PIN vs the registered-mapping boot line;
#       ENGINE_OTHER live == 0) vs the system delta (MemAvailable
#       pre-boot minus settle vs census live totals, stated tolerance,
#       sources logged).  An unstable starting memory state ABORTS (plan
#       §13), never warns-and-continues.  Substrate tripwire bit-identity:
#       census totals line == fit line, same boot.
#   (c) CPU/Metal: local `make ds4_server_cpu.o` compiles and the Metal
#       units binary runs the server suite green (census reads report
#       unsupported, never zero).
#   (d) NO-HOT-PATH: every census/injection call in ds4_cuda.cu sits in an
#       allow-listed control-plane function (alloc/free/ensure/trim/init/
#       cleanup/import/artifact); ABBA perf close vs the control build
#       (short decode + prefill, ABBA order, <5% with both clocks logged).
#
# Usage: bash gates/legs/memory/census.sh          (from the repo root)
#   env: HOST, CTX (default 32768), DEEP_CTX (default 240000), TOL_GIB.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HOST=${HOST:-${DS4_BOX33:?}}
TEST_TREE=${DS4_BOX_TREE:?}
# Control tree: pre-stage tip, re-based per stage (D0a: ds4-v0563base @
# b9c97ad; D0b: ds4-cebaff7base @ cebaff7 -- rider #37).  Overridable.
CTRL_TREE=${CTRL_TREE:-'~/code/ds4-cebaff7base'}
CTX=${CTX:-32768}
DEEP_CTX=${DEEP_CTX:-240000}
TOL_GIB=${TOL_GIB:-3}
PORT=8000
TS=$(date +%s); NONCE="d0aclose-$TS-$$"
LOCK=/tmp/ds4_box_lock
R() { ssh -o ConnectTimeout=10 "$HOST" "$@"; }
SPID=""
log() { echo "[$(date +%H:%M:%S)] $*"; }
cleanup() {
    [ -n "$SPID" ] && R "kill $SPID" 2>/dev/null
    R 'ps -eo pid,comm | awk '"'"'$2=="ds4-server"{print $1}'"'"' | xargs -r kill' 2>/dev/null
    R "rm -rf $LOCK" 2>/dev/null
}
die() { log "FAIL: $*"; cleanup; exit 1; }

# ---- box ownership ----
R "ps -eo comm | grep -q '^ds4-server' && echo BUSY || echo IDLE" | grep -q IDLE \
    || die "a ds4-server is already running on $HOST"
R "mkdir $LOCK 2>/dev/null && echo $NONCE > $LOCK/nonce" \
    || die "box lock held: $(R cat $LOCK/nonce 2>/dev/null || echo unknown)"
trap cleanup EXIT
log "lock acquired ($NONCE)"

# ---- (c) CPU + Metal legs, local ----
log "(c) CPU object + Metal server units (local)"
( cd "$(dirname "$0")/../../.." && make ds4_server_cpu.o >/tmp/d0aclose_cpu.log 2>&1 ) \
    || die "CPU object build failed (see /tmp/d0aclose_cpu.log)"
( cd "$(dirname "$0")/../../.." && make ds4_test >/dev/null 2>&1 && ./ds4_test --server >/tmp/d0aclose_metal.log 2>&1 ) \
    || die "Metal server units failed (see /tmp/d0aclose_metal.log)"
grep -q "server: OK" /tmp/d0aclose_metal.log || die "Metal units did not report server: OK"
log "(c) PASS"

# ---- (d1) no-hot-path source proof ----
log "(d1) no-hot-path allow-list scan (ds4_cuda.cu)"
python3 - "$(dirname "$0")/../../../ds4_cuda.cu" <<'EOF' || exit 1
import re, sys
src = open(sys.argv[1]).read().splitlines()
# Enclosing-function scan: registry/injection calls may appear ONLY inside
# the EXPLICIT reviewed owner set below (inventoried 2026-08-11 from the
# D0a tree; every entry adjudicated control-plane: funnel, arenas, sticky
# growers -- capture-refused by the sticky-scratch law -- load/teardown,
# and ONE documented kernel-wrapper exception: the norm_q8 sticky grow in
# ds4_gpu_rms_norm_weight_rows_f16_tensor).  A note in ANY other function
# fails the gate until reviewed and added here.  No wildcards: a wildcard
# that matches a kernel wrapper is a hole, not a convenience.
allow = {
    "cuda_mem_note_alloc", "cuda_mem_note_free", "cuda_trim_inject_fire",
    "ds4_gpu_mem_census_read",
    "cuda_model_arena_alloc", "cuda_model_copy_chunked",
    "cuda_model_range_populate_device_copy", "cuda_model_range_release_all",
    "cuda_model_stage_pool_alloc",
    # memgov D1b-2 (reviewed 2026-08-13): the stage-pin free-note moved
    # into the extracted release helper -- same accounting the alloc's
    # grow path always carried; control-plane only (eager-pass end +
    # first-miss grow), never the token path.
    "cuda_model_stage_pool_release",
    "cuda_moe_iq2_derepack_scratch", "cuda_moe_q2k_derepack_scratch",
    "fp8_predecode_scratch_alloc", "rr_scratch_ensure", "tt_scratch_ensure",
    "cuda_tmp_alloc",
    "cuda_pinned_file_note_register", "cuda_pinned_file_note_unregister",
    "cuda_q8_f16_cache_release_all", "cuda_q8_f16_ptr", "cuda_q8_f32_ptr",
    "cuda_vmm_arena_alloc", "cuda_vmm_arenas_release_all",
    "ds4_gpu_build_derived_artifacts", "ds4_gpu_import_model_ipc_manifest",
    "ds4_gpu_set_model_map",
    "import_vmm_allocation", "import_vmm_derived_allocation",
    "ds4_gpu_init", "ds4_gpu_cleanup",
    "ds4_gpu_decode_scalars_init", "ds4_gpu_decode_scalars_cleanup",
    "ds4_gpu_decode_layer_scalars_init", "ds4_gpu_decode_layer_scalars_cleanup",
    "ds4_gpu_decode_row_scalars_init",
    "ds4_gpu_tensor_alloc", "ds4_gpu_tensor_alloc_managed",
    "ds4_gpu_tensor_ensure", "ds4_gpu_tensor_trim", "ds4_gpu_tensor_free",
    "ds4_gpu_rms_norm_weight_rows_f16_tensor",
}
call = re.compile(r'\b(cuda_mem_note_(alloc|free)|cuda_trim_inject_fire|ds4_gpu_mem_census_read)\s*\(')
fndef = re.compile(r'^(?:static\s+|extern\s+"C"\s+)?[A-Za-z_][\w\s\*:<>,]*?\b([A-Za-z_]\w*)\s*\([^;]*$')
cur, depth, bad = "?", 0, []
for i, ln in enumerate(src, 1):
    if depth == 0:
        m = fndef.match(ln)
        if m:
            cur = m.group(1)
    depth += ln.count('{') - ln.count('}')
    if depth < 0: depth = 0
    if call.search(ln) and cur not in allow:
        bad.append((i, cur, ln.strip()[:90]))
if bad:
    for b in bad: print("HOT-PATH LEAK: line %d in %s: %s" % b)
    sys.exit(1)
print("no-hot-path scan clean (%d call sites, all in the reviewed owner set)" %
      sum(1 for ln in src if call.search(ln)))
EOF
[ $? -eq 0 ] || die "no-hot-path scan found registry calls outside the allow-list"
log "(d1) PASS"

# ---- remote helpers ----
BOOT_FLAGS=""   # extra ds4-server flags (ABBA sets --no-spec --no-mtp)
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
stable_avail() { # ABORT-on-unstable (plan §13): two reads within 150M
    local a b
    for t in 1 2 3 4 5; do
        a=$(avail_mb); sleep 5; b=$(avail_mb)
        local d=$((a - b)); [ ${d#-} -lt 150 ] && { echo "$b"; return 0; }
    done
    return 1
}
# Decision families, normalized: strip the timestamp/pid prefix, mask the
# free-memory inputs (fit-jitter law) -- formulas and verdict shapes must
# byte-match; the masked inputs are band-checked separately.
extract_decisions() { # $1 remote log  $2 local out
    # memgov D2-3 amendment (08-14): the session-graph-fit refusal's
    # "only N MiB allocatable" operand is the SAME free-input class as
    # free= (observed: T=1520 C=1567 on deep boots a minute apart on the
    # stale box, decisions identical) -- masked; the refusal itself still
    # byte-compares.  need ~N MiB is graph-shape-derived and stays exact.
    # max_seq is masked HERE (it is free-input-derived, the fit-jitter law);
    # its equality is asserted separately with a one-retry jitter policy.
    # The batch-vmm ledger's banks/budget/plan/capacity are ALL free-input
    # derived (plan = banks x per-bank; banks from the free-derived fit):
    # masked here, with the bank COUNT adjudicated separately like max_seq.
    # The D1a-4 range-plan porcelain ends in "refused N" and matches the
    # 'refus' family pattern -- excluded by construction (the control
    # cannot print it; d1a residency-gate l1 first-run finding).
    # memgov D2-2b amendment (08-14): the fit CLAMP line's PRESENCE is
    # free-input-derived (prints only when n < requested); a fresh-box
    # T/C pair whose free sits AT the requested boundary can straddle
    # the print threshold (residency l1 finding, T=31 clamped vs C=32
    # unclamped).  Excluded from the byte-diff; fit_shape byte-compares
    # its shape-derived operands when both sides print, and the
    # presence-jitter rule in oracle_leg adjudicates asymmetry.
    R "grep -E 'batch fit|batch vmm|mem floor|boot ledger|refus|reject|no graph fits|admission|serial fit' $1" \
        | grep -v 'ds4: range plan ' \
        | grep -v 'batch fit: free=' \
        | sed -E 's/^[0-9]{4} [0-9:]{8} //; s/free=[0-9.]+ GiB/free=# GiB/g; s/MemAvailable[= ][0-9.]+/MemAvailable=#/g; s/capacity [0-9.]+ GiB/capacity # GiB/g; s/max_seq [0-9]+/max_seq #/g; s/x [0-9]+ banks/x # banks/g; s/budget=[0-9.]+ GiB/budget=# GiB/g; s/\[plan [0-9.]+, capacity [0-9.]+\]/[plan #, capacity #]/g; s/only [0-9]+ MiB allocatable/only # MiB allocatable/g; s/ deficit=[0-9.]+ MiB avail=[0-9.]+ MiB//g' \
        > "$2"
}
# Fit-jitter adjudication for a free-derived integer decision: equal is
# clean; a one-step difference whose DIRECTION matches the free inputs is
# the documented jitter shape (D0a-1 law); anything else is real.
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
fit_free_gib() { R "grep 'batch fit' $1 | head -1" | grep -o 'free=[0-9.]*' | cut -d= -f2; }
fit_max_seq()  { R "grep 'batch fit' $1 | head -1" | grep -o 'max_seq [0-9]*' | awk '{print $2}'; }
fit_requested() { R "grep -m1 'batch fit: free=' $1" | grep -o '(requested [0-9]*' | grep -o '[0-9]*'; }
# The clamp line with free-input operands masked: what remains is
# shape-derived (per_bank, headroom, substrate) and must byte-match.
fit_shape() { R "grep -m1 'batch fit: free=' $1" | sed -E 's/free=[0-9.]+ GiB/free=# GiB/; s/max_seq [0-9]+/max_seq #/'; }
vmm_banks()    { R "grep 'batch vmm' $1 | head -1" | grep -o 'x [0-9]* banks' | grep -o '[0-9]*'; }

# ---- (a) decision oracle: three legs x (test, control) ----
oracle_leg() { # $1 name  $2 env  $3 ctx
    local name=$1 env=$2 ctx=$3 attempt
    for attempt in 1 2; do
        log "(a) $name attempt $attempt: test boot"
        stable_avail >/dev/null || die "$name: MemAvailable unstable pre-boot (ABORT, plan §13)"
        boot_tree "$TEST_TREE" "$env" "$ctx" "/tmp/d0aclose_${name}_T.log" || die "$name test boot failed"
        kill_server
        log "(a) $name attempt $attempt: control boot"
        stable_avail >/dev/null || die "$name: MemAvailable unstable between pair (ABORT)"
        boot_tree "$CTRL_TREE" "$env" "$ctx" "/tmp/d0aclose_${name}_C.log" || die "$name control boot failed"
        kill_server
        extract_decisions "/tmp/d0aclose_${name}_T.log" "/tmp/d0aclose_${name}_T.dec"
        extract_decisions "/tmp/d0aclose_${name}_C.log" "/tmp/d0aclose_${name}_C.dec"
        # The test tree ADDS census lines the control cannot have; decision
        # families exclude them by grep construction.  Byte-diff:
        if ! diff -q "/tmp/d0aclose_${name}_T.dec" "/tmp/d0aclose_${name}_C.dec" >/dev/null; then
            diff "/tmp/d0aclose_${name}_T.dec" "/tmp/d0aclose_${name}_C.dec" | head -12
            die "$name: decision families differ (normalized byte-diff)"
        fi
        local ft fc mt mc bt bc mv bv req pres
        ft=$(fit_free_gib "/tmp/d0aclose_${name}_T.log"); fc=$(fit_free_gib "/tmp/d0aclose_${name}_C.log")
        mt=$(fit_max_seq  "/tmp/d0aclose_${name}_T.log"); mc=$(fit_max_seq  "/tmp/d0aclose_${name}_C.log")
        bt=$(vmm_banks    "/tmp/d0aclose_${name}_T.log"); bc=$(vmm_banks    "/tmp/d0aclose_${name}_C.log")
        log "(a) $name inputs: free T=$ft C=$fc GiB; max_seq T=$mt C=$mc; vmm_banks T=${bt:--} C=${bc:--}"
        # D2-2b v2 (residency l2 finding): THE FORMULA IS THE ORACLE.
        # Both trees draw fit-time free from the same noisy boot-to-boot
        # distribution (observed 21.8-23.5 GiB at the basemtp shape;
        # mmap-mode MemAvailable accounting), so cross-side output
        # equality was only ever a proxy that breaks when input noise
        # reaches one bank.  fit_adjudicate asserts: each side's printed
        # decision equals the formula on its OWN inputs (violation is
        # REAL -> die); shape operands byte-match; free inputs in the
        # 1.0 GiB band; cross distance bounded by band/per_bank+1 with
        # direction tracking the inputs; each side's vmm banks equal its
        # own funded plan.  Input-noise verdicts retry the pair.
        if [ -n "$ft" ] && [ -n "$fc" ] && \
           [ "$(fit_shape "/tmp/d0aclose_${name}_T.log")" != "$(fit_shape "/tmp/d0aclose_${name}_C.log")" ]; then
            fit_shape "/tmp/d0aclose_${name}_T.log"; fit_shape "/tmp/d0aclose_${name}_C.log"
            die "$name: fit-line shape mismatch (per_bank/headroom/substrate)"
        fi
        local verdict
        verdict=$(fit_adjudicate "$(R "grep -m1 'batch fit: free=' /tmp/d0aclose_${name}_T.log")" \
                                 "$(R "grep -m1 'batch fit: free=' /tmp/d0aclose_${name}_C.log")" \
                                 "${bt:--}" "${bc:--}")
        case "$verdict" in
        PASS*)  log "(a) $name PASS ($verdict)"; return 0 ;;
        FAIL*)  die "$name: $verdict" ;;
        *)      log "(a) $name: $verdict -- retrying pair once" ;;
        esac
    done
    die "$name: input noise on both attempts -- box state, not the formula (stable_avail law)"
}
oracle_leg eager  ""                       "$CTX"
# nohbm: the vmm budget is PINNED on both trees (handoff law: pin the plan)
# so the dominant free-derived quantity is deterministic; capacity stays
# masked and the bank count still runs the jitter adjudication.
oracle_leg nohbm  "DS4_CUDA_NO_HBM_CACHE=1 DS4_BATCH_VMM_BUDGET_MB=8000" "$CTX"
oracle_leg deep   ""                       "$DEEP_CTX"

# ---- (b) reconciliation at settle (test tree, eager boot) ----
log "(b) reconciliation boot (test tree)"
PRE_MB=$(stable_avail) || die "reconciliation: MemAvailable unstable pre-boot (ABORT)"
log "(b) pre-boot MemAvailable=${PRE_MB}M (stable)"
boot_tree "$TEST_TREE" "" "$CTX" /tmp/d0aclose_recon.log || die "reconciliation boot failed"
sleep 20   # settle: post-prewarm idle
POST_MB=$(stable_avail) || die "reconciliation: MemAvailable unstable at settle (ABORT)"
R "curl -s -m 30 localhost:$PORT/metrics > /tmp/d0aclose_metrics.txt" || die "metrics curl"
# ENGINE_OTHER settles to zero on server boots (census doc §11 contract).
EO=$(R "grep 'class=\"engine_other\",state=\"allocated\"' /tmp/d0aclose_metrics.txt" | awk '{s+=$NF} END{print s}')
[ "$EO" = "0" ] || die "ENGINE_OTHER allocated != 0 ($EO) -- untagged allocation on a server boot"
FAULTS=$(R "grep '^ds4_memory_census_faults_total' /tmp/d0aclose_metrics.txt" | awk '{print $NF}')
[ "$FAULTS" = "0" ] || die "census faults at settle ($FAULTS)"
# WEIGHT_HOST_PIN == registered-mapping boot line (single funnel, 1:1).
HP=$(R "grep 'domain=\"pinned_host\",class=\"weight_host_pin\",state=\"allocated\"' /tmp/d0aclose_metrics.txt" | awk '{print $NF}')
REG_GIB=$(R "grep -o 'registered [0-9.]* GiB model mapping' /tmp/d0aclose_recon.log | head -1" | grep -o '[0-9.]*' | head -1)
HP_OK=$(python3 -c "print(1 if abs($HP/2**30 - $REG_GIB) < 0.02 else 0)")
[ "$HP_OK" = "1" ] || die "WEIGHT_HOST_PIN ($HP B) != registered mapping (${REG_GIB} GiB)"
# System delta vs census totals, stated tolerance, sources logged.
DEV=$(R "grep 'state=\"allocated\"' /tmp/d0aclose_metrics.txt | grep 'unified_device'" | awk '{s+=$NF} END{print s}')
HOSTB=$(R "grep 'state=\"allocated\"' /tmp/d0aclose_metrics.txt | grep 'pinned_host'" | awk '{s+=$NF} END{print s}')
DELTA_MB=$((PRE_MB - POST_MB))
# Tolerance = max(TOL_GIB, 5% of census): MemAvailable is a kernel
# ESTIMATE (watermark + reclaimable heuristics) and an 86 GiB model file
# streams through page cache during boot -- a flat few-GiB band is
# mis-sized against a ~107 GiB footprint.  The SHARP cross-checks are the
# exact ones above (host_pin==registered 1:1, ENGINE_OTHER==0, faults==0,
# census rows == D0a-0 ground truth); this delta exists to catch GROSS
# drift (a missing class, double counting), not estimate noise.
RECON=$(python3 -c "
census=($DEV+$HOSTB)/2**30; sys=$DELTA_MB/1024.0
tol=max(float($TOL_GIB), 0.05*census)
print('census=%.2f sys_delta=%.2f diff=%.2f tol=%.2f -> %s' % (census, sys, abs(census-sys), tol, 'OK' if abs(census-sys)<tol else 'FAIL'))")
log "(b) reconciliation: $RECON (sources: /metrics census totals; /proc/meminfo MemAvailable ${PRE_MB}M->${POST_MB}M)"
echo "$RECON" | grep -q 'OK$' || die "reconciliation outside tolerance"
# Substrate tripwire bit-identity (census totals line vs fit line).
# D2-2b amendment: an UNCLAMPED recon boot (fresh box funds the full
# request) prints no fit line -- the tripwire is vacuous there, not a
# failure; any clamped boot still asserts it.
SUB_C=$(R "grep 'memory census totals:' /tmp/d0aclose_recon.log | head -1" | grep -o 'substrate_outstanding=[0-9.]*' | cut -d= -f2)
SUB_F=$(R "grep 'batch fit' /tmp/d0aclose_recon.log | head -1" | grep -o 'substrate outstanding=[0-9.]*' | grep -o '[0-9.]*')
if [ -n "$SUB_F" ]; then
    [ "$SUB_C" = "$SUB_F" ] || die "substrate tripwire mismatch census=$SUB_C fit=$SUB_F"
else
    log "(b) substrate tripwire: recon boot unclamped (no fit line) -- vacuous"
fi
log "(b) PASS (ENGINE_OTHER=0 faults=0 host_pin==registered substrate bit-identical)"

# ---- (b2) D5-1 epoch-coherent porcelains (tip-only; (b)'s server still up) ----
# One capture per render, both stamps in VALUES across every porcelain.
# Boot stamps are older than render stamps (the boot lease publish alone
# advances the governor), so cross-surface asserts compare only the
# render-time scrapes of the idle server -- where reads never advance an
# epoch, the property "compare only equal epochs" depends on.
log "(b2) D5-1 epoch coherence (one snapshot per render)"
EPB=$(R "grep -o 'memory census epoch: census=[0-9]* governor=[0-9]*' /tmp/d0aclose_recon.log | head -1")
[ -n "$EPB" ] || die "boot log missing the D5-1 epoch line"
CE_B=$(echo "$EPB" | grep -o 'census=[0-9]*' | cut -d= -f2)
GE_B=$(echo "$EPB" | grep -o 'governor=[0-9]*' | cut -d= -f2)
[ $((CE_B % 2)) -eq 0 ] && [ $((GE_B % 2)) -eq 0 ] \
    || die "boot epoch stamps not even -- a porcelain rendered a torn image (census=$CE_B governor=$GE_B)"
CE_1=$(R "grep '^ds4_memory_census_epoch' /tmp/d0aclose_metrics.txt" | awk '{print $NF}')
GE_1=$(R "grep '^ds4_memory_governor_epoch' /tmp/d0aclose_metrics.txt" | awk '{print $NF}')
[ -n "$CE_1" ] && [ -n "$GE_1" ] || die "metrics missing epoch gauges"
[ $((CE_1 % 2)) -eq 0 ] && [ $((GE_1 % 2)) -eq 0 ] || die "metrics epoch stamps not even (census=$CE_1 governor=$GE_1)"
R "curl -s -m 30 localhost:$PORT/metrics > /tmp/d0aclose_metrics2.txt" || die "metrics re-curl"
CE_2=$(R "grep '^ds4_memory_census_epoch' /tmp/d0aclose_metrics2.txt" | awk '{print $NF}')
GE_2=$(R "grep '^ds4_memory_governor_epoch' /tmp/d0aclose_metrics2.txt" | awk '{print $NF}')
[ "$CE_1" = "$CE_2" ] && [ "$GE_1" = "$GE_2" ] \
    || die "idle epochs moved across scrapes (census $CE_1->$CE_2 governor $GE_1->$GE_2)"
R "curl -s -m 30 -H 'Accept: application/json' localhost:$PORT/v1/stats > /tmp/d0aclose_stats.json" || die "stats json curl"
R "curl -s -m 30 localhost:$PORT/v1/stats > /tmp/d0aclose_stats.txt" || die "stats board curl"
JQ=$(R "python3 -c \"import json;d=json.load(open('/tmp/d0aclose_stats.json'));print(d['memory']['census_epoch'],d['governor']['epoch'],d['cache']['bank_lineage_clock'])\"") \
    || die "stats json missing D5-1 keys (census_epoch / governor epoch / bank_lineage_clock)"
CE_J=$(echo "$JQ" | awk '{print $1}'); GE_J=$(echo "$JQ" | awk '{print $2}'); LC_J=$(echo "$JQ" | awk '{print $3}')
[ "$CE_J" = "$CE_1" ] && [ "$GE_J" = "$GE_1" ] \
    || die "stats JSON epochs != metrics (census $CE_J vs $CE_1, governor $GE_J vs $GE_1)"
CE_T=$(R "grep '^census_epoch:' /tmp/d0aclose_stats.txt" | cut -d: -f2)
GE_T=$(R "grep '^governor_epoch:' /tmp/d0aclose_stats.txt" | cut -d: -f2)
LC_T=$(R "grep '^bank_lineage_clock:' /tmp/d0aclose_stats.txt" | cut -d: -f2)
[ "$CE_T" = "$CE_1" ] && [ "$GE_T" = "$GE_1" ] || die "stats board epochs != metrics (census $CE_T governor $GE_T)"
[ -n "$LC_J" ] && [ "$LC_J" = "$LC_T" ] || die "lineage clock JSON ($LC_J) != board ($LC_T)"
log "(b2) PASS (census_epoch=$CE_1 governor_epoch=$GE_1 lineage_clock=$LC_J; boot even, metrics/json/board agree, idle-stable)"

# ---- (b3) D5-2/3 families (tip-only; (b)'s server still up) ----
# Residency units reconcile against the materialize boot lines: on the
# unclamped eager recon boot populated/ready/import/lazy/failed are
# EXACT (the boot pass is the only ticker before the scrape) while
# mapped-policy is >= (post-boot touches re-tick it per miss).  Failures
# and typed rejections must be all-zero on a healthy idle boot; the
# legacy reject scalars stay present and untyped beside the new family.
log "(b3) D5-2/3 families (residency units/failures, deficit gauge, typed rejections)"
R "curl -s -m 30 localhost:$PORT/metrics" > /tmp/d0aclose_b3_metrics.txt || die "metrics curl (b3)"
R "cat /tmp/d0aclose_recon.log" > /tmp/d0aclose_b3_recon.log || die "recon log fetch (b3)"
B3=$(python3 - <<'PYEOF' 2>&1
import re
m = open('/tmp/d0aclose_b3_metrics.txt').read()
log = open('/tmp/d0aclose_b3_recon.log').read()
def cells(fam):
    out = {}
    for l in m.splitlines():
        if l.startswith(fam + '{'):
            labels, val = l.rsplit(' ', 1)
            out[tuple(re.findall(r'(\w+)="([^"]+)"', labels))] = int(val)
    return out
u = cells('ds4_residency_units')
f = cells('ds4_residency_failures_total')
d = cells('ds4_memory_decision_deficit_bytes')
r = cells('ds4_requests_rejected_total')
assert len(u) == 4 * 8, ('units cardinality', len(u))
assert len(f) == 3 * 2, ('failures cardinality', len(f))
assert len(d) == 5, ('deficit cardinality', len(d))
assert len(r) == 3 * 6, ('rejected cardinality', len(r))
assert all(v == 0 for v in f.values()), ('failures nonzero', f)
assert all(v == 0 for v in r.values()), ('rejections on idle boot', r)
for legacy in ('ds4_cont_admit_rejects_total ', 'ds4_graph_fit_refusals_total '):
    assert any(l.startswith(legacy) for l in m.splitlines()), ('legacy scalar gone', legacy)
lines = re.findall(r'materialize \S+: funded \S+ promote units \(\d+ unfunded\), '
                   r'populated (\d+) \([^)]*\), ready (\d+), import (\d+), '
                   r'mapped-policy (\d+), lazy-policy (\d+), failed (\d+)', log)
assert lines, 'no materialize lines in recon log'
tot = [sum(int(t[i]) for t in lines) for i in range(6)]
def urow(state):
    return sum(v for k, v in u.items() if dict(k)['state'] == state)
assert urow('populated') == tot[0], ('populated', urow('populated'), tot[0])
assert urow('already_ready') == tot[1], ('ready', urow('already_ready'), tot[1])
assert urow('satisfied_import') == tot[2], ('import', urow('satisfied_import'), tot[2])
assert urow('host_mapped_by_policy') >= tot[3], ('mapped', urow('host_mapped_by_policy'), tot[3])
assert urow('lazy_deferred') == tot[4], ('lazy', urow('lazy_deferred'), tot[4])
assert urow('failed') == tot[5] == 0, ('failed', urow('failed'), tot[5])
print('B3OK populated=%d ready=%d import=%d mapped_min=%d lazy=%d' %
      (tot[0], tot[1], tot[2], tot[3], tot[4]))
PYEOF
)
echo "$B3" | grep -q '^B3OK' || { echo "$B3" | tail -4; die "(b3) family asserts failed"; }
log "(b3) PASS ($B3)"

# ---- (d2) ABBA perf close vs control (short decode leg) ----
# Instrument: the measured request's OWN timings.decode_tok_s (exact for
# that request), never the 60s window gauge (run-1 lesson: the gauge read
# 9.85 and 12.8 tok/s for the SAME tree across two boots while the
# per-request path is stable).  Both sides boot --no-spec --no-mtp
# (launch-defaults law: accept-rate variance alone busts a 5% band) and
# pay one warmup request (graph capture + page warm) before measuring.
# D2 stage-close amendment (08-14): every measurement boot gets the
# stable-start guard -- the previous boot's ~110 GiB teardown can still
# be reclaiming when the next launches, and an unsettled boot raced the
# reclaim and was OOM-shot ~20 s in (C1 12:09; the SAME tree booted
# clean 90 s later).  And `die` inside $() only kills the SUBSHELL --
# the parent kept computing with boot-log text as a number -- so the
# subshell now returns sentinel words and the PARENT validates.
abba_settle() { # the teardown's reclaim must finish before a boot
    for i in $(seq 60); do
        [ "$(avail_mb)" -ge 90000 ] && return 0
        sleep 5
    done
    return 1
}
abba_ms_tok() { # $1 tree -> echoes decode_tok_s of a measured 256-token request
    kill_server
    abba_settle || { echo UNSETTLED; return 1; }
    BOOT_FLAGS="--no-spec --no-mtp"
    if ! boot_tree "$1" "" "$CTX" /tmp/d0aclose_abba.log; then
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
abba_num() { # $1 label  $2 value: numeric or die (in the PARENT)
    echo "$2" | grep -qE '^[0-9]+(\.[0-9]+)?$' \
        || die "ABBA $1 measurement invalid ($2)"
}
log "(d2) ABBA decode close (T C C T, per-request timings, plain decode)"
T1=$(abba_ms_tok "$TEST_TREE");  abba_num T1 "$T1"; log "  T1=$T1 tok/s"
C1=$(abba_ms_tok "$CTRL_TREE");  abba_num C1 "$C1"; log "  C1=$C1 tok/s"
C2=$(abba_ms_tok "$CTRL_TREE");  abba_num C2 "$C2"; log "  C2=$C2 tok/s"
T2=$(abba_ms_tok "$TEST_TREE");  abba_num T2 "$T2"; log "  T2=$T2 tok/s"
kill_server
ABBA=$(python3 -c "
t=($T1+$T2)/2; c=($C1+$C2)/2
d=abs(t-c)/c*100
print('test=%.1f ctrl=%.1f delta=%.1f%% -> %s' % (t, c, d, 'OK' if d < 5.0 else 'FAIL'))")
log "(d2) ABBA: $ABBA"
echo "$ABBA" | grep -q 'OK$' || die "ABBA decode delta >= 5%"

cleanup
trap - EXIT
log "D0A CENSUS GATE: ALL LEGS PASS"
