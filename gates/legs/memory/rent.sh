#!/bin/bash
# memory/rent.sh — D3-1: THE resident-vs-host-mapped rent gate.
#
# Measures the whole-system cost of device weight residency (the "rent")
# so the D3-2 release default is CHOSEN FROM DATA (plan sec 12 D3 forbids
# pre-deciding from the old eager-vs-lazy A/B, which never removed device
# residency).  Same binary both sides; the A/B is ENV ONLY:
#   R (resident): shipped defaults — eager pass device-copies the funded
#                 non-expert plan (WEIGHT_ARENA/WEIGHT_SPAN device bytes).
#   M (mapped):   DS4_CUDA_NO_HBM_CACHE=1 DS4_CUDA_NO_FD_CACHE=1 — every
#                 unit stamped HOST_MAPPED, lazy tier fenced off, tier-6
#                 per-range cudaHostRegisterMapped serves everything.
# Aligned artifacts (expert weights) stay device-resident on BOTH legs and
# derived caches are equal — the A/B isolates exactly the span-cache rent.
#
# Two boot shapes, each ABBA R M M R (fresh boots, stable-start guard):
#   SERVING (CTX=32768, SEATS=12): boot wall + cold TTFT, 12k-admit TTFT,
#     N=1 decode 256, N=8 wave (aggregate + per-req).
#   DEEP (CTX=131072, SEATS=4): 120k deep-admit TTFT, deep turn-2 decode,
#     pressure = 3-wave beside the resident deep bank.
# Laws honored: per-request timings.decode_tok_s only (never the window
# gauge); --no-spec --no-mtp (accept-rate variance busts tight bands);
# stable-start settle >= 90 GiB before every boot; subshells return
# sentinels and the PARENT validates; kill BY PID; bank plan PINNED by
# requesting SEATS well under both legs' funding (fit caps at the request:
# n_eff = min(n, max_seq)) and ASSERTED equal; MemAvailable trajectories
# are NOT compared across legs (mapped pins file-backed pages by design).
# Engagement is PROVEN per leg: mapped boots must show funded 0/N +
# mapped-policy N + populated 0 and END serving with zero device
# WEIGHT_ARENA/WEIGHT_SPAN bytes; resident boots must show funded N/N.
#
# This gate MEASURES; it does not band.  PASS = every assert + every leg
# served.  The verdict table is the deliverable (the default change, if
# mapped/lazy wins, is a USER checkpoint).
#
# Usage: bash gates/legs/memory/rent.sh   (from the repo root, Mac side)
# Env: HOST (${DS4_BOX33:?}) TEST_TREE (DS4_BOX_TREE) PORT (18031)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HOST=${HOST:-${DS4_BOX33:?}}
TEST_TREE=${TEST_TREE:-${DS4_BOX_TREE:?}}
PORT=${PORT:-18031}
SERV_CTX=32768;  SERV_SEATS=12
DEEP_CTX=131072; DEEP_SEATS=4; DEEP_TOK=120000
RES=/tmp/rent_gate_results.csv
REPORT=/tmp/rent_gate_report.txt
SLOG=/tmp/rentgate_srv.log        # on the box
SPID=""

log(){ echo "[$(date +%H:%M:%S)] rent_gate: $*"; }
die(){ log "FAIL: $*"; cleanup; exit 1; }
R(){ ssh -o ConnectTimeout=10 "$HOST" "$@"; }
num_ok(){ echo "$2" | grep -qE '^[0-9]+(\.[0-9]+)?$' || die "$1 not numeric ($2)"; }

kill_server(){
    [ -n "$SPID" ] && R "kill $SPID" 2>/dev/null
    SPID=""
    R 'p=$(pgrep -x ds4-server); [ -n "$p" ] && kill $p; sleep 2; p=$(pgrep -x ds4-server); [ -n "$p" ] && kill -9 $p; exit 0' 2>/dev/null
}
cleanup(){ kill_server; }
trap cleanup EXIT

avail_mb(){ R "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo"; }
settle(){   # stable-start guard (D2 close amendment 6cd4792)
    for i in $(seq 60); do
        local a; a=$(avail_mb)
        echo "$a" | grep -qE '^[0-9]+$' && [ "$a" -ge 90000 ] && return 0
        sleep 5
    done
    return 1
}

# ---- boot one leg; echoes boot wall seconds (poll-granular) ------------
BOOT_WALL=""
boot_leg(){ # $1 leg-env  $2 ctx  $3 seats
    settle || die "memory did not settle before boot (stable-start guard)"
    local t0 t1 n=0
    t0=$(date +%s)
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
    t1=$(date +%s)
    BOOT_WALL=$((t1 - t0))
    SPID=$(R "pgrep -x ds4-server | head -1")
    [ -n "$SPID" ] || die "no server pid after listening"
}

# ---- per-leg engagement + parity asserts -------------------------------
FIT_REF=""; VMM_REF=""
assert_shape(){ # $1 leg(R|M)  $2 seats  $3 shape-tag (fresh refs per shape)
    local mat fitseq vmmsig
    mat=$(R "grep -m1 'materialize base:' $SLOG" || true)
    [ -n "$mat" ] || die "($3/$1) no materialize line in boot log"
    if [ "$1" = M ]; then
        echo "$mat" | grep -qE 'funded 0/[0-9]+' || die "($3/M) eager pass funded units: $mat"
        echo "$mat" | grep -qE 'mapped-policy [1-9]' || die "($3/M) mapped-policy 0: $mat"
        echo "$mat" | grep -q 'populated 0 (0.00 GiB)' || die "($3/M) populated != 0: $mat"
    else
        echo "$mat" | grep -qE 'funded ([0-9]+)/\1[^0-9]' || die "($3/R) partial funding: $mat"
        echo "$mat" | grep -qE 'funded 0/' && die "($3/R) resident leg funded nothing: $mat"
        echo "$mat" | grep -q 'mapped-policy 0' || die "($3/R) resident leg has mapped-policy units: $mat"
    fi
    # 'batch fit:' prints only when the budget CLAMPS below the request;
    # the always-printed authority is the batch-ctx ready line.
    fitseq=$(R "grep -m1 'persistent batch ctx ready' $SLOG | grep -oE 'max_seq=[0-9]+' | cut -d= -f2" || true)
    [ "$fitseq" = "$2" ] || die "($3/$1) seat pin broken: max_seq=$fitseq want $2 (raise SEATS margin or headroom)"
    # vmm plan signature: the geometry between page= and budget= must be
    # byte-equal across legs of a shape (budget itself is pinned by env)
    vmmsig=$(R "grep -m1 'batch vmm:' $SLOG | sed -e 's/.*page=/page=/' -e 's/budget=.*//'" || true)
    [ -n "$vmmsig" ] || die "($3/$1) no batch vmm line"
    if [ -z "$VMM_REF" ]; then VMM_REF="$vmmsig"; FIT_REF="$fitseq"
    else
        [ "$vmmsig" = "$VMM_REF" ] || die "($3/$1) vmm plan diverged: '$vmmsig' vs '$VMM_REF'"
    fi
}
assert_mapped_zero_device(){ # $1 shape-tag — post-serving: NO_FD held
    local v
    v=$(R "curl -s -m 20 localhost:$PORT/metrics" | python3 -c '
import sys,re
tot=0
for l in sys.stdin:
    m=re.match(r"ds4_memory_bytes\{domain=\"unified_device\",class=\"weight_(arena|span)\",state=\"allocated\"\} (\d+)",l)
    if m: tot+=int(m.group(2))
print(tot)')
    num_ok "($1/M) device-span-bytes" "$v"
    [ "$v" = 0 ] || die "($1/M) mapped leg allocated $v device span/arena bytes mid-serving (lazy tier engaged?)"
}

# ---- measurement primitives (all echo value-or-sentinel) ---------------
CHAT="localhost:$PORT/v1/chat/completions"
admit_ttft(){ # $1 tokens $2 tag -> ttft seconds
    R "cd $TEST_TREE && python3 gates/lib/needle_prompt.py $1 /tmp/rent_p.json 0.35 '$2' >/dev/null 2>&1" \
        || { echo PROMPTFAIL; return 1; }
    local out
    out=$(R "cd $TEST_TREE && timeout 1200 python3 gates/lib/sse_probe_client.py /tmp/rent_p.json http://127.0.0.1:$PORT/v1/chat/completions ADMIT /tmp/rent_admit.json 2>&1")
    echo "$out" | grep -m1 -oE 'ttft=[0-9.]+' | cut -d= -f2 | grep . || { echo ADMITFAIL; return 1; }
}
decode_toks(){ # $1 prompt -> decode_tok_s of a 256-token request
    local out
    out=$(R "curl -s -m 900 $CHAT -H 'Content-Type: application/json' \
        -d '{\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":256}'")
    echo "$out" | grep -q finish_reason || { echo REQFAIL; return 1; }
    echo "$out" | python3 -c "import json,sys; print(json.load(sys.stdin)['timings']['decode_tok_s'])"
}
TOPICS=(x "a lighthouse keeper" "a desert caravan" "a clockmaker" "a tidal island" \
          "a night train" "an apiarist" "a glassblower" "a cartographer")
wave(){ # $1 n  -> "aggregate_tok_s mean_per_req_tok_s" or sentinel
    local n=$1 i t0 t1 f
    rm -f /tmp/rentwave_*.json
    t0=$(R "date +%s.%N")
    for i in $(seq "$n"); do
        R "curl -s -m 900 $CHAT -H 'Content-Type: application/json' \
            -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Write a 400-word story about ${TOPICS[$i]}.\"}],\"max_tokens\":256}'" \
            > "/tmp/rentwave_$i.json" &
    done
    wait
    t1=$(R "date +%s.%N")
    for i in $(seq "$n"); do
        grep -q finish_reason "/tmp/rentwave_$i.json" || { echo WAVEFAIL; return 1; }
    done
    python3 - "$t0" "$t1" "$n" /tmp/rentwave_*.json <<'EOF' || echo WAVEPARSE
import json, sys
t0, t1, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
per = [json.load(open(f))["timings"]["decode_tok_s"] for f in sys.argv[4:]]
print("%.2f %.2f" % (n * 256 / (t1 - t0), sum(per) / len(per)))
EOF
}
deep_turn2(){ # -> decode_tok_s of a 256-token decode at depth
    R "cat > /tmp/rent_turn2.py" <<'EOF'
import json, sys
d = json.load(open("/tmp/rent_p.json")); r = json.load(open("/tmp/rent_admit.json"))
d["messages"] += [{"role": "assistant", "content": r.get("text") or ""},
                  {"role": "user", "content": "Now write a detailed summary of the main technical topics covered in the archive above. Prose only, no lists."}]
d["max_tokens"] = 256
json.dump(d, open("/tmp/rent_pgen.json", "w"))
EOF
    R "python3 /tmp/rent_turn2.py" || { echo TURN2FAIL; return 1; }
    local out
    out=$(R "curl -s -m 900 $CHAT -H 'Content-Type: application/json' -d @/tmp/rent_pgen.json")
    echo "$out" | grep -q finish_reason || { echo DEEPREQFAIL; return 1; }
    echo "$out" | python3 -c "import json,sys; print(json.load(sys.stdin)['timings']['decode_tok_s'])"
}
record(){ # $1 shape $2 leg $3 pos $4 metric $5 value
    num_ok "$1/$2/$4" "$5"
    echo "$1,$2,$3,$4,$5" >> "$RES"
    log "  $1/$2#$3 $4 = $5"
}
memnote(){ # $1 shape $2 leg $3 pos — recorded, never compared across legs
    local rss pin
    rss=$(R "grep VmRSS /proc/$SPID/status | awk '{print \$2}'" || echo 0)
    pin=$(R "curl -s -m 20 localhost:$PORT/metrics | grep -m1 'domain=\"pinned_host\",class=\"weight_host_pin\",state=\"allocated\"' | awk '{print \$2}'" || echo 0)
    echo "$1,$2,$3,vmrss_kb,${rss:-0}" >> "$RES"
    echo "$1,$2,$3,host_pin_bytes,${pin:-0}" >> "$RES"
}

# ---- one leg of a shape -------------------------------------------------
MAPPED_ENV='DS4_CUDA_NO_HBM_CACHE=1 DS4_CUDA_NO_FD_CACHE=1'
serving_leg(){ # $1 leg(R|M) $2 pos
    local env=""; [ "$1" = M ] && env=$MAPPED_ENV
    log "SERVING leg $1 (#$2) booting"
    boot_leg "$env" $SERV_CTX $SERV_SEATS
    assert_shape "$1" $SERV_SEATS serving
    record serving "$1" "$2" boot_wall_s "$BOOT_WALL"
    local v
    v=$(admit_ttft 64 "RENT-COLD-$1$2");   record serving "$1" "$2" cold_ttft_s "$v"
    v=$(decode_toks "Warm up with a 200-word note on rivers."); num_ok warmup "$v"   # warmup, unrecorded
    v=$(admit_ttft 12000 "RENT-P12K-$1$2"); record serving "$1" "$2" p12k_ttft_s "$v"
    v=$(decode_toks "Write a 400-word story about a ledger."); record serving "$1" "$2" n1_decode_toks "$v"
    v=$(wave 8) || die "(serving/$1) wave failed: $v"
    record serving "$1" "$2" n8_aggregate_toks "${v%% *}"
    record serving "$1" "$2" n8_perreq_toks   "${v##* }"
    [ "$1" = M ] && assert_mapped_zero_device serving
    memnote serving "$1" "$2"
    kill_server
}
deep_leg(){ # $1 leg(R|M) $2 pos
    local env=""; [ "$1" = M ] && env=$MAPPED_ENV
    log "DEEP leg $1 (#$2) booting"
    boot_leg "$env" $DEEP_CTX $DEEP_SEATS
    assert_shape "$1" $DEEP_SEATS deep
    local v
    v=$(decode_toks "Warm up with a 200-word note on tides."); num_ok warmup "$v"
    v=$(admit_ttft $DEEP_TOK "RENT-DEEP-$1$2"); record deep "$1" "$2" deep_admit_ttft_s "$v"
    v=$(deep_turn2);                            record deep "$1" "$2" deep_decode_toks "$v"
    v=$(wave 3) || die "(deep/$1) pressure wave failed: $v"
    record deep "$1" "$2" pressure3_aggregate_toks "${v%% *}"
    [ "$1" = M ] && assert_mapped_zero_device deep
    memnote deep "$1" "$2"
    kill_server
}

# ---- run: ABBA R M M R per shape ---------------------------------------
: > "$RES"
log "shape 1/2: SERVING (ctx=$SERV_CTX seats=$SERV_SEATS), ABBA R M M R"
FIT_REF=""; VMM_REF=""
serving_leg R 1; serving_leg M 1; serving_leg M 2; serving_leg R 2

log "shape 2/2: DEEP (ctx=$DEEP_CTX seats=$DEEP_SEATS tok=$DEEP_TOK), ABBA R M M R"
FIT_REF=""; VMM_REF=""
deep_leg R 1; deep_leg M 1; deep_leg M 2; deep_leg R 2

# ---- verdict table ------------------------------------------------------
python3 - "$RES" <<'EOF' > "$REPORT" || die "report render"
import sys, collections
rows = [l.strip().split(",") for l in open(sys.argv[1]) if l.strip()]
m = collections.defaultdict(list)
for shape, leg, pos, metric, val in rows:
    m[(shape, metric, leg)].append(float(val))
print("RENT GATE VERDICT TABLE (R=resident eager, M=host-mapped; means of 2)")
print("%-9s %-24s %10s %10s %8s" % ("shape", "metric", "R", "M", "M vs R"))
lower_better = {"boot_wall_s", "cold_ttft_s", "p12k_ttft_s", "deep_admit_ttft_s"}
for (shape, metric) in sorted({(s, mt) for (s, mt, _) in m}):
    if metric in ("vmrss_kb", "host_pin_bytes"):
        continue
    r = m.get((shape, metric, "R")); mm = m.get((shape, metric, "M"))
    if not r or not mm:
        continue
    ra, ma = sum(r) / len(r), sum(mm) / len(mm)
    d = (ma - ra) / ra * 100 if ra else 0.0
    good = (d <= 0) if metric in lower_better else (d >= 0)
    print("%-9s %-24s %10.2f %10.2f %+7.1f%% %s"
          % (shape, metric, ra, ma, d, "(M better)" if good and abs(d) > 1 else ""))
print("\nmemory notes (NOT cross-leg comparable by design):")
for (shape, metric) in sorted({(s, mt) for (s, mt, _) in m if mt in ("vmrss_kb", "host_pin_bytes")}):
    for leg in ("R", "M"):
        v = m.get((shape, metric, leg))
        if v: print("  %s %s %s: %s" % (shape, leg, metric, ["%.0f" % x for x in v]))
EOF
cat "$REPORT"
log "RENT GATE: ALL LEGS SERVED, ALL ASSERTS PASS — verdict table above"
log "results: $RES  report: $REPORT (Mac side)"
