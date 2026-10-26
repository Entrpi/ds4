#!/bin/bash
# Ported into gates/legs/vision/ on 2026-09-04 (release-process §3.0): the Vision-Exp release-battery member.
# Runs ON the GB10 box-side (the tree is the rsync mirror at $T); helper scripts live beside this file.
# V2 token-identity leg (runs ON .33, detached): the fork's serial image prefill
# vs the upstream 110afdd oracle, SAME probe source built in both trees.
# Oracle twin first (A10: an oracle that disagrees with itself cannot fail the
# port), fork twin second, then the cross comparison.  Fork runs with
# DS4_CUDA_FP8_KV=0 (visual attention needs the F32 compressed-KV primary).
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HERE=$(cd "$(dirname "$0")" && pwd)
T=${BINDIR:-${DS4_BOX_TREE:?}}
O=${DS4_ORACLE_TREE:?}
VBASE=${VBASE:-${DS4_VISION_BASE:?}}
SIDE=${ENC:-${DS4_VISION_ENCODER:?}}
IMG=$T/tests/test-vectors/flash-vision-exp/images
RD=${RD:-${DS4_VISION_ORACLE_RUN:?}}; mkdir -p "$RD"
FORK_ENV=${FORK_ENV-DS4_CUDA_FP8_KV=0}   # empty string = the shipping config (packed FP8 comp KV)
OUT=$RD/leg.log; SENT=$RD/leg.DONE; rm -f "$SENT"; : > "$OUT"
NTOK=${NTOK:-64}; CTX=${CTX:-8192}
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
die(){ log "FAIL: $*"; echo "LEG_FAIL" >> "$OUT"; touch "$SENT"; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH
[ -z "$(pgrep -x ds4-server)" ] || die "a ds4-server is up"

log "== build fork probe in $T"
( cd "$T" && make tests/vision_serial_probe CUDA_ARCH=sm_121 > "$RD/build_fork_probe.log" 2>&1; echo "rc=$?" >> "$RD/build_fork_probe.log" )
tail -1 "$RD/build_fork_probe.log" | grep -q "rc=0" || die "fork probe build ($RD/build_fork_probe.log)"

log "== build oracle probe in $O (same source)"
cp "$T/tests/vision_serial_probe.c" "$O/tests/" || die "copy probe source"
( cd "$O" && cc=$(make -n -B ds4-eval CUDA_ARCH=sm_121 2>/dev/null | grep -E 'ds4_eval\.c' | tail -1) && ln=$(make -n -B ds4-eval CUDA_ARCH=sm_121 2>/dev/null | grep -E ' -o ds4-eval ' | tail -1) \
  && cc2=${cc//ds4_eval.c/tests/vision_serial_probe.c} && cc2=${cc2//-o ds4_eval.o/-I. -o tests/vision_serial_probe.o} \
  && ln2=${ln//ds4_eval.o/tests/vision_serial_probe.o} && ln2=${ln2//-o ds4-eval/-o tests/vision_serial_probe} \
  && { echo "CC2: $cc2"; echo "LN2: $ln2"; } > "$RD/build_oracle_probe.log" && rm -f tests/vision_serial_probe \
  && eval "$cc2" 2>> "$RD/build_oracle_probe.log" && eval "$ln2" 2>> "$RD/build_oracle_probe.log"; echo "rc=$?" >> "$RD/build_oracle_probe.log" )
[ -x "$O/tests/vision_serial_probe" ] || die "oracle probe did not link ($RD/build_oracle_probe.log)"

BEFORE="" ; AFTER="Describe this image in one sentence."
args(){ printf "%s\n%s\n%s\n" "-" "" "Explain how a hash table resolves collisions, in two sentences."; printf "%s\n%s\n%s\n" "-" "" "Name three prime numbers and say why they are prime."; for f in tiny_16x16.png odd_17x9.png typical_532x280.png typical_532x280.jpg portrait_600x900.png checker_640x480.png wide_8to1_2000x250.png large_2400x1800.png; do printf "%s\n%s\n%s\n" "$IMG/$f" "$BEFORE" "$AFTER"; done; }
run(){ local tree=$1 name=$2 env=$3
  log "== run $name ($tree)"
  mapfile -t A < <(args)
  ( cd "$tree" && env ${env:-DS4_PROBE_NOENV=1} timeout 2400 ./tests/vision_serial_probe "$VBASE" "$SIDE" "$NTOK" "$CTX" "${A[@]}" > "$RD/$name.jsonl" 2> "$RD/$name.err"; echo "rc=$?" >> "$RD/$name.err" )
  log "   $name: $(wc -l < "$RD/$name.jsonl") lines, $(tail -1 "$RD/$name.err"); $(grep -aE 'probe:|refus|not implemented|unhandled|error' "$RD/$name.err" | head -3 | cut -c1-160 | tr '\n' '|')"
}
if [ "${SKIP_ORACLE:-0}" = "1" ] && [ -s "$RD/oracle1.jsonl" ] && [ -s "$RD/oracle2.jsonl" ]; then log "== oracle runs reused (SKIP_ORACLE=1)"; else
run "$O" oracle1 ""
run "$O" oracle2 ""
fi
run "$T" fork1 "$FORK_ENV"
run "$T" fork2 "$FORK_ENV"

log "== compare"
python3 $HERE/compare_identity.py "$RD"; exit_compare=$?; : <<'PY' | tee -a "$OUT"
import json, sys, os
rd = sys.argv[1]
def load(n):
    p = os.path.join(rd, n + ".jsonl"); out = {}
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line); out[os.path.basename(d["image"])] = d
    return out
o1, o2, f1, f2 = load("oracle1"), load("oracle2"), load("fork1"), load("fork2")
def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n
imgs = sorted(set(o1) | set(o2) | set(f1) | set(f2))
print(f"{'image':26s} {'o-twin':>7s} {'f-twin':>7s} {'ids':>5s} {'span':>5s} {'block':>5s} {'gen lcp/64':>10s}")
for im in imgs:
    a, b, c, d = o1.get(im), o2.get(im), f1.get(im), f2.get(im)
    def same(x, y, k): return "ok" if x and y and x[k] == y[k] else "DIFF"
    otwin = "n/a" if not (a and b) else ("ok" if a["generated"] == b["generated"] else f"lcp{lcp(a['generated'], b['generated'])}")
    ftwin = "n/a" if not (c and d) else ("ok" if c["generated"] == d["generated"] else f"lcp{lcp(c['generated'], d['generated'])}")
    if not (a and c):
        print(f"{im:26s} {otwin:>7s} {ftwin:>7s}   missing: oracle={bool(a)} fork={bool(c)}"); continue
    ids = same(a, c, "prompt_ids"); sp = same(a, c, "span"); bl = same(a, c, "block_fnv1a64")
    gl = lcp(a["generated"], c["generated"]); g = f"{gl}/{len(a['generated'])}" + ("" if a["generated"] == c["generated"] else "*")
    extra = ""
    if ids == "DIFF":
        i = lcp(a["prompt_ids"], c["prompt_ids"]); extra = f"  first id diff @{i} (span {c['span']['token_start']}..{c['span']['token_start']+c['span']['token_count']})"
    print(f"{im:26s} {otwin:>7s} {ftwin:>7s} {ids:>5s} {sp:>5s} {bl:>5s} {g:>10s}{extra}")
PY
log "== LEG_DONE"; touch "$SENT"
