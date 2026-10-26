#!/bin/bash
# Ported into gates/legs/vision/ on 2026-09-04 (release-process §3.0): the Vision-Exp release-battery member.
# Runs ON the GB10 box-side (the tree is the rsync mirror at $T); helper scripts live beside this file.
# V4 needle-in-image battery (runs ON .33, detached, after the A8 leg): the 24
# synthetic items through the same probe in both trees, replies scored by
# tests/vision_needle_score.py; the PASS band is the oracle's own score.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HERE=$(cd "$(dirname "$0")" && pwd)
T=${BINDIR:-${DS4_BOX_TREE:?}}
O=${DS4_ORACLE_TREE:?}
VBASE=${VBASE:-${DS4_VISION_BASE:?}}
SIDE=${ENC:-${DS4_VISION_ENCODER:?}}
NEEDLE=$T/tests/test-vectors/flash-vision-exp/needle
RD=${RD:-${DS4_BOX_SCRATCH:?}/vision_needle_$(date +%Y%m%d_%H%M)}; mkdir -p "$RD"
OUT=$RD/leg.log; SENT=$RD/leg.DONE; rm -f "$SENT"; : > "$OUT"
NTOK=${NTOK:-48}; CTX=${CTX:-8192}
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
die(){ log "FAIL: $*"; echo "LEG_FAIL" >> "$OUT"; touch "$SENT"; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH
[ -z "$(pgrep -x ds4-server)" ] || die "a ds4-server is up"
log "== build probes"
( cd "$T" && rm -f tests/vision_serial_probe tests/vision_serial_probe.o && make tests/vision_serial_probe CUDA_ARCH=sm_121 > "$RD/build_fork_probe.log" 2>&1; echo "rc=$?" >> "$RD/build_fork_probe.log" )
tail -1 "$RD/build_fork_probe.log" | grep -q "rc=0" || die "fork probe build"
cp "$T/tests/vision_serial_probe.c" "$O/tests/" || die "copy"
( cd "$O" && cc=$(make -n -B ds4-eval CUDA_ARCH=sm_121 2>/dev/null | grep -E 'ds4_eval\.c' | tail -1) && ln=$(make -n -B ds4-eval CUDA_ARCH=sm_121 2>/dev/null | grep -E ' -o ds4-eval ' | tail -1) \
  && cc2=${cc//ds4_eval.c/tests/vision_serial_probe.c} && cc2=${cc2//-o ds4_eval.o/-I. -o tests/vision_serial_probe.o} \
  && ln2=${ln//ds4_eval.o/tests/vision_serial_probe.o} && ln2=${ln2//-o ds4-eval/-o tests/vision_serial_probe} \
  && rm -f tests/vision_serial_probe tests/vision_serial_probe.o && eval "$cc2" 2> "$RD/build_oracle_probe.log" && eval "$ln2" 2>> "$RD/build_oracle_probe.log"; echo "rc=$?" >> "$RD/build_oracle_probe.log" )
[ -x "$O/tests/vision_serial_probe" ] || die "oracle probe did not link"
python3 - "$NEEDLE/manifest.json" "$RD/args.txt" <<'PY'
import json, sys, os
m = json.load(open(sys.argv[1])); items = m["items"] if isinstance(m, dict) else m
nd = os.path.dirname(sys.argv[1])
with open(sys.argv[2], "w") as f:
    for it in items:
        f.write(os.path.join(nd, it["image"]) + "\n"); f.write("\n"); f.write(it["question"] + "\n")
print(len(items), "items")
PY
run(){ local tree=$1 name=$2 env=$3
  log "== run $name ($tree)"
  mapfile -t A < "$RD/args.txt"
  ( cd "$tree" && env $env timeout 3600 ./tests/vision_serial_probe "$VBASE" "$SIDE" "$NTOK" "$CTX" "${A[@]}" > "$RD/$name.jsonl" 2> "$RD/$name.err"; echo "rc=$?" >> "$RD/$name.err" )
  log "   $name: $(wc -l < "$RD/$name.jsonl") lines, $(tail -1 "$RD/$name.err"); $(grep -aE 'probe:|refus|unhandled|error' "$RD/$name.err" | head -3 | cut -c1-140 | tr '\n' '|')"
  python3 - "$RD/$name.jsonl" "$NEEDLE/manifest.json" "$RD/$name.replies.jsonl" <<'PY'
import json, sys, os
m = json.load(open(sys.argv[2])); items = m["items"] if isinstance(m, dict) else m
byimg = {it["image"]: it["id"] for it in items}
n = 0
with open(sys.argv[3], "w") as out:
    for line in open(sys.argv[1]):
        if not line.startswith("{"): continue
        d = json.loads(line); iid = byimg.get(os.path.basename(d["image"]))
        if iid: out.write(json.dumps({"id": iid, "reply": d.get("text", "")}) + "\n"); n += 1
print(n, "replies")
PY
  ( cd "$T" && python3 tests/vision_needle_score.py --manifest "$NEEDLE/manifest.json" "$RD/$name.replies.jsonl" > "$RD/$name.score.txt" 2>&1 || python3 tests/vision_needle_score.py "$RD/$name.replies.jsonl" > "$RD/$name.score.txt" 2>&1 )
  log "   score $name: $(tail -8 "$RD/$name.score.txt" | tr '\n' '|' | cut -c1-400)"
}
run "$O" oracle ""
run "$T" fork "${FORK_ENV-DS4_CUDA_FP8_KV=0}"
log "== LEG_DONE"; touch "$SENT"
