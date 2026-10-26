#!/bin/bash
# Ported into gates/legs/vision/ on 2026-09-04 (release-process §3.0): the Vision-Exp release-battery member.
# Runs ON the GB10 box-side (the tree is the rsync mirror at $T); helper scripts live beside this file.
# Track B inc 5 gate G1 (run ON .33): the fork's REPL builds the same transcript
# as upstream 110afdd for two images across three turns.  Both binaries take the
# same stdin script with -n 1 --temp 0; DS4_REPL_TOKEN_DUMP records the synced
# transcript per turn; compare_repl_dumps.py allows differences only at the
# generated-token positions.  Greedy outputs are kept as information.
set -u
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=/usr/local/cuda/bin:$PATH
STAMP=${STAMP:-$(date +%Y%m%d_%H%M)}
OUT=${OUT:-${DS4_BOX_SCRATCH:?}/vision_cli_${STAMP}}   # the members chain passes OUT explicitly
mkdir -p "$OUT"
FORK=${BINDIR:-${DS4_BOX_TREE:?}}
ORACLE=${DS4_ORACLE_TREE:?}
BASE=${VBASE:-${DS4_VISION_BASE:?}}
ENC=${ENC:-${DS4_VISION_ENCODER:?}}
IMG1=$FORK/tests/test-vectors/flash-vision-exp/images/typical_532x280.png
IMG2=$FORK/tests/test-vectors/flash-vision-exp/images/checker_640x480.png
SCRIPT="$OUT/script.txt"
printf '/read %s\nWhat is in the picture? Answer in one short sentence.\n/read %s\n/quit\n' "$IMG1" "$IMG2" > "$SCRIPT"
echo "cli leg $STAMP out=$OUT" | tee "$OUT/leg.log"
for tree in fork oracle; do
    dir=$FORK; [ $tree = oracle ] && dir=$ORACLE
    echo "=== $tree $(date +%H:%M:%S)" | tee -a "$OUT/leg.log"
    rm -f "$OUT/tokens_$tree.txt"
    ( cd "$dir" && DS4_REPL_TOKEN_DUMP="$OUT/tokens_$tree.txt" \
      ./ds4 -m "$BASE" --vision "$ENC" -c 8192 -n 1 --temp 0 --nothink \
      < "$SCRIPT" > "$OUT/stdout_$tree.txt" 2> "$OUT/stderr_$tree.txt" )
    echo "$tree rc=$? turns=$(grep -c '^turn ' "$OUT/tokens_$tree.txt" 2>/dev/null)" | tee -a "$OUT/leg.log"
    grep -E "ds4: image |image failed|prompt processing failed|failed to add image" "$OUT/stderr_$tree.txt" | tee -a "$OUT/leg.log"
done
echo "=== compare" | tee -a "$OUT/leg.log"
python3 $HERE/compare_repl_dumps.py "$OUT/tokens_fork.txt" "$OUT/tokens_oracle.txt" 2>&1 | tee -a "$OUT/leg.log"
grep -q REPL_TOKENS_PASS "$OUT/leg.log" && echo "CLI_LEG_PASS" | tee -a "$OUT/leg.log" || echo "CLI_LEG_FAIL" | tee -a "$OUT/leg.log"
touch "$OUT/done"
