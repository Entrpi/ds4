#!/bin/bash
# Ported into gates/legs/vision/ on 2026-09-04 (release-process §3.0): the Vision-Exp release-battery member.
# Runs ON the GB10 box-side (the tree is the rsync mirror at $T); helper scripts live beside this file.
# Track B inc 5 gate G2 (run ON .33): ds4-agent view_image end to end with the
# disk session cache ON.  Run A: view a screenshot of hello.c, name the bug,
# edit the file, /save.  Run B: /switch to the saved session (the image record
# restores; the worker rebuilds its image list as reference spans) and ask a
# follow-up: its sync must extend the restored checkpoint (cached>0), never
# re-prefill the image block.  Run C: view the image, then force a compaction
# with an oversized read; the retained image block is rebased into the tail
# (trace "compaction retained"); a follow-up after compaction still sees it.
set -u
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=/usr/local/cuda/bin:$PATH
STAMP=${STAMP:-$(date +%Y%m%d_%H%M)}
OUT=${OUT:-${DS4_BOX_SCRATCH:?}/vision_agent_${STAMP}}   # the members chain passes OUT explicitly
mkdir -p "$OUT"
FORK=${BINDIR:-${DS4_BOX_TREE:?}}
BASE=${VBASE:-${DS4_VISION_BASE:?}}
ENC=${ENC:-${DS4_VISION_ENCODER:?}}
E2E=${DS4_BOX_SCRATCH:?}/vision_e2e
DRIVE=$HERE/agent_pty_drive.py
AGENT="$FORK/ds4-agent -m $BASE --vision $ENC -n 600 --temp 0"
LOG="$OUT/leg.log"
echo "agent leg $STAMP out=$OUT" | tee "$LOG"
[ -f "$E2E/hello.c.orig" ] || cp "$E2E/hello.c" "$E2E/hello.c.orig"
cp -f "$E2E/hello.c.orig" "$E2E/hello.c"

echo "=== run A (view + edit + save) $(date +%H:%M:%S)" | tee -a "$LOG"
cat > "$OUT/stepsA.json" <<JSON
[{"send": "Use view_image to look at $E2E/screenshot.png. It is a screenshot of a source file. Say which file and line it shows and what the bug is. Then read $E2E/hello.c and fix that bug with the edit tool. Do not use bash.", "quiet": 8, "timeout": 1500, "wait_active": true},
 {"send": "/save", "quiet": 8, "timeout": 900, "wait_active": true},
 {"send": "/quit", "expect": "saved session [0-9a-f]{8}|save failed", "quiet": 3, "timeout": 600, "wait_active": false}]
JSON
python3 "$DRIVE" "$OUT/ptyA.log" "$OUT/stepsA.json" -- $AGENT -c 16384 --trace "$OUT/traceA.log" 2>&1 | tee -a "$LOG"
echo "--- hello.c after run A:" | tee -a "$LOG"; cat "$E2E/hello.c" | tee -a "$LOG"
grep -q "wrold" "$E2E/hello.c" && echo "A_EDIT_FAIL (typo still present)" | tee -a "$LOG" || echo "A_EDIT_OK" | tee -a "$LOG"
SHA=$(tr -d '\r' < "$OUT/ptyA.log" | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | grep -o "saved session [0-9a-f]\{8\}" | tail -1 | awk '{print $3}')
echo "saved sha8=${SHA:-NONE}" | tee -a "$LOG"
grep -E "view_image|prefill sync done|prefill tool_round|kv stored|session saved|compact" "$OUT/traceA.log" | grep -v "token index" | head -20 | tee -a "$LOG"
tr -d '\r' < "$OUT/ptyA.log" | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' | grep -o "ds4: image prefill[^|]*" | head -2 | tee -a "$LOG"
ls -la --time-style=+%H:%M ~/.ds4/kvcache/${SHA}*.kv 2>/dev/null | tee -a "$LOG"

echo "=== run B (switch to the saved session, follow-up) $(date +%H:%M:%S)" | tee -a "$LOG"
cat > "$OUT/stepsB.json" <<JSON
[{"send": "/switch ${SHA:-00000000}", "quiet": 8, "timeout": 1500, "wait_active": true},
 {"send": "What did the screenshot show, and what did you change in the file? One sentence each.", "expect": "switched to session|switch failed|no such session|not found", "quiet": 8, "timeout": 1200, "wait_active": true},
 {"send": "/quit", "quiet": 8, "timeout": 300, "wait_active": true}]
JSON
python3 "$DRIVE" "$OUT/ptyB.log" "$OUT/stepsB.json" -- $AGENT -c 16384 --trace "$OUT/traceB.log" 2>&1 | tee -a "$LOG"
tr -d '\r' < "$OUT/ptyB.log" | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | grep -E "switched to session|has no embedding|error|failed" | head -5 | tee -a "$LOG"
grep -E "prefill sync done|prefill tool_round|kv hit|switch|has no embedding" "$OUT/traceB.log" | grep -v "token index" | head -10 | tee -a "$LOG"
tr -d '\r' < "$OUT/ptyB.log" | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' | grep -o "ds4: image prefill[^|]*" | head -2 | tee -a "$LOG"

echo "=== run C (view, forced compaction, follow-up) $(date +%H:%M:%S)" | tee -a "$LOG"
BIG=$FORK/ds4_kvstore.c
echo "big file: $(wc -c < $BIG) bytes" | tee -a "$LOG"
cat > "$OUT/stepsC.json" <<JSON
[{"send": "Use view_image to look at $E2E/screenshot.png and say in one sentence what it shows. Then read the whole file $BIG with the read tool (whole=true) and report its first function name.", "quiet": 8, "timeout": 1800, "wait_active": true},
 {"send": "Which file name was shown in the screenshot you viewed? Answer with the file name only.", "quiet": 8, "timeout": 1200, "wait_active": true},
 {"send": "/quit", "quiet": 8, "timeout": 300, "wait_active": true}]
JSON
python3 "$DRIVE" "$OUT/ptyC.log" "$OUT/stepsC.json" -- $AGENT -c 12288 --trace "$OUT/traceC.log" 2>&1 | tee -a "$LOG"
grep -E "compact|prefill sync done|prefill tool_round|has no embedding" "$OUT/traceC.log" | grep -v "token index" | head -20 | tee -a "$LOG"
tr -d '\r' < "$OUT/ptyC.log" | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' | grep -o "ds4: image prefill[^|]*" | head -2 | tee -a "$LOG"
tr -d '\r' < "$OUT/ptyC.log" | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | grep -E "COMPACTING|hello.c|error" | head -8 | tee -a "$LOG"
echo "=== run D (text-only control: the battery path with no image) $(date +%H:%M:%S)" | tee -a "$LOG"
cat > "$OUT/stepsD.json" <<JSON
[{"send": "Read $E2E/hello.c and tell me in one sentence what the program prints. Do not use bash.", "quiet": 8, "timeout": 1200, "wait_active": true},
 {"send": "/quit", "quiet": 8, "timeout": 300, "wait_active": true}]
JSON
python3 "$DRIVE" "$OUT/ptyD.log" "$OUT/stepsD.json" -- $AGENT -c 16384 --trace "$OUT/traceD.log" 2>&1 | tee -a "$LOG"
grep -E "prefill tool_round|view_image|image" "$OUT/traceD.log" | grep -v "token index" | head -6 | tee -a "$LOG"
tr -d '\r' < "$OUT/ptyD.log" | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' | grep -o "ds4: image prefill[^|]*" | head -2 | tee -a "$LOG"
echo "=== done $(date +%H:%M:%S)" | tee -a "$LOG"
touch "$OUT/done"
