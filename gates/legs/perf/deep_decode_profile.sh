#!/bin/bash
# perf/deep_decode_profile.sh — nsys kernel profile of the deep-decode window.
#
# Boots ds4-server on $R under `nsys launch` (collection idle, ~zero
# overhead), cold-admits a $TOK-token needle prompt, then profiles ONLY the
# warm decode: nsys start → re-send the same prompt with max_tokens=$GEN
# (warm in-place admit + $GEN-token decode) → nsys stop. Prints the CUDA
# kernel summary for the window. Laws honored: env prefix BEFORE nsys,
# --cuda-graph-trace=node, drive AFTER collection start, fresh export.
#
# Usage:
#   EXTRA_ENV="DS4_CUDA_FP8_KV=1 DS4_CUDA_FP4_INDEX=1" \
#     bash gates/legs/perf/deep_decode_profile.sh fp8fp4
#   EXTRA_ENV="" bash gates/legs/perf/deep_decode_profile.sh f32
# Env: R (${DS4_BOX33:?}) BINDIR PORT (8000) TUNNEL_PORT (18000)
#      CTX (262144) TOK (240000) GEN (256) GGUF (${DS4_GGUF_DIR:?})
# NOTE: kills any running ds4-server on $R; leaves the box server-free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
CTX=${CTX:-262144}
TOK=${TOK:-240000}
GEN=${GEN:-256}
GGUF=${GGUF:-${DS4_GGUF_DIR:?}}
LABEL=${1:?usage: perf/deep_decode_profile.sh <label>}
EXTRA_ENV=${EXTRA_ENV:-}
BASE=$GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MTP=$GGUF/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DRAFTER=$GGUF/DSpark-drafter-Q2K-Q8.gguf
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/tmp/ddprof_${LABEL}
RREP=/tmp/ddprof_${LABEL}.nsys-rep
SRV=/tmp/ddprof_srv.log
URL="http://127.0.0.1:$TUNNEL/v1/chat/completions"
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] ddprof($LABEL): $*"; }
fail(){ log "FAIL: $*"; exit 1; }

# pkill -x ONLY (a -f pattern like 'nsys launch' self-matches the remote
# shell running this very kill sequence and aborts it mid-way). The nsys
# launch wrapper exits on its own when the app dies.
log "killing old ds4-server on $R"
ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; sleep 1; rm -f /tmp/ds4.lock $RREP ${RREP%.nsys-rep}.sqlite; exit 0"
sleep 3
ssh "$R" "pgrep -x ds4-server >/dev/null && exit 1; exit 0" || fail "old ds4-server still alive after kill"

# Boot-fit hygiene (2026-07-14 law, same as deep_ctx_gate): wait for
# MemAvailable to stabilize after the kill — a transient low reading
# shrinks the batch fit AND silences the live budget refresh.
ssh "$R" "prev=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo); i=0
  while [ \$i -lt 24 ]; do sleep 5
    cur=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
    d=\$((cur - prev)); [ \$d -lt 0 ] && d=\$((-d))
    [ \$d -lt 512000 ] && { echo \"mem stable: \$((cur/1048576)) GiB\"; exit 0; }
    prev=\$cur; i=\$((i+1)); done
  echo 'mem NOT stable after 120s (continuing)'"

# Deep-ctx boot shape mirrors deep_ctx_gate: LAZY serial graph (no
# DS4_SESSION_LAZY_GRAPH=0 — the eager serial graph at deep ctx eats the
# memory the deep bank needs and the admit 503s on fit).
log "boot under nsys launch (ship cont env + '$EXTRA_ENV', ctx=$CTX)"
ssh "$R" ": > $SRV; cd $BINDIR; env $EXTRA_ENV \
    DS4_CUDA_NO_HBM_CACHE=1 \
    DS4_BATCH_FIT_HEADROOM_MB=6272 DS4_SERVER_COALESCE_MAX=8 \
    DS4_CONT_PREFILL_CHUNK=2048 DS4_CONT_MTP_MODE=2 DS4_CONT_DSPARK=1 \
    DS4_DSPARK_MODEL=$DRAFTER DS4_CONT_CAPTURE=1 DS4_SERVER_DEFAULT_TEMP=0 \
    setsid nohup nsys launch --session-new=ddprof -t cuda --cuda-graph-trace=node \
    ./ds4-server -m $BASE --mtp $MTP --cuda -c $CTX --port $PORT \
    > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
  ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || {
    sleep 3
    ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
      fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" 2>/dev/null | tr '\n' ' ')"
  }
  sleep 10; n=$((n+10)); [ $n -ge 1200 ] && fail "boot timeout"
done
curl -s -m 5 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1 || {
  ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true
  sleep 2
  curl -s -m 10 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null || fail "tunnel :$TUNNEL unreachable"
}
log "boot up"

log "building $TOK-token prompt + cold admit (long)"
python3 "$DS4_GATES/lib/needle_prompt.py" "$TOK" "$OUT/p.json" 0.35 "7311-PROF-ALFA" >/dev/null || fail "prompt build"
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/p.json" "$URL" ADMIT "$OUT/admit_result.json" > "$OUT/admit.out" 2>&1 || fail "admit: $(tail -2 "$OUT/admit.out")"
grep -q 'ttft=' "$OUT/admit.out" || fail "admit produced no ttft: $(tail -2 "$OUT/admit.out")"
log "admit done: $(grep -m1 'ttft=' "$OUT/admit.out")"

# Turn-2 shape (same as deep_ctx_gate stage 2): append the ACTUAL answer +
# a long-form user turn so the profiled window is a real $GEN-token decode,
# not an 8-token needle answer that EOSes early.
python3 - "$OUT/p.json" "$OUT/admit_result.json" "$OUT/pgen.json" "$GEN" <<'EOF' || fail "pgen build"
import json, sys
d = json.load(open(sys.argv[1]))
r = json.load(open(sys.argv[2]))
ans = r.get("text") or ""
d["messages"] = d["messages"] + [
    {"role": "assistant", "content": ans},
    {"role": "user", "content": "Now write a detailed summary of the main "
     "technical topics covered in the archive above. Prose only, no lists."},
]
d["max_tokens"] = int(sys.argv[4])
json.dump(d, open(sys.argv[3], "w"))
EOF

log "nsys start + profiled warm decode ($GEN tokens)"
ssh "$R" "nsys start --session=ddprof -o ${RREP%.nsys-rep}" || fail "nsys start"
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/pgen.json" "$URL" PROF > "$OUT/prof.out" 2>&1 || fail "profiled decode: $(tail -2 "$OUT/prof.out")"
ssh "$R" "nsys stop --session=ddprof" || fail "nsys stop"
log "profiled decode: $(grep -m1 'ttft=' "$OUT/prof.out")"
grep -q 'warm admit' <(ssh "$R" "cat $SRV") || log "WARN: no 'warm admit' line — profiled window may include re-prefill"

log "kernel summary (fresh export)"
ssh "$R" "nsys stats --force-export=true --report cuda_gpu_kern_sum $RREP 2>/dev/null" | tee "$OUT/kern_sum.txt" | head -30

ssh "$R" "pkill -x ds4-server; exit 0"
log "done — artifacts: $OUT + $R:$RREP (server killed)"
