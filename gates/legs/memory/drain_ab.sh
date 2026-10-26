#!/bin/bash
# memory/drain_ab.sh — ONE leg of the 378855/86 memory-drain A/B.
# Runs ENTIRELY ON .33 (box-local; no tunnel). Replicates the reporter's
# protocol: -c 32768 default boot, plain sequential no-tools chat,
# ~6 s cadence, 10-minute measured window, sampling MemAvailable +
# VmRSS/VmHWM + /metrics every 15 s.
#
# Usage (on .33):
#   bash gates/legs/memory/drain_ab.sh <LEG> <BIN> <BINDIR> <NONCE> [EXTRA_ENV...]
#   e.g. bash gates/legs/memory/drain_ab.sh v050   ./ds4-server ~/code/ds4-v050base  v0562-memdrain
#        bash gates/legs/memory/drain_ab.sh v0562  ./ds4-server $DS4_BOX_TREE    v0562-memdrain
#        bash gates/legs/memory/drain_ab.sh v0562nocache ./ds4-server $DS4_BOX_TREE v0562-memdrain DS4_CUDA_NO_HBM_CACHE=1
#
# v0.5.0 baseline tree prep (from the Mac, once):
#   cd ~/code/ds4 && git archive v0.5.0 | ssh ${DS4_BOX33:?} 'mkdir -p ~/code/ds4-v050base && tar -x -C ~/code/ds4-v050base'
#   ssh ${DS4_BOX33:?} 'cd ~/code/ds4-v050base && make -j6 cuda-spark'
#   (verify sm_121a: /usr/local/cuda/bin/cuobjdump --dump-elf ds4-server | grep -m1 sm_121)
#
# Outputs: /tmp/memdrain_<LEG>_srv.log, /tmp/memdrain_<LEG>_samples.log,
#          verdict lines on stdout (MEMDRAIN_LEG_* prefixed).
set -u
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

LEG="${1:?leg name}"; BIN="${2:?server binary}"; BINDIR="${3:?bin dir}"
NONCE="${4:?box lock nonce}"; shift 4
GGUF="${DS4_TEXT_BASE:?}"
PORT=18011
SRVLOG="/tmp/memdrain_${LEG}_srv.log"
SAMPLES="/tmp/memdrain_${LEG}_samples.log"
WINDOW_S="${MEMDRAIN_WINDOW_S:-600}"
CADENCE_S=6

# --- box ownership (two-session law) ---
if [ ! -d /tmp/ds4_box_lock ]; then echo "MEMDRAIN_LEG_ABORT no box lock"; exit 2; fi
HELD=$(cat /tmp/ds4_box_lock/nonce 2>/dev/null || echo NONE)
if [ "$HELD" != "$NONCE" ]; then echo "MEMDRAIN_LEG_ABORT lock nonce '$HELD' != '$NONCE'"; exit 2; fi
if [ "$(pgrep -cx ds4-server)" != "0" ]; then echo "MEMDRAIN_LEG_ABORT ds4-server already running"; exit 2; fi

# --- boot (default shape = shipped defaults; only ADMIT_DEBUG added) ---
cd "$BINDIR" || { echo "MEMDRAIN_LEG_ABORT bad bindir"; exit 2; }
env DS4_ADMIT_DEBUG=1 "$@" setsid nohup "$BIN" -m "$GGUF" -c 32768 --port "$PORT" \
    > "$SRVLOG" 2>&1 < /dev/null &
sleep 3
PID=$(pgrep -x ds4-server | head -1)
if [ -z "$PID" ]; then echo "MEMDRAIN_LEG_ABORT boot died early"; tail -5 "$SRVLOG"; exit 2; fi
echo "MEMDRAIN_LEG_BOOT leg=$LEG pid=$PID"

UP=0
for _ in $(seq 1 180); do
    if curl -s --max-time 3 "localhost:$PORT/v1/models" | grep -q '"object"'; then UP=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
    sleep 5
done
if [ "$UP" != "1" ]; then echo "MEMDRAIN_LEG_ABORT server never became healthy"; tail -20 "$SRVLOG"; kill "$PID" 2>/dev/null; exit 2; fi

req() {  # req <id> — unique FIRST token per request (warm-reuse law)
    curl -s --max-time 90 "localhost:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"ds4\",\"max_tokens\":120,\"messages\":[{\"role\":\"user\",\"content\":\"Ref$1: pick a physical process (different each time) and explain it in about 80 words.\"}]}" \
      > /dev/null 2>&1
}

# --- warmup (outside the window) ---
for i in 1 2 3 4 5; do req "warm$i"; done
sleep 5

sample() {
    TS=$(date +%s)
    MA=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    RSS=$(awk '/VmRSS/{print $2}' "/proc/$PID/status" 2>/dev/null || echo 0)
    HWM=$(awk '/VmHWM/{print $2}' "/proc/$PID/status" 2>/dev/null || echo 0)
    MET=$(curl -s --max-time 4 "localhost:$PORT/metrics" 2>/dev/null \
          | grep -E 'admit|trim|reclaim|banks|mem_|vmm' | grep -v '^#' | tr '\n' ';' | head -c 800)
    echo "$TS MemAvailable_kB=$MA VmRSS_kB=$RSS VmHWM_kB=$HWM $MET"
}

: > "$SAMPLES"
sample >> "$SAMPLES"
T0=$(date +%s)
N=0
while :; do
    NOW=$(date +%s); ELAPSED=$((NOW - T0))
    [ "$ELAPSED" -ge "$WINDOW_S" ] && break
    RS=$(date +%s)
    N=$((N + 1))
    req "$N"
    RE=$(date +%s); DUR=$((RE - RS))
    [ $((RE - T0)) -ge "$WINDOW_S" ] || [ "$DUR" -ge "$CADENCE_S" ] || sleep $((CADENCE_S - DUR))
    if [ $((N % 3)) -eq 0 ]; then sample >> "$SAMPLES"; fi
done
sample >> "$SAMPLES"

# --- teardown BY PID ---
FIRST_MA=$(head -1 "$SAMPLES" | grep -o 'MemAvailable_kB=[0-9]*' | cut -d= -f2)
LAST_MA=$(tail -1 "$SAMPLES" | grep -o 'MemAvailable_kB=[0-9]*' | cut -d= -f2)
HWM_END=$(awk '/VmHWM/{print $2}' "/proc/$PID/status" 2>/dev/null || echo 0)
kill "$PID" 2>/dev/null
for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null

# --- verdict ---
DRAIN_KB=$((FIRST_MA - LAST_MA))
echo "MEMDRAIN_LEG_RESULT leg=$LEG requests=$N window_s=$WINDOW_S drain_kB=$DRAIN_KB drain_GB=$(awk "BEGIN{printf \"%.2f\", $DRAIN_KB/1048576}") vmhwm_kB=$HWM_END"
echo "MEMDRAIN_LEG_BOOTLINES:"
grep -iE 'bank|budget|cache prepared|floor|max_seq|fit' "$SRVLOG" | head -12
echo "MEMDRAIN_LEG_COUNTERS:"
REJ=$(grep -c 'admit rejected on comp-cache budget' "$SRVLOG" || true)
TRIM=$(grep -ci 'trim' "$SRVLOG" || true)
FLOOR=$(grep -ci 'floor' "$SRVLOG" || true)
echo "rejects=$REJ trim_lines=$TRIM floor_lines=$FLOOR"
echo "MEMDRAIN_LEG_DONE leg=$LEG"
