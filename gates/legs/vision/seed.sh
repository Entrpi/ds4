#!/bin/bash
# Ported into gates/legs/vision/ on 2026-09-04 (release-process §3.0): the Vision-Exp release-battery member.
# Runs ON the GB10 box-side (the tree is the rsync mirror at $T); helper scripts live beside this file.
# inc 7 seed-logit twin (runs ON .33 after vision/api.sh): the server's
# admission seed logits (DS4_CONT_DUMP_SEED_LOGITS) for the 8 identity prompts
# vs the cont probe's serial-lane dump (CONT_DUMP_SERIAL_LOGITS) for the same
# 8 prompts; compare on the Mac with compare_seed_logits.py (numpy).
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HERE=$(cd "$(dirname "$0")" && pwd); export RB_HERE=$HERE
T=${BINDIR:-${DS4_BOX_TREE:?}}
VBASE=${VBASE:-${DS4_VISION_BASE:?}}
SIDE=${ENC:-${DS4_VISION_ENCODER:?}}
IMG=$T/tests/test-vectors/flash-vision-exp/images
RD=${RD:-${DS4_BOX_SCRATCH:?}/vision_apiseed_$(date +%Y%m%d_%H%M)}; mkdir -p "$RD"
OUT=$RD/leg.log; SENT=$RD/leg.DONE; rm -f "$SENT"; : > "$OUT"
rm -f "$RD/server_seed.bin" "$RD/serial_seed.bin"   # the server APPENDS its seed dump: a re-run into the same RD doubled the records (09-04 resumed battery)
PORT=${PORT:-8090}; CTX=${CTX:-8192}
export API_BASE=http://127.0.0.1:$PORT RD IMG
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
stop_server(){ for x in $(pgrep -x ds4-server); do kill "$x" 2>/dev/null; done; sleep 3; for x in $(pgrep -x ds4-server); do kill -9 "$x" 2>/dev/null; done; }
die(){ log "FAIL: $*"; echo "SEED_LEG_FAIL" >> "$OUT"; stop_server; touch "$SENT"; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH
[ -z "$(pgrep -x ds4-server)" ] || die "a ds4-server is up"
AFTER="Describe this image in one sentence."; AFTERJ="Describe this photograph in one sentence."
log "== cont probe, 8 prompts (serial dump)"
( cd "$T" && env DS4_PROBE_NOENV=1 CONT_DUMP_SERIAL_LOGITS="$RD/serial_seed.bin" timeout 2400 ./tests/vision_cont_probe "$VBASE" "$SIDE" "$CTX" 24 \
  "-" "" "Explain how a hash table resolves collisions, in two sentences." \
  "$IMG/tiny_16x16.png" "" "$AFTER" "$IMG/odd_17x9.png" "" "$AFTER" "$IMG/typical_532x280.png" "" "$AFTER" "$IMG/typical_532x280.jpg" "" "$AFTERJ" \
  "$IMG/portrait_600x900.png" "" "$AFTER" "$IMG/checker_640x480.png" "" "$AFTER" "$IMG/wide_8to1_2000x250.png" "" "$AFTER" \
  > "$RD/probe.out" 2> "$RD/probe.err"; echo "rc=$?" >> "$RD/probe.err" )
log "   probe: $(tail -1 "$RD/probe.err") $(grep -E '^(PASS|FAIL|CONT_LEG)' "$RD/probe.out" | tr '\n' ' ' | cut -c1-300)"
log "== boot ds4-server (seed dump)"
( cd "$T" && env DS4_CONT_DUMP_SEED_LOGITS="$RD/server_seed.bin" ./ds4-server -m "$VBASE" --vision "$SIDE" -c "$CTX" --host 127.0.0.1 --port "$PORT" --no-spec --no-mtp > "$RD/server.log" 2>&1 < /dev/null & )
for i in $(seq 1 180); do sleep 5; code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null); [ "$code" = "200" ] && break; done
[ "$code" = "200" ] || die "server not ready"
log "== identity once (8 prompts: no large_2400x1800, no primes text)"
python3 - <<'PY' | tee -a "$OUT"
import os, sys
sys.argv = ["x"]; sys.path.insert(0, os.environ["RB_HERE"])
import api_client as c
c.RD = os.environ["RD"]; c.IMG = os.environ["IMG"]; c.BASE = os.environ["API_BASE"]
c.TEXT_PROMPTS = c.TEXT_PROMPTS[:1]; c.IMAGES = [i for i in c.IMAGES if i != "large_2400x1800.png"]
c.cmd_identity("seed_api", "batched")
PY
log "== stop server"; stop_server; log "   servers left: $(pgrep -cx ds4-server) records: server=$(( $(stat -c %s "$RD/server_seed.bin" 2>/dev/null || echo 0) / 517128 )) serial=$(( $(stat -c %s "$RD/serial_seed.bin" 2>/dev/null || echo 0) / 517128 ))"
log "== seed twin compare (box-side, numpy-free): argmax equal per length-distinct pair, else a serial near-tie inside the band"
python3 "$HERE/compare_seed_logits.py" "$RD/serial_seed.bin" "$RD/server_seed.bin" 129280 0.34 2>&1 | tee "$RD/compare.txt" | tail -2 | sed 's/^/   /' | tee -a "$OUT"
log "== SEED_LEG_DONE"; touch "$SENT"
