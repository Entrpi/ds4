#!/bin/bash
# gates/legs/memory/kv_crossmode.sh — disk-KV checkpoints across storage modes
# (v0.2.4 flip risk: packed FP8/FP4 primaries are default ON, so checkpoints
# written by an F32-primary server WILL be loaded by packed-primary boots in
# the wild, and vice versa. Recon said the restore path is F32-normalized;
# this gate proves it end to end.)
#
# Six fresh boots on the gate host, ship config (launch defaults), disk KV
# armed with --kv-disk-dir:
#   1. A_store : F32 primaries (explicit =0s), fresh DIR1, prompt P, gen 64.
#   2. L1_f32  : F32 primaries,    DIR1 — disk hit, gen 64.
#   3. L1_pk   : packed (default), DIR1 — disk hit (packed restore of an
#                F32-written checkpoint), gen 64.
#   4. B_store : packed primaries, fresh DIR2, prompt P, gen 64.
#   5. L2_pk   : packed (default), DIR2 — disk hit (packed restore of a
#                packed-written checkpoint), gen 64.
#   6. L2_f32  : F32 primaries,    DIR2 — disk hit, gen 64 (cross-mode read
#                of a packed-written checkpoint — the save-expansion proof).
# PASS = all four load legs hit ("kv cache hit text"); all four restored
# continuations are byte-identical to each other (packed restores must
# byte-match the F32 restores — the v0.3 exact-restore proof: restore-time
# mirror re-encode is bitwise idempotent since the dsv4_pow2_ceil_scale
# fix); the two STORE continuations are byte-identical (fresh-compute
# cross-mode identity); zero corrupt/failed/discarded/refused kv lines
# anywhere.
#
# ORACLE NOTE: restored continuations are compared ONLY against other
# restored continuations, never against the fresh-compute text — a restore
# recomputes the post-boundary prompt tail at a different prefill width, and
# width-dependent FP noise legitimately flips near-ties (see memory
# phase0-prefill-verify-width-noise). Restore-vs-restore holds the tail
# width fixed, so byte-identity is the correct bar there.
#
# Laws respected: pkill -x only; each boot kills any server on the host.
# Boots run DS4_SERVER_CONTINUOUS=0: disk-KV checkpoints are a
# serial-session feature — the cold/continued/shutdown store sites and the
# text-prefix load site all live on the serial job path, so a cont-admitted
# request never touches them (first run of this gate proved that the hard
# way: 8.2K-token cont request, zero kv lines).
#
# Env overrides (defaults):
#   KV_GATE_HOST (${DS4_BOX33:?})  BINDIR (${DS4_BOX_TREE:?})
#   CTX (16384)  PORT (8000)  PROMPT_BYTES (26000)
set -u
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
R=${KV_GATE_HOST:-${DS4_BOX33:?}}
RT=${BINDIR:-${DS4_BOX_TREE:?}}
CTX=${CTX:-16384}
PORT=${PORT:-8000}
PROMPT_BYTES=${PROMPT_BYTES:-26000}
F32ENV="DS4_CUDA_FP8_KV=0 DS4_CUDA_FP4_INDEX=0"
OUT=${KV_GATE_OUT:-/tmp/kv_crossmode_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { log "KV-CROSSMODE GATE: FAIL — $*"; exit 1; }

# Deterministic prompt: fixed corpus prefix + a question forcing a
# continuation that depends on the whole prefix.
python3 - "$PROMPT_BYTES" > "$OUT/payload.json" << 'PYEOF'
import json, sys
n = int(sys.argv[1])
text = open("speed-bench/promessi_sposi.txt", encoding="utf-8", errors="replace").read()[:n]
msg = text + "\n\nSummarize the mood of the final paragraph above in one sentence."
# model MUST be the non-thinking alias: think mode force-overrides the
# request temperature to the sampled default (ds4_server.c serial decode
# loop), so greedy byte-compares are impossible on a thinking model name.
print(json.dumps({"model": "deepseek-chat", "messages": [{"role": "user", "content": msg}],
                  "max_tokens": 64, "temperature": 0, "seed": 7}))
PYEOF
scp -q "$OUT/payload.json" "$R:/tmp/kvx_payload.json" || fail "payload scp"

boot_and_ask() { # $1=tag $2=env-string $3=dir $4=fresh(0/1)
  local tag=$1 envs=$2 dir=$3 fresh=$4
  # NOTE the (…&) subshell wrap: a bare `… & exit 0` keeps the launcher ssh
  # attached and it never returns (bit us twice, .15 and .33).
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server 2>/dev/null; rm -f /tmp/ds4.lock; \
            $([ "$fresh" = 1 ] && echo "rm -rf $dir;") mkdir -p $dir; \
            cd $RT && (env DS4_SERVER_CONTINUOUS=0 DS4_SERVER_DEFAULT_TEMP=0 $envs setsid nohup ./ds4-server -c $CTX --port $PORT \
              --kv-disk-dir $dir --kv-disk-space-mb 6000 \
              > /tmp/kvx_${tag}.log 2>&1 < /dev/null &) ; exit 0"
  local n=0
  until ssh "$R" "curl -sf -o /dev/null -m 3 http://127.0.0.1:$PORT/v1/models" 2>/dev/null; do
    n=$((n+1)); [ $n -ge 90 ] && fail "$tag boot timeout"
    ssh "$R" "pgrep -x ds4-server >/dev/null" || fail "$tag BOOT-DIED (check /tmp/kvx_${tag}.log)"
    sleep 2
  done
  ssh "$R" "curl -s -m 300 http://127.0.0.1:$PORT/v1/chat/completions \
              -H 'Content-Type: application/json' -d @/tmp/kvx_payload.json" \
    > "$OUT/resp_${tag}.json" || fail "$tag completion request"
  python3 - "$OUT/resp_${tag}.json" << 'PYEOF' > "$OUT/text_${tag}.txt" || exit 1
import json, sys
r = json.load(open(sys.argv[1]))
c = r["choices"][0]["message"]["content"]
sys.stdout.write(c)
sys.stderr.write("cached=%d prompt=%d\n" % (
    r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0),
    r["usage"]["prompt_tokens"]))
PYEOF
  [ -s "$OUT/text_${tag}.txt" ] || fail "$tag empty completion (resp: $(head -c 200 "$OUT/resp_${tag}.json"))"
}

stop_and_check_store() { # $1=tag
  local tag=$1
  ssh "$R" "pkill -x ds4-server"          # SIGTERM: graceful, flushes shutdown store
  local n=0
  while ssh "$R" "pgrep -x ds4-server >/dev/null"; do
    n=$((n+1)); [ $n -ge 45 ] && { ssh "$R" "pkill -9 -x ds4-server"; break; }
    sleep 2
  done
  ssh "$R" "grep -a 'kv cache stored' /tmp/kvx_${tag}.log" >> "$OUT/kv_lines.txt" \
    || fail "$tag: no 'kv cache stored' line (log tail: $(ssh $R "grep -a 'kv cache' /tmp/kvx_${tag}.log | tail -3"))"
  log "$tag: stored OK"
}

check_load_hit() { # $1=tag
  local tag=$1
  ssh "$R" "grep -a 'kv cache hit text' /tmp/kvx_${tag}.log" >> "$OUT/kv_lines.txt" \
    || fail "$tag: no disk hit (kv lines: $(ssh $R "grep -a 'kv cache' /tmp/kvx_${tag}.log | tail -3"))"
  log "$tag: disk hit OK"
}

log "gate start host=$R ctx=$CTX out=$OUT"
boot_and_ask A_store "$F32ENV" ${DS4_BOX_SCRATCH:?}/kvx_dir1 1
stop_and_check_store A_store
boot_and_ask L1_f32  "$F32ENV" ${DS4_BOX_SCRATCH:?}/kvx_dir1 0
check_load_hit L1_f32
boot_and_ask L1_pk   ""        ${DS4_BOX_SCRATCH:?}/kvx_dir1 0
check_load_hit L1_pk
boot_and_ask B_store ""        ${DS4_BOX_SCRATCH:?}/kvx_dir2 1
stop_and_check_store B_store
boot_and_ask L2_pk   ""        ${DS4_BOX_SCRATCH:?}/kvx_dir2 0
check_load_hit L2_pk
boot_and_ask L2_f32  "$F32ENV" ${DS4_BOX_SCRATCH:?}/kvx_dir2 0
check_load_hit L2_f32
ssh "$R" "pkill -x ds4-server; sleep 1; rm -f /tmp/ds4.lock" 2>/dev/null

for t in A_store L1_f32 L1_pk B_store L2_pk L2_f32; do
  for bad in "load failed" "discarded corrupt" "store failed" "failed to discard" "restore refused"; do
    ssh "$R" "grep -a \"kv cache.*$bad\" /tmp/kvx_${t}.log" >/dev/null 2>&1 \
      && fail "$t: '$bad' in log"
  done
done

ok=1
cmp -s "$OUT/text_A_store.txt" "$OUT/text_B_store.txt" \
  || { ok=0; log "MISMATCH: fresh-compute A_store vs B_store"; }
# Restore-vs-restore oracle, all four legs against one reference: the two
# packed restores must byte-match the two F32 restores (exact-restore proof).
for t in L1_pk L2_pk L2_f32; do
  cmp -s "$OUT/text_L1_f32.txt" "$OUT/text_$t.txt" \
    || { ok=0; log "MISMATCH: restored L1_f32 vs restored $t"; }
done
if [ $ok = 1 ]; then
  log "fresh identity and 4-way restored identity (incl. packed restores) hold"
  # v3 receipt: DIR2 was written by a packed-primary server (packed-native
  # rows) and should be ~2.8-3x smaller than DIR1's F32-row payloads.
  log "payload sizes: DIR1(F32-written)=$(ssh $R "du -sh ${DS4_BOX_SCRATCH:?}/kvx_dir1 | cut -f1") DIR2(packed-written)=$(ssh $R "du -sh ${DS4_BOX_SCRATCH:?}/kvx_dir2 | cut -f1")"
else
  for t in A_store L1_f32 L1_pk B_store L2_pk L2_f32; do
    log "--- $t: $(cat "$OUT/text_$t.txt")"
  done
  fail "continuation mismatch across storage modes"
fi
log "KV-CROSSMODE GATE: ALL PASS (artifacts: $OUT)"
