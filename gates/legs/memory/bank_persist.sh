#!/bin/bash
# gates/legs/memory/bank_persist.sh — durable pinned banks gate (v0.3).
#
# Proves the cont bank disk tier end to end on ONE GB10:
#   P2  deep needle trunk A (~$TRUNK_TOK tok) admits cold, retires a warm
#       record; needle answered exactly (pre-persist baseline).
#   P3  graceful shutdown persists the bank (reason=bank-shutdown, KVC file).
#   P5  fresh boot; turn 2 of A triggers a DISK-RESTORED warm admit
#       ('bank restore admit' + 'warm admit bank='), TTFT bounded, and the
#       needle is retrieved again THROUGH THE RESTORED TENSORS.
#   P6  bank churn: shallow colds fill + recycle banks (plain tier: no
#       persist), until A's deep bank is the only candidate -- its eviction
#       persists (reason=bank-evict, 'cold admit evicting DEEP record').
#   P7  turn 2 again with EVERY bank holding a shallow record: the
#       depth-split restore displaces a shallow LRU (never a pin-tier
#       record) and serves warm from disk again, needle exact.
#   P8  the bank record is a valid SERIAL checkpoint: two fresh serial
#       boots restore it (disk hit) and their greedy continuations must be
#       BYTE-IDENTICAL (serial decode is deterministic; this is the
#       payload-integrity leg -- cont generations are not run-to-run
#       deterministic and are never byte-compared here).
#   P9  (v0.5.1 inc3a) DIVERGENT cross-boot replay: turn 2 whose
#       assistant turn is TRUNCATED (the field shape: a length-limited
#       turn re-rendered with its close marker never byte-prefix-matches
#       its record again).  Control boot (DS4_SERVER_DISK_PARTIAL=0)
#       must silently cold-prefill (0 restore lines = the pre-fix
#       behavior); the default boot must serve a PARTIAL disk restore
#       ('bank restore admit ... partial=1' + partial fork/truncate
#       admit), needle exact through the cut tensors.
#   P10 (v0.5.1 inc3b) checkpoint-at-retire + crash durability: a fresh
#       deep trunk's retire persists reason=bank-checkpoint WITHOUT any
#       evict; a short turn 2 must NOT re-persist (pacing); kill -9 (no
#       graceful persist) then reboot must disk-restore the checkpoint
#       and answer the needle.
#
# PASS = all of the above + 0 illegal / 0 'continuous batch failed' /
# 0 'prompt start' / 0 'cont admit rejected' across every cont segment.
#
# Env overrides (defaults in parens):
#   BP_GATE_HOST (${DS4_BOX33:?})  BINDIR (${DS4_BOX_TREE:?})
#   CTX (131072)  TRUNK_TOK (80000)  PORT (8000)  TUNNEL_PORT (18000)
#   BP_GATE_OUT (/tmp/bank_persist_<ts>)  GGUF (${DS4_GGUF_DIR:?})
#   HEADROOM_MB (6272)  KVDIR (/tmp/bank_persist_kv)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${BP_GATE_HOST:-${DS4_BOX33:?}}
RT=${BINDIR:-${DS4_BOX_TREE:?}}
CTX=${CTX:-131072}
TRUNK_TOK=${TRUNK_TOK:-80000}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
HEADROOM_MB=${HEADROOM_MB:-6272}
OUT=${BP_GATE_OUT:-/tmp/bank_persist_$(date +%Y%m%d_%H%M%S)}
GGUF=${GGUF:-${DS4_GGUF_DIR:?}}
KVDIR=${KVDIR:-/tmp/bank_persist_kv}
BASE=$GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MTP=${MTP-$GGUF/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}
DRAFTER=$GGUF/DSpark-drafter-Q2K-Q8.gguf
SRV=/tmp/bank_persist_srv.log
HERE=$(cd "$(dirname "$0")" && pwd)
NEEDLE_CODE="7371-DELTA-ECHO"

mkdir -p "$OUT"
DRV=$OUT/driver.log
: > "$DRV"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRV"; }
fail(){ log "FAIL: $*"; log "BANK-PERSIST GATE: FAIL"; exit 1; }
URL="http://127.0.0.1:$TUNNEL/v1/chat/completions"

kill_server(){
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server 2>/dev/null; rm -f /tmp/ds4.lock; exit 0"
}

# SIGKILL only -- no SIGTERM first, so nothing gets a graceful shutdown
# persist.  P9 uses it to keep control/fix legs from contaminating the store;
# P10 uses it as the crash under test.
kill9_server(){
  ssh "$R" "pkill -9 -x ds4-server 2>/dev/null; sleep 1; rm -f /tmp/ds4.lock; exit 0"
}

# Boot-time memory guard (standing law).  Back-to-back boots race the dying
# server's reclaim: the session-graph fit check at request time sees the
# shortfall and refuses (observed twice as a "transient" P8 serial failure --
# 8837 MiB allocatable vs ~9098 needed while the previous leg's pages
# drained).  MemAvailable recovers within seconds of the old server's exit,
# so waiting here removes the flake class entirely.
wait_mem(){ # $1=min MemAvailable GiB
  local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge "$1" ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable ${got:-?}G never reached ${1}G"
    sleep 5
  done
}

graceful_stop(){
  ssh "$R" "pkill -x ds4-server"          # SIGTERM: drains, then shutdown persists
  local n=0
  while ssh "$R" "pgrep -x ds4-server >/dev/null" 2>/dev/null; do
    n=$((n+1)); [ $n -ge 60 ] && { ssh "$R" "pkill -9 -x ds4-server"; break; }
    sleep 2
  done
}

boot_cont(){ # $1=fresh_kvdir(0/1) $2=extra env assignments (optional)
  kill_server
  wait_mem 100
  ssh "$R" "$([ "$1" = 1 ] && echo "rm -rf $KVDIR;") mkdir -p $KVDIR; : > $SRV; cd $RT; env \
      DS4_CUDA_NO_HBM_CACHE=1 \
      DS4_BATCH_FIT_HEADROOM_MB=$HEADROOM_MB DS4_SERVER_COALESCE_MAX=2 \
      DS4_CONT_PREFILL_CHUNK=2048 DS4_CONT_MTP_MODE=2 DS4_CONT_DSPARK=1 \
      DS4_DSPARK_MODEL=$DRAFTER DS4_CONT_CAPTURE=1 DS4_SERVER_DEFAULT_TEMP=0 \
      ${2:-} \
      setsid nohup ./ds4-server -m $BASE ${MTP:+--mtp} ${MTP:---no-mtp} --cuda -c $CTX --port $PORT \
      --kv-disk-dir $KVDIR --kv-disk-space-mb 20000 \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
      sleep 3
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "BOOT-DIED: $(ssh "$R" "tail -3 $SRV" 2>/dev/null | tr '\n' ' ')"
    fi
    sleep 10; n=$((n+10)); [ $n -ge 1500 ] && fail "boot timeout"
  done
  ssh "$R" "grep -q 'max_seq=2' $SRV" || fail "booted max_seq != 2 (placement drift)"
  curl -s -m 5 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null 2>&1 || {
    ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true
    sleep 2
    curl -s -m 10 "http://127.0.0.1:$TUNNEL/v1/models" >/dev/null || fail "tunnel/:$TUNNEL unreachable"
  }
}

cont_health(){ # $1=segment-file $2=tag
  for pat in 'illegal' 'continuous batch failed' 'prompt start' 'cont admit rejected'; do
    [ "$(grep -c "$pat" "$1")" -eq 0 ] || fail "$2: '$pat' in server segment"
  done
}

log "P0: build trunk needle prompt (~$TRUNK_TOK tok @40%) + shallow colds"
python3 "$DS4_GATES/lib/needle_prompt.py" "$TRUNK_TOK" "$OUT/trunk.json" 0.40 "$NEEDLE_CODE" \
  | tee -a "$DRV" || fail "trunk prompt build"
python3 "$DS4_GATES/lib/needle_prompt.py" "$TRUNK_TOK" "$OUT/trunkb.json" 0.60 "4415-LIMA-TANGO" \
  | tee -a "$DRV" || fail "trunk B prompt build"
python3 - "$OUT" <<'EOF' || fail "cold prompt build"
import json, sys
out = sys.argv[1]
text = open("speed-bench/promessi_sposi.txt", encoding="utf-8", errors="replace").read()[:7000]
msg = "[cold-0] " + text + "\n\nOne-sentence mood summary, please."
json.dump({"model": "deepseek-chat",
           "messages": [{"role": "user", "content": msg}],
           "max_tokens": 8, "temperature": 0},
          open("%s/cold0.json" % out, "w"))
EOF

log "P1: boot cont server (fresh kv dir $KVDIR)"
boot_cont 1
OFF=$(ssh "$R" "wc -l < $SRV")

log "P2: admit trunk A cold (deep prefill) + retire"
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/trunk.json" "$URL" TRUNK "$OUT/trunk_result.json" \
  > "$OUT/trunk.out" 2>&1 || fail "trunk client"
cat "$OUT/trunk.out" | tee -a "$DRV"
grep -q "$NEEDLE_CODE" "$OUT/trunk.out" || fail "P2 needle miss (pre-persist)"

log "P3: graceful shutdown -> bank-shutdown persist"
graceful_stop
ssh "$R" "tail -n +$((OFF+1)) $SRV" > "$OUT/seg_p2.log"
cont_health "$OUT/seg_p2.log" "P2"
grep -a 'bank persisted' "$OUT/seg_p2.log" | tee -a "$DRV"
grep -aq 'bank persisted.*reason=bank-shutdown' "$OUT/seg_p2.log" \
  || fail "no bank-shutdown persist line"
NKVC=$(ssh "$R" "ls $KVDIR/*.kv 2>/dev/null | wc -l")
[ "$NKVC" -ge 1 ] || fail "no .kv file in $KVDIR after shutdown"
log "P3: $NKVC .kv file(s) on disk"

log "P4: reboot on the same kv dir"
boot_cont 0
OFF=$(ssh "$R" "wc -l < $SRV")

log "P5: turn 2 -> disk-restored warm admit + needle through restored tensors"
python3 - "$OUT/trunk.json" "$OUT/trunk_result.json" "$OUT/turn2.json" <<'EOF' || fail "turn2 build"
import json, sys
b = json.load(open(sys.argv[1]))
r = json.load(open(sys.argv[2]))
b["messages"].append({"role": "assistant", "content": r["text"]})
b["messages"].append({"role": "user", "content":
    "State the keystone code once more, exactly, then describe in one "
    "sentence where in the archive it appeared."})
b["max_tokens"] = 96
json.dump(b, open(sys.argv[3], "w"))
EOF
T5_START=$(date +%s)
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/turn2.json" "$URL" TURN2 > "$OUT/turn2.out" 2>&1 \
  || fail "turn2 client"
cat "$OUT/turn2.out" | tee -a "$DRV"
ssh "$R" "tail -n +$((OFF+1)) $SRV" > "$OUT/seg_p5.log"
cont_health "$OUT/seg_p5.log" "P5"
grep -aq 'bank restore admit' "$OUT/seg_p5.log" || fail "P5: no 'bank restore admit'"
grep -aqE 'warm admit bank=[0-9]+ cached=[0-9]{4,}' "$OUT/seg_p5.log" \
  || fail "P5: restored record did not serve a warm admit"
grep -q "$NEEDLE_CODE" "$OUT/turn2.out" || fail "P5 needle miss THROUGH RESTORED TENSORS"
TTFT=$(grep -oE 'ttft=[0-9.]+' "$OUT/turn2.out" | head -1 | cut -d= -f2)
python3 -c "import sys; sys.exit(0 if float('$TTFT') < 120.0 else 1)" \
  || fail "P5 TTFT ${TTFT}s not seconds-class (cold prefill would be ~200s+)"
log "P5: restored warm admit, ttft=${TTFT}s, needle exact"

log "P6: second deep trunk fills the free bank; one cold then MUST evict the LRU deep bank"
OFF=$(ssh "$R" "wc -l < $SRV")
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/trunkb.json" "$URL" TRUNKB > "$OUT/trunkb.out" 2>&1 \
  || fail "trunk B client"
grep -q "4415-LIMA-TANGO" "$OUT/trunkb.out" || fail "P6 trunk B needle miss"
# trunk B must have taken the free (no-value) bank without evicting anything.
ssh "$R" "tail -n +$((OFF+1)) $SRV" > "$OUT/seg_p6a.log"
[ "$(grep -ac 'bank persisted.*reason=bank-evict' "$OUT/seg_p6a.log")" -eq 0 ] \
  || fail "P6: trunk B admission persisted something (should take the free bank)"
curl -s -m 300 "$URL" -H 'Content-Type: application/json' \
  -d @"$OUT/cold0.json" > "$OUT/cold0.resp" || fail "cold0 request"
ssh "$R" "tail -n +$((OFF+1)) $SRV" > "$OUT/seg_p6.log"
cont_health "$OUT/seg_p6.log" "P6"
grep -a 'bank persisted\|evicting DEEP' "$OUT/seg_p6.log" | tee -a "$DRV"
grep -aq 'cold admit evicting DEEP record' "$OUT/seg_p6.log" \
  || fail "P6: deep eviction never happened (both banks deep, cold must evict LRU deep)"
grep -aq 'bank persisted.*reason=bank-evict' "$OUT/seg_p6.log" \
  || fail "P6: deep eviction did not persist (pin tier broken)"
NEVICT=$(grep -ac 'bank persisted.*reason=bank-evict' "$OUT/seg_p6.log")
[ "$NEVICT" -eq 1 ] || fail "P6: expected exactly 1 bank-evict persist, got $NEVICT"

log "P7: turn 2 again -- banks hold {shallow cold, deep pin} -> depth-split must displace the SHALLOW one"
OFF=$(ssh "$R" "wc -l < $SRV")
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/turn2.json" "$URL" TURN2B > "$OUT/turn2b.out" 2>&1 \
  || fail "turn2b client"
cat "$OUT/turn2b.out" | tee -a "$DRV"
ssh "$R" "tail -n +$((OFF+1)) $SRV" > "$OUT/seg_p7.log"
cont_health "$OUT/seg_p7.log" "P7"
grep -aq 'bank restore admit' "$OUT/seg_p7.log" || fail "P7: no displace-restore"
grep -q "$NEEDLE_CODE" "$OUT/turn2b.out" || fail "P7 needle miss after displace-restore"
graceful_stop

log "P8: serial cross-restore identity (two fresh serial boots, byte compare)"
# Same question the cont turn-2 legs use: that phrasing reliably elicits the
# FULL code (run 3 taught us "then stop" phrasing gets an obedient model that
# truncates mid-code — the byte-identity held, the grep didn't).
python3 - "$OUT/trunk.json" "$OUT/trunk_result.json" "$OUT/serial2.json" <<'EOF' || fail "serial payload build"
import json, sys
b = json.load(open(sys.argv[1]))
r = json.load(open(sys.argv[2]))
b["messages"].append({"role": "assistant", "content": r["text"]})
b["messages"].append({"role": "user", "content":
    "State the keystone code once more, exactly, then describe in one "
    "sentence where in the archive it appeared."})
b["max_tokens"] = 96
b["temperature"] = 0
b["model"] = "deepseek-chat"
json.dump(b, open(sys.argv[3], "w"))
EOF
scp -q "$OUT/serial2.json" "$R:/tmp/bp_serial2.json" || fail "serial payload scp"
# Serial hits CONSUME records above cold_max_tokens (30K default), so leg s1
# would eat the deep record and s2 would miss.  Snapshot the store and put it
# back between legs -- the point of P8 is two independent restores of the
# SAME bytes.
ssh "$R" "rm -rf ${KVDIR}.bak; cp -a $KVDIR ${KVDIR}.bak"
for leg in s1 s2; do
  ssh "$R" "rm -rf $KVDIR; cp -a ${KVDIR}.bak $KVDIR"
  kill_server
  wait_mem 100
  # v0.5.2: DS4_SERVER_COALESCE=0 makes these boots TRULY serial-only (no
  # batch ctx, no banks).  CONTINUOUS=0 alone still builds banks that eat the
  # box for nothing here -- the P8 memory-marginality flake class, and under
  # serial right-sizing (bank-holding boots only) the shrunken session would
  # skip the deep ring-image record this leg exists to restore.  The leg's
  # subject is serial-tier restore semantics; banks were never part of it.
  ssh "$R" ": > /tmp/bp_${leg}.log; cd $RT && (env DS4_SERVER_CONTINUOUS=0 DS4_SERVER_COALESCE=0 DS4_SERVER_DEFAULT_TEMP=0 \
      setsid nohup ./ds4-server -m $BASE --no-mtp --cuda -c $CTX --port $PORT \
      --kv-disk-dir $KVDIR --kv-disk-space-mb 20000 \
      > /tmp/bp_${leg}.log 2>&1 < /dev/null &) ; exit 0"
  n=0
  until ssh "$R" "curl -sf -o /dev/null -m 3 http://127.0.0.1:$PORT/v1/models" 2>/dev/null; do
    n=$((n+1)); [ $n -ge 240 ] && fail "$leg boot timeout"
    ssh "$R" "pgrep -x ds4-server >/dev/null" 2>/dev/null || fail "$leg BOOT-DIED"
    sleep 5
  done
  ssh "$R" "curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions \
              -H 'Content-Type: application/json' -d @/tmp/bp_serial2.json" \
    > "$OUT/serial_${leg}.json" || fail "$leg completion"
  python3 - "$OUT/serial_${leg}.json" <<'PYEOF' > "$OUT/serial_${leg}.txt" || fail "serial parse"
import json, sys
r = json.load(open(sys.argv[1]))
sys.stdout.write(r["choices"][0]["message"]["content"])
sys.stderr.write("cached=%d prompt=%d\n" % (
    r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0),
    r["usage"]["prompt_tokens"]))
PYEOF
  ssh "$R" "grep -a 'kv cache' /tmp/bp_${leg}.log | tail -3" | tee -a "$DRV"
  ssh "$R" "grep -aq 'kv cache hit' /tmp/bp_${leg}.log" || fail "$leg: serial restore did not disk-hit"
done
cmp -s "$OUT/serial_s1.txt" "$OUT/serial_s2.txt" \
  || fail "P8: serial restored continuations differ (payload integrity)"
grep -q "$NEEDLE_CODE" "$OUT/serial_s1.txt" || fail "P8: serial needle miss"
kill_server

log "P9: divergent cross-boot replay (v0.5.1 inc3a)"
# The field shape: the assistant turn is TRUNCATED mid-sentence, so the next
# prompt's re-render (which closes the turn) diverges from every stored
# record before its end -- the exact byte-prefix lookup can never hit.
python3 - "$OUT/trunk.json" "$OUT/trunk_result.json" "$OUT/turn2_div.json" <<'EOF' || fail "P9 divergent payload build"
import json, sys
b = json.load(open(sys.argv[1]))
r = json.load(open(sys.argv[2]))
cutlen = max(1, (len(r["text"]) * 3) // 5)      # keep 60%, cut mid-sentence
b["messages"].append({"role": "assistant", "content": r["text"][:cutlen]})
b["messages"].append({"role": "user", "content":
    "State the keystone code once more, exactly, then describe in one "
    "sentence where in the archive it appeared."})
b["max_tokens"] = 96
json.dump(b, open(sys.argv[3], "w"))
EOF

# P9a CONTROL (DS4_SERVER_DISK_PARTIAL=0): must silently cold-prefill --
# zero restore lines is exactly the pre-fix field behavior.
ssh "$R" "rm -rf $KVDIR; cp -a ${KVDIR}.bak $KVDIR"
boot_cont 0 "DS4_SERVER_DISK_PARTIAL=0"
OFF=$(ssh "$R" "wc -l < $SRV")
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/turn2_div.json" "$URL" P9CTRL > "$OUT/p9ctrl.out" 2>&1 \
  || fail "P9a client"
cat "$OUT/p9ctrl.out" | tee -a "$DRV"
ssh "$R" "tail -n +$((OFF+1)) $SRV" > "$OUT/seg_p9a.log"
cont_health "$OUT/seg_p9a.log" "P9a"
[ "$(grep -ac 'bank restore admit' "$OUT/seg_p9a.log")" -eq 0 ] \
  || fail "P9a: control restored from disk with the kill switch set"
grep -q "$NEEDLE_CODE" "$OUT/p9ctrl.out" || fail "P9a needle miss (cold control)"
kill9_server

# P9b FIX: partial disk restore + partial admit through the cut.
ssh "$R" "rm -rf $KVDIR; cp -a ${KVDIR}.bak $KVDIR"
boot_cont 0
OFF=$(ssh "$R" "wc -l < $SRV")
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/turn2_div.json" "$URL" P9FIX > "$OUT/p9fix.out" 2>&1 \
  || fail "P9b client"
cat "$OUT/p9fix.out" | tee -a "$DRV"
ssh "$R" "tail -n +$((OFF+1)) $SRV" > "$OUT/seg_p9b.log"
cont_health "$OUT/seg_p9b.log" "P9b"
grep -aq 'bank restore admit.*partial=1' "$OUT/seg_p9b.log" \
  || fail "P9b: no partial disk restore line"
grep -aqE 'partial (fork|truncate) admit' "$OUT/seg_p9b.log" \
  || fail "P9b: restored record did not serve a partial admit"
grep -q "$NEEDLE_CODE" "$OUT/p9fix.out" || fail "P9b needle miss THROUGH CUT TENSORS"
TTFT=$(grep -oE 'ttft=[0-9.]+' "$OUT/p9fix.out" | head -1 | cut -d= -f2)
python3 -c "import sys; sys.exit(0 if float('$TTFT') < 120.0 else 1)" \
  || fail "P9b TTFT ${TTFT}s not restore-class"
log "P9: control cold, fix partial-restored (ttft=${TTFT}s), needle exact both"
kill9_server

log "P10: checkpoint-at-retire + kill -9 crash durability (v0.5.1 inc3b)"
boot_cont 1                              # clean store: nothing pre-persisted
OFF=$(ssh "$R" "wc -l < $SRV")
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/trunk.json" "$URL" P10TRUNK "$OUT/p10_result.json" \
  > "$OUT/p10trunk.out" 2>&1 || fail "P10 trunk client"
grep -q "$NEEDLE_CODE" "$OUT/p10trunk.out" || fail "P10 trunk needle miss"
ssh "$R" "tail -n +$((OFF+1)) $SRV" > "$OUT/seg_p10a.log"
cont_health "$OUT/seg_p10a.log" "P10a"
NCKPT=$(grep -ac 'bank persisted.*reason=bank-checkpoint' "$OUT/seg_p10a.log")
[ "$NCKPT" -eq 1 ] || fail "P10: expected exactly 1 checkpoint persist at retire, got $NCKPT"
# Short turn 2 (same conversation, from THIS run's answer -- cont decode is
# not run-to-run deterministic): grows the bank by ~100 tokens, far below the
# continued interval, so its retire must NOT re-persist.
python3 - "$OUT/trunk.json" "$OUT/p10_result.json" "$OUT/p10_turn2.json" <<'EOF' || fail "P10 turn2 build"
import json, sys
b = json.load(open(sys.argv[1]))
r = json.load(open(sys.argv[2]))
b["messages"].append({"role": "assistant", "content": r["text"]})
b["messages"].append({"role": "user", "content":
    "State the keystone code once more, exactly, then describe in one "
    "sentence where in the archive it appeared."})
b["max_tokens"] = 96
json.dump(b, open(sys.argv[3], "w"))
EOF
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/p10_turn2.json" "$URL" P10T2 > "$OUT/p10t2.out" 2>&1 \
  || fail "P10 turn2 client"
grep -q "$NEEDLE_CODE" "$OUT/p10t2.out" || fail "P10 turn2 needle miss (pre-crash)"
ssh "$R" "tail -n +$((OFF+1)) $SRV" > "$OUT/seg_p10b.log"
cont_health "$OUT/seg_p10b.log" "P10b"
NCKPT=$(grep -ac 'bank persisted.*reason=bank-checkpoint' "$OUT/seg_p10b.log")
[ "$NCKPT" -eq 1 ] || fail "P10: pacing broken -- checkpoint count $NCKPT after short turn 2 (want 1)"

log "P10: kill -9 (crash: no graceful persist) then reboot + replay"
kill9_server
boot_cont 0
OFF=$(ssh "$R" "wc -l < $SRV")
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/p10_turn2.json" "$URL" P10CRASH > "$OUT/p10crash.out" 2>&1 \
  || fail "P10 post-crash client"
cat "$OUT/p10crash.out" | tee -a "$DRV"
ssh "$R" "tail -n +$((OFF+1)) $SRV" > "$OUT/seg_p10c.log"
cont_health "$OUT/seg_p10c.log" "P10c"
grep -aq 'bank restore admit' "$OUT/seg_p10c.log" \
  || fail "P10: crash checkpoint did not restore (retire never became durable)"
grep -aqE 'warm admit bank=[0-9]+ cached=[0-9]{4,}' "$OUT/seg_p10c.log" \
  || fail "P10: restored checkpoint did not serve a warm admit"
grep -q "$NEEDLE_CODE" "$OUT/p10crash.out" || fail "P10 needle miss through crash-restored tensors"
TTFT=$(grep -oE 'ttft=[0-9.]+' "$OUT/p10crash.out" | head -1 | cut -d= -f2)
python3 -c "import sys; sys.exit(0 if float('$TTFT') < 120.0 else 1)" \
  || fail "P10 TTFT ${TTFT}s not restore-class"
log "P10: checkpoint survived kill -9, restored warm (ttft=${TTFT}s), needle exact"
kill_server

log "BANK-PERSIST GATE: ALL PHASES PASS"
exit 0
