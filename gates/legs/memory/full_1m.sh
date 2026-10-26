#!/bin/bash
# gates/legs/memory/full_1m.sh — v0.6.3 Inc 4 gate: the full 1,048,576 window
# (audit docket local/docs/v062/audit-1m-depth-overflow.md).
#
# Legs, one GB10 over SSH (self-locking, v062-gate mold):
#   F2   -c 32768 EXACT-FULL bank persist/restore (audit Finding 2):
#        a counting fill decodes the bank to exactly seq_cap tokens
#        (usage must say prompt+completion == 32768, finish length),
#        graceful shutdown persists it, a fresh boot's turn 2 must serve
#        a DISK-RESTORED admit ('bank restore admit') and answer the
#        needle -- pre-fix the exactly-full payload was rejected at the
#        header ("does not fit current per-bank token bound").
#   CAL  -c 1048576 ship-env boot; a ~50k probe measures the corpus
#        chars/token ratio (the 3.4 estimate drifts +/-4% per rotation,
#        wider than the 32k-token target band -- so the gate calibrates
#        itself; probe-must-reach-stop-condition law).
#   A    calibrated ~1.03M-token needle prompt, needle at depth 0.999 --
#        INSIDE the deepest ~32k tokens (comp rows past 7,936) that the
#        pre-fix fixed-buffer dispatch truncated or refused.  PASS =
#        usage.prompt_tokens lands in (1,016,500 .. 1,044,000], the
#        needle answers exactly, and the serving log carries 0 illegal /
#        0 'score buffer too small' / 0 cont failures.  Default path
#        (capture ON records HG, uncapped).
#   B    fresh boot with DS4_CUDA_NO_ATTN_HG=1: the SAME calibrated
#        prompt rides the online-fallback class the fix made live-scalar
#        (the class that silently truncated or refused pre-fix).  Same
#        asserts.
#
# Env overrides: R (${DS4_BOX33:?}) BINDIR (${DS4_BOX_TREE:?})
#   PORT (8000) OUT (/tmp/full1m_gate_<ts>) GGUF (${DS4_GGUF_DIR:?})
#   TARGET_MID (1030000) BAND_LO (1016500) BAND_HI (1044000)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
OUT=${OUT:-/tmp/full1m_gate_$(date +%Y%m%d_%H%M%S)}
GGUF=${GGUF:-${DS4_GGUF_DIR:?}}
TARGET_MID=${TARGET_MID:-1030000}
BAND_LO=${BAND_LO:-1016500}
BAND_HI=${BAND_HI:-1044000}
BASE=$GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MTP=$GGUF/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DRAFTER=$GGUF/DSpark-drafter-Q2K-Q8.gguf
SRV=/tmp/full1m_srv.log
KVDIR=/tmp/full1m_kv
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
DRV=$OUT/driver.log
log(){ echo "[$(date +%H:%M:%S)] F1M: $*" | tee -a "$DRV"; }

SSH(){ perl -e 'alarm 240; exec @ARGV or die' ssh "$@"; }
SSH_SERVE(){ perl -e 'alarm 5400; exec @ARGV or die' ssh "$@"; }

TS=$(date +%s); NONCE="full1m_${TS}_$$"
LOCKDIR=/tmp/full1m_gate.lockdir
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "another full1m_gate instance is running"; exit 3
fi
echo $$ > "$LOCKDIR/pid"
BOX_LOCKED=0
full_exit(){
  rc=$?
  SSH "$R" "pkill -x ds4-server 2>/dev/null; sleep 1; pkill -9 -x ds4-server 2>/dev/null; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null
  if [ "$BOX_LOCKED" = 1 ]; then
    SSH "$R" "rm -rf /tmp/ds4_box_lock; exit 0" 2>/dev/null
    log "cleanup: server down, box lock freed; artifacts in $OUT"
  fi
  rm -rf "$LOCKDIR"
  exit $rc
}
trap full_exit EXIT INT TERM
fail(){ log "FAIL: $*"; SSH "$R" "tail -8 $SRV 2>/dev/null; exit 0" 2>/dev/null | tee -a "$DRV"; log "FULL1M GATE: FAILED"; exit 1; }

OWN=$(SSH "$R" "cat /tmp/ds4_box_lock/nonce 2>/dev/null; exit 0" 2>/dev/null)
[ -n "$OWN" ] && fail "box lock held by another session ($OWN) -- refusing"
SSH "$R" "mkdir /tmp/ds4_box_lock && echo '$NONCE' > /tmp/ds4_box_lock/nonce" \
  || fail "could not take the box lock"
BOX_LOCKED=1
log "box lock taken ($NONCE)"

wait_mem(){ local n=0 got=0
  while :; do
    got=$(SSH "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable ${got:-?}G never reached 100G"; sleep 5
  done }

boot(){ # $1 = ctx  $2 = extra env  $3 = extra args
  log "boot: ctx=$1 env='${2:-}' args='${3:-}'"
  SSH "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0"
  wait_mem
  SSH "$R" ": > $SRV; cd $BINDIR && env ${2:-} \
      DS4_CONT_PREFILL_CHUNK=2048 DS4_CONT_CAPTURE=1 DS4_SERVER_DEFAULT_TEMP=0 \
      setsid --fork nohup ./ds4-server -m $BASE --cuda -c $1 --port $PORT ${3:-} \
      > $SRV 2>&1 < /dev/null; exit 0"
  local n=0
  until SSH "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    SSH "$R" "pgrep -x ds4-server >/dev/null" 2>/dev/null || {
      sleep 3
      SSH "$R" "pgrep -x ds4-server >/dev/null" 2>/dev/null || \
        fail "BOOT-DIED: $(SSH "$R" "tail -3 $SRV; exit 0" 2>/dev/null | tr '\n' ' ')"; }
    sleep 10; n=$((n+10)); [ $n -ge 1500 ] && fail "boot timeout"
  done
  log "boot up"
}

serve(){ # $1 = local request json  $2 = local response json  (on-box curl)
  scp -q "$1" "$R:/tmp/full1m_req.json" || fail "req scp"
  SSH_SERVE "$R" "curl -s -m 5100 http://127.0.0.1:$PORT/v1/chat/completions \
      -H 'Content-Type: application/json' --data-binary @/tmp/full1m_req.json \
      > /tmp/full1m_resp.json 2>/dev/null; exit 0"
  scp -q "$R:/tmp/full1m_resp.json" "$2" || fail "resp scp"
  [ -s "$2" ] || fail "empty response for $1"
}

jget(){ python3 -c "
import json,sys
d = json.load(open('$1'))
print(d$2)
" 2>/dev/null; }

srv_zero(){ # $1 = pattern that must be ABSENT
  local c
  c=$(SSH "$R" "grep -c '$1' $SRV 2>/dev/null; exit 0" 2>/dev/null | tail -1)
  [ "${c:-0}" = "0" ] || fail "srv.log carries '$1' x$c"
}

# ============================ Leg F2 =======================================
log "LEG F2: exact-full bank persist/restore at -c 32768"
SSH "$R" "rm -rf $KVDIR; mkdir -p $KVDIR; exit 0"
boot 32768 "" "--no-spec --no-mtp --kv-disk-dir $KVDIR --kv-disk-space-mb 8000"
python3 - "$OUT/f2_fill.json" "$HERE/../../.." <<'PYEOF'
import glob, json, os, sys
out = sys.argv[1]
repo = sys.argv[2]
srcs = sorted(glob.glob(os.path.join(repo, 'local/docs/**/*.md'), recursive=True))
srcs += sorted(glob.glob(os.path.join(repo, '*.md')))
if not srcs: sys.exit('f2: no corpus files')
parts, total = [], 0
target_chars = int(26000 * 3.4)
for p in srcs * 8:
    try: t = open(p, encoding='utf-8', errors='replace').read()
    except OSError: continue
    parts.append(t); total += len(t)
    if total >= target_chars: break
doc = ''.join(parts)[:target_chars]
cut = len(doc) // 2
doc = doc[:cut] + "\n\nIMPORTANT: the keystone code is 6402-FULL-BANK. Remember it.\n\n" + doc[cut:]
body = {"model": "deepseek-chat", "temperature": 0, "max_tokens": 32768,
        "messages": [{"role": "user", "content":
            "Below is a long archive. Read it.\n\n" + doc +
            "\n\n===== END =====\n\nFirst state the keystone code exactly. Then count upward from 1 "
            "(one number per line) and do not stop counting for any reason."}]}
json.dump(body, open(out, 'w'))
print("f2 fill request written,", len(doc), "chars")
PYEOF
[ -s "$OUT/f2_fill.json" ] || fail "F2 fill build"
log "F2: serving the counting fill (decodes to the bank bound)"
serve "$OUT/f2_fill.json" "$OUT/f2_fill_resp.json"
P=$(jget "$OUT/f2_fill_resp.json" "['usage']['prompt_tokens']")
C=$(jget "$OUT/f2_fill_resp.json" "['usage']['completion_tokens']")
FIN=$(jget "$OUT/f2_fill_resp.json" "['choices'][0]['finish_reason']")
[ -n "$P" ] && [ -n "$C" ] || fail "F2: fill response unparsable"
TOT=$((P + C))
log "F2: fill prompt=$P completion=$C total=$TOT finish=$FIN"
[ "$TOT" = "32768" ] || fail "F2: bank not exactly full ($TOT != 32768; the == case is the leg)"
[ "$FIN" = "length" ] || fail "F2: fill did not length-stop (finish=$FIN)"
log "F2: graceful shutdown (persists the exactly-full bank)"
SSH "$R" "pkill -x ds4-server; exit 0"
n=0
until SSH "$R" "grep -q 'reason=bank-shutdown' $SRV" 2>/dev/null; do
  n=$((n+1)); [ $n -ge 12 ] && fail "F2: no bank-shutdown persist line after 60s"
  sleep 5
done
cp /dev/null "$OUT/f2_srv1.log"; SSH "$R" "cat $SRV; exit 0" > "$OUT/f2_srv1.log" 2>/dev/null
boot 32768 "" "--no-spec --no-mtp --kv-disk-dir $KVDIR --kv-disk-space-mb 8000"
python3 - "$OUT/f2_fill.json" "$OUT/f2_fill_resp.json" "$OUT/f2_turn2.json" <<'PYEOF'
import json, sys
req = json.load(open(sys.argv[1]))
resp = json.load(open(sys.argv[2]))
answer = resp['choices'][0]['message']['content']
# An exactly-full bank leaves no context room for a follow-up: replaying
# the whole conversation plus a new turn exceeds -c and is (honestly)
# refused at parse.  So turn 2 TRUNCATES the assistant turn (the P9
# partial-restore shape: the counting output loses its last ~60 lines at
# a clean line boundary) -- the request fits, and the restore header
# check, where Finding 2 lives, still runs against the FULL persisted
# payload with saved_tokens == seq_cap.  Pre-fix that header rejected
# ("does not fit the per-bank token bound") and no restore admit could
# happen at all; post-fix the partial disk restore serves the prefix.
lines = answer.rstrip().split("\n")
answer = "\n".join(lines[:max(1, len(lines) - 60)])
req['messages'].append({"role": "assistant", "content": answer})
req['messages'].append({"role": "user", "content": "What was the keystone code? Answer with the code only."})
req['max_tokens'] = 32
json.dump(req, open(sys.argv[3], 'w'))
PYEOF
log "F2: turn 2 on the fresh boot (must be a disk-restored admit)"
serve "$OUT/f2_turn2.json" "$OUT/f2_turn2_resp.json"
grep -q '6402-FULL-BANK' "$OUT/f2_turn2_resp.json" || fail "F2: needle missed on turn 2"
SSH "$R" "grep -q 'bank restore admit' $SRV" 2>/dev/null || fail "F2: turn 2 was not a disk restore (Finding 2 regressed?)"
srv_zero "does not fit current per-bank token bound"
log "F2 PASS (exactly-full bank persisted, restored, needle exact)"
SSH "$R" "cat $SRV; exit 0" > "$OUT/f2_srv2.log" 2>/dev/null

# ====================== Legs CAL + A (one boot) ============================
SPEC_ENV="DS4_CONT_MTP_MODE=2 DS4_CONT_DSPARK=1 DS4_DSPARK_MODEL=$DRAFTER"
log "LEG CAL+A: -c 1048576 ship-env boot"
boot 1048576 "$SPEC_ENV" "--mtp $MTP"
log "CAL: 50k probe measures the corpus ratio"
python3 "$DS4_GATES/lib/needle_prompt.py" 50000 "$OUT/cal.json" 0.5 "1111-CAL-PROBE" deepseek-chat 1 | tee -a "$DRV" || fail "cal prompt"
serve "$OUT/cal.json" "$OUT/cal_resp.json"
CALP=$(jget "$OUT/cal_resp.json" "['usage']['prompt_tokens']")
[ -n "$CALP" ] && [ "$CALP" -gt 20000 ] || fail "CAL: probe unparsable (prompt_tokens=$CALP)"
log "CAL: 50000-target prompt landed at $CALP tokens"

ATTEMPT=1
TARGET=$TARGET_MID
while :; do
  TGT=$(python3 -c "print(int($TARGET * 50000 / $CALP))")
  log "A: building ~$TARGET-token prompt (builder target $TGT, needle @0.999) [attempt $ATTEMPT]"
  python3 "$DS4_GATES/lib/needle_prompt.py" "$TGT" "$OUT/deep.json" 0.999 "9331-TAIL-ZULU" | tee -a "$DRV" || fail "deep prompt"
  log "A: serving the deep needle (prefill ~1M tokens; this is the long pole)"
  serve "$OUT/deep.json" "$OUT/deep_resp.json"
  DP=$(jget "$OUT/deep_resp.json" "['usage']['prompt_tokens']")
  [ -n "$DP" ] || fail "A: deep response unparsable: $(head -c 300 "$OUT/deep_resp.json")"
  log "A: prompt landed at $DP tokens (band $BAND_LO..$BAND_HI)"
  if [ "$DP" -gt "$BAND_LO" ] && [ "$DP" -le "$BAND_HI" ]; then break; fi
  [ $ATTEMPT -ge 2 ] && fail "A: prompt outside the band twice ($DP)"
  CALP=$(python3 -c "print(int($CALP * $DP / $TARGET))")   # refresh ratio from the big sample
  ATTEMPT=2
done
grep -q '9331-TAIL-ZULU' "$OUT/deep_resp.json" || fail "A: tail needle missed (deepest rows not attended)"
srv_zero "score buffer too small"
srv_zero "illegal"
srv_zero "continuous batch failed"
srv_zero "cont admit rejected"
log "A PASS (prompt=$DP > 1,015,936 cliff; tail needle exact on the default path)"
SSH "$R" "cat $SRV; exit 0" > "$OUT/a_srv.log" 2>/dev/null

# ============================ Leg B ========================================
log "LEG B: forced fallback (DS4_CUDA_NO_ATTN_HG=1), same calibrated prompt"
boot 1048576 "DS4_CUDA_NO_ATTN_HG=1 $SPEC_ENV" "--mtp $MTP"
serve "$OUT/deep.json" "$OUT/b_resp.json"
BP=$(jget "$OUT/b_resp.json" "['usage']['prompt_tokens']")
[ "$BP" = "$DP" ] || fail "B: prompt token drift vs leg A ($BP vs $DP)"
grep -q '9331-TAIL-ZULU' "$OUT/b_resp.json" || fail "B: tail needle missed on the fallback path (the pre-fix truncation class)"
srv_zero "score buffer too small"
srv_zero "illegal"
srv_zero "continuous batch failed"
srv_zero "cont admit rejected"
log "B PASS (tail needle exact on the forced online-fallback path)"
SSH "$R" "cat $SRV; exit 0" > "$OUT/b_srv.log" 2>/dev/null

log "FULL1M GATE: ALL LEGS PASS — artifacts in $OUT"
