#!/bin/bash
# gates/legs/memory/warm_replay_micro.sh — the warm-record replay micro gate (2026-09-04, from the soak-replay
# adjudication).  Box-side on the server box: boot the soak's shape (-c 32768, COALESCE_MAX=18), send the SAME
# prompt twice on the continuous lane, read ds4_tokens_prefilled_total{kind="cached"} around each send.  Three
# configs: Vision base + encoder with a 31k text prompt, the same with a small prompt, and the TEXT base with
# the 31k prompt.  Verdict: WARM_MICRO_PASS when every second send is cached; else WARM_MICRO_FAIL <n of 3>.
# Bodies: $DS4_BOX_SCRATCH/warm_micro/{tenant_1.json,wave_1.json} (a soak's tenant + wave body; any two chat
# bodies of ~31k and ~350 prompt tokens do).  Receipt 2026-09-04 20:12: PASS, vision 31k cached 30976/31018.
set -uo pipefail
VERDICT_TAG=WARM_MICRO; . "$(dirname "$0")/../../lib/verdict.sh"
T=${DS4_BOX_TREE:?}; OUT=${RB_OUT:-${OUT:-${DS4_BOX_SCRATCH:?}/warm_micro_out}}; mkdir -p "$OUT"; PORT=${PORT:-8097}
BIG=${BIG:-${DS4_BOX_SCRATCH:?}/warm_micro/tenant_1.json}; SMALL=${SMALL:-${DS4_BOX_SCRATCH:?}/warm_micro/wave_1.json}
[ -s "$BIG" ] && [ -s "$SMALL" ] || fail "prompt bodies missing: $BIG $SMALL"
cached(){ curl -s "http://127.0.0.1:$PORT/metrics" | awk '/^ds4_tokens_prefilled_total\{kind="cached"\}/{print $2}' | tail -1; }
killsrv(){ for p in $(pgrep -x ds4-server); do kill $p; done; sleep 2; for p in $(pgrep -x ds4-server); do kill -9 $p; done; }
boot(){ # $1 tag, $2 model args
  killsrv; : > "$OUT/srv_$1.log"
  ( cd "$T" && setsid nohup env DS4_SERVER_COALESCE_MAX=18 ./ds4-server -c 32768 --port $PORT $2 > "$OUT/srv_$1.log" 2>&1 < /dev/null & )
  local n=0; until grep -qE 'persistent batch ctx ready' "$OUT/srv_$1.log" 2>/dev/null; do n=$((n+1)); [ $n -gt 60 ] && fail "$1: boot timeout"; sleep 5; done
  n=0; until curl -s -m 5 "http://127.0.0.1:$PORT/metrics" | grep -q ds4_tokens_prefilled_total; do n=$((n+1)); [ $n -gt 60 ] && fail "$1: listener never answered /metrics"; sleep 2; done   # the ready line precedes the listener
  grep -E 'batch fit:.*max_seq|persistent batch ctx ready' "$OUT/srv_$1.log" | tail -2 | cut -c1-160 | sed 's/^/   /'; }
send(){ curl -s -m 600 -o "$OUT/$1.out" -w '%{http_code}' "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @"$2"; }
fails=0
run(){ # $1 tag $2 model args $3 body
  boot "$1" "$2"; local c0 c1 c2 h1 h2
  c0=$(cached); h1=$(send "${1}_1" "$3"); c1=$(cached); h2=$(send "${1}_2" "$3"); c2=$(cached)
  local pt; pt=$(python3 -c "import json,sys; j=json.load(open('$OUT/${1}_2.out')); print(j.get('usage',{}).get('prompt_tokens'), j.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens'))" 2>/dev/null)
  log "$1: http $h1/$h2 cached-counter $c0 -> $c1 -> $c2 (second send +$((c2-c1))); second usage prompt/cached_tokens = $pt"
  grep -E 'keeps .*warm record|warm|reuse|fork|match' "$OUT/srv_$1.log" | grep -vE tokentile | tail -4 | cut -c1-160 | sed 's/^/   /'
  [ "$h2" = 200 ] && [ $((c2-c1)) -gt 0 ] || { fails=$((fails+1)); log "   -> NO WARM HIT on the second send ($1)"; }
  killsrv; }
run vision_31k "-m ${DS4_VISION_BASE:?} --vision ${DS4_VISION_ENCODER:?}" "$BIG"
run vision_small "-m ${DS4_VISION_BASE:?} --vision ${DS4_VISION_ENCODER:?}" "$SMALL"
run text_31k "-m ${DS4_TEXT_BASE:?}" "$BIG"
[ $fails = 0 ] && pass "all three second sends warm-hit" || fail "$fails of 3 second sends cold"
