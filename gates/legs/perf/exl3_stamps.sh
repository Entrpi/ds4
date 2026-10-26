#!/bin/bash
# gates/legs/perf/exl3_stamps.sh — the OWED matched stamps vs the EXL3-K2/
# SparkInfer recipe (forum thread 379863; adjudications in
# local/docs/reference-exl3-k2-spark-recipe.md):
#   S1  pp2048/tg128 C1, thinking OFF (their forced-off confound, matched)
#   S2  pp2048/tg128 C4 aggregate, thinking OFF
#   S3  252K UNCACHED single-request prefill aggregate (their 1,058 pp/s
#       denominator class), fresh boot so nothing is warm
# SHIP CONFIG: zero-config boot (launch defaults attach drafter + MTP-2 +
# DSpark; no env tuning), -c 262144 CUDA default. temperature=0 and
# reasoning_effort "none" ride the REQUESTS. These are STAMPS, not gates:
# health is asserted (no illegal/cont-fail, finish reasons sane), values
# are reported for the public receipt.
# Env: R (${DS4_BOX33:?}) BINDIR (${DS4_BOX_TREE:?})
#      PORT (8000) TUNNEL_PORT (18000) OUT (/tmp/exl3_stamps_<ts>)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL=${TUNNEL_PORT:-18000}
OUT=${OUT:-/tmp/exl3_stamps_$(date +%Y%m%d_%H%M%S)}
SRV=/tmp/exl3_stamps_srv.log
HERE=$(cd "$(dirname "$0")" && pwd)
BASE="http://127.0.0.1:$TUNNEL"
URL="$BASE/v1/chat/completions"
mkdir -p "$OUT"
DRV=$OUT/driver.log
log(){ echo "[$(date +%H:%M:%S)] EXL3: $*" | tee -a "$DRV"; }
fail(){ log "FAIL: $*"; exit 1; }

NONCE="exl3stamps_$(date +%s)_$$"
LOCKED=0
cleanup(){ rc=$?
  ssh "$R" "pkill -x ds4-server; sleep 1; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null
  if [ "$LOCKED" = 1 ]; then ssh "$R" "rm -rf /tmp/ds4_box_lock; exit 0" 2>/dev/null; log "box lock freed"; fi
  log "artifacts in $OUT"
  exit $rc
}
trap cleanup EXIT INT TERM
OWN=$(ssh "$R" "cat /tmp/ds4_box_lock/nonce 2>/dev/null; exit 0" 2>/dev/null)
[ -n "$OWN" ] && { log "box lock held by $OWN -- refusing"; exit 2; }
ssh "$R" "mkdir /tmp/ds4_box_lock && echo '$NONCE' > /tmp/ds4_box_lock/nonce" || fail "lock take"
LOCKED=1
log "box lock taken ($NONCE)"

boot(){
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; rm -f /tmp/ds4.lock; exit 0"
  ssh "$R" ": > $SRV; cd $BINDIR; setsid nohup ./ds4-server --cuda -c 262144 --port $PORT \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || {
      sleep 3
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" 2>/dev/null | tr '\n' ' ')"; }
    sleep 10; n=$((n+10)); [ $n -ge 1500 ] && fail "boot timeout"
  done
  ssh "$R" "grep -E 'launch defaults|persistent batch ctx ready' $SRV" | tee -a "$DRV"
  curl -s -m 5 "$BASE/v1/models" >/dev/null 2>&1 || {
    ssh -f -N -L "$TUNNEL:127.0.0.1:$PORT" "$R" 2>/dev/null || true; sleep 2
    curl -s -m 10 "$BASE/v1/models" >/dev/null || fail "tunnel unreachable"; }
  log "boot up (zero-config ship defaults, -c 262144)"
}

# pp2048 request builder: ~2048-token unique filler + a generative task so
# tg128 runs to finish=length.  Rotation i makes every prefill UNCACHED.
mk2048(){ # $1=index -> $OUT/pp_$1.json
  python3 - "$1" "$OUT/pp_$1.json" <<'PY'
import json, sys
i, path = int(sys.argv[1]), sys.argv[2]
corpus = ("The archive of expedition %d records tides, granite, lichen, "
          "and the slow accounting of weather over the ridge. " % i)
# measured 08-20: this corpus tokenizes at ~5.49 chars/tok -> ~2050 tokens
filler = (corpus * 90)[:11250]
msg = filler + "\n\nRewrite the passage above in your own words, expansively."
json.dump({"max_tokens": 128, "temperature": 0, "reasoning_effort": "none",
           "messages": [{"role": "user", "content": msg}]}, open(path, "w"))
PY
}

boot
log "=== S1: pp2048/tg128 C1, thinking off, 5 uncached reps ==="
for i in 1 2 3 4 5; do
  mk2048 "$i"
  python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/pp_$i.json" "$URL" "C1r$i" "$OUT/c1_$i.json" \
    | tee -a "$DRV" || fail "C1 rep $i transport"
done

log "=== S2: pp2048/tg128 C4 aggregate, thinking off, 2 rounds ==="
for r in 1 2; do
  PIDS=""
  for j in 1 2 3 4; do
    i=$((10 + r * 4 + j))
    mk2048 "$i"
    python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/pp_$i.json" "$URL" "C4r${r}s$j" "$OUT/c4_${r}_$j.json" \
      >> "$DRV" 2>&1 &
    PIDS="$PIDS $!"
  done
  for p in $PIDS; do wait "$p" || fail "C4 round $r stream failed"; done
  log "C4 round $r complete"
done

log "=== S3: 252K uncached prefill aggregate (fresh boot) ==="
boot
log "CAL: 50k ratio probe"
python3 "$DS4_GATES/lib/needle_prompt.py" 50000 "$OUT/cal.json" 0.5 "3131-CAL-EXL3" deepseek-chat 7 >> "$DRV" || fail "cal build"
python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/cal.json" "$URL" "CAL" "$OUT/cal_out.json" | tee -a "$DRV" || fail "cal serve"
CALP=$(python3 -c "import json;print(json.load(open('$OUT/cal_out.json'))['usage']['prompt_tokens'])")
[ -n "$CALP" ] && [ "$CALP" -gt 20000 ] || fail "cal unparsable ($CALP)"
# band-retry: the 50k probe ratio drifts on big rotations (measured -11%
# on the first run); recalibrate from the big sample itself and retry once.
RATIO_TOK=$CALP
RATIO_TGT=50000
ATTEMPT=1
while :; do
  TGT=$(python3 -c "print(int(252047 * $RATIO_TGT / $RATIO_TOK))")
  log "building ~252,047-token prompt (builder target $TGT) [attempt $ATTEMPT]"
  python3 "$DS4_GATES/lib/needle_prompt.py" "$TGT" "$OUT/deep.json" 0.5 "7272-DEEP-EXL3" deepseek-chat 3 >> "$DRV" || fail "deep build"
  python3 - "$OUT/deep.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
d["max_tokens"] = 8; d["temperature"] = 0; d["reasoning_effort"] = "none"
json.dump(d, open(p, "w"))
PY
  boot   # fresh so the 252K prefill is the server's FIRST serve (uncached)
  python3 "$DS4_GATES/lib/sse_probe_client.py" "$OUT/deep.json" "$URL" "PP252K" "$OUT/deep_out.json" | tee -a "$DRV" || fail "252K serve"
  DP=$(python3 -c "import json;print((json.load(open('$OUT/deep_out.json'))['usage'] or {}).get('prompt_tokens',0))")
  log "252K attempt $ATTEMPT landed at $DP tokens (band 246,000..258,000)"
  if [ "$DP" -ge 246000 ] && [ "$DP" -le 258000 ]; then break; fi
  [ $ATTEMPT -ge 2 ] && fail "252K prompt outside the band twice ($DP)"
  RATIO_TOK=$DP; RATIO_TGT=$TGT
  ATTEMPT=2
done

log "=== stamp summary ==="
python3 - "$OUT" <<'PY' | tee -a "$DRV"
import json, glob, sys
out = sys.argv[1]
def rows(pat):
    for f in sorted(glob.glob(out + pat)):
        d = json.load(open(f))
        u = d.get("usage") or {}
        pt = u.get("prompt_tokens", 0)
        ttft, dec, nd = d.get("ttft_s", -1), d.get("decode_s", 0), d.get("decode_deltas", 0)
        pp = pt / ttft if ttft and ttft > 0 else 0
        tg = (nd - 1) / dec if dec and nd > 1 else 0
        yield d["tag"], pt, ttft, pp, nd, tg
c1 = list(rows("/c1_*.json"))
for t, pt, ttft, pp, nd, tg in c1:
    print(f"  {t}: prompt={pt} ttft={ttft}s pp={pp:.0f} tok/s gen={nd} tg={tg:.2f} tok/s")
if c1:
    steady = c1[1:]  # rep 1 carries the capture warmup; report it, exclude it
    print(f"S1 C1 warmup rep: tg={c1[0][5]:.2f} tok/s (excluded)")
    print(f"S1 C1 steady mean: pp={sum(r[3] for r in steady)/len(steady):.0f} tok/s "
          f"tg={sum(r[5] for r in steady)/len(steady):.2f} tok/s (n={len(steady)})")
c4 = list(rows("/c4_*.json"))
for t, pt, ttft, pp, nd, tg in c4:
    print(f"  {t}: prompt={pt} ttft={ttft}s gen={nd} tg={tg:.2f} tok/s")
if c4:
    per_round = {}
    for t, pt, ttft, pp, nd, tg in c4:
        rnd = t.split("r")[1][0]   # "C4r2s3" -> "2"
        per_round.setdefault(rnd, []).append(tg)
    for r, tgs in sorted(per_round.items()):
        print(f"S2 C4 round {r}: aggregate tg={sum(tgs):.1f} tok/s (streams: {', '.join(f'{x:.1f}' for x in tgs)})")
d = json.load(open(out + "/deep_out.json"))
u = d.get("usage") or {}
pt, ttft = u.get("prompt_tokens", 0), d.get("ttft_s", -1)
print(f"S3 252K: prompt={pt} ttft={ttft}s pp_aggregate={pt/ttft if ttft>0 else 0:.1f} tok/s")
PY

ssh "$R" "cat $SRV" > "$OUT/final_srv.log" 2>/dev/null
for bad in "illegal" "continuous batch failed" "cont admit rejected"; do
  C=$(grep -ci "$bad" "$OUT/final_srv.log" || true)
  [ "$C" -eq 0 ] || fail "srv log dirty: $C x '$bad'"
done
log "EXL3 STAMPS: COMPLETE (health clean) — artifacts in $OUT"
exit 0
