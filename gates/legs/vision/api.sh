#!/bin/bash
# Ported into gates/legs/vision/ on 2026-09-04 (release-process §3.0): the Vision-Exp release-battery member.
# Runs ON the GB10 box-side (the tree is the rsync mirror at $T); helper scripts live beside this file.
# Track B inc 7 API gate (runs ON .33, detached): ds4-server on the Vision base
# with the encoder sidecar serves image requests on the three chat surfaces.
#  1. build fork + units;  2. cont probe with the API prompt set (serial seed
#  dumps);  3. one server boot (seed-logit dump ON);  4. identity twin through
#  /v1/chat/completions (token ids vs the V2 oracle + the serial fork run);
#  5. three surfaces same prompt;  6. repeat + text-after;  7. refusals by
#  name;  8. concurrency (7 text + image, twin, vs 8 text twin);  9. V4 needle
#  battery through each surface;  10. /metrics;  11. stop by PID; compare.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HERE=$(cd "$(dirname "$0")" && pwd)
T=${BINDIR:-${DS4_BOX_TREE:?}}
VBASE=${VBASE:-${DS4_VISION_BASE:?}}
SIDE=${ENC:-${DS4_VISION_ENCODER:?}}
IMG=$T/tests/test-vectors/flash-vision-exp/images
NEEDLE=$T/tests/test-vectors/flash-vision-exp/needle
ORACLE=${ORACLE:-${DS4_VISION_ORACLE_RUN:?}}   # the V2 oracle run dir (prompt lengths); a data dependency of the API leg
RD=${RD:-${DS4_BOX_SCRATCH:?}/vision_api_$(date +%Y%m%d_%H%M)}; mkdir -p "$RD"
OUT=$RD/leg.log; SENT=$RD/leg.DONE; rm -f "$SENT"; : > "$OUT"
PORT=${PORT:-8090}; CTX=${CTX:-8192}
export API_BASE=http://127.0.0.1:$PORT RD IMG NEEDLE
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
die(){ log "FAIL: $*"; echo "API_LEG_FAIL" >> "$OUT"; stop_server; touch "$SENT"; exit 1; }
stop_server(){ for x in $(pgrep -x ds4-server); do kill "$x" 2>/dev/null; done; sleep 3; for x in $(pgrep -x ds4-server); do kill -9 "$x" 2>/dev/null; done; }
export PATH=/usr/local/cuda/bin:$PATH
[ -z "$(pgrep -x ds4-server)" ] || die "a ds4-server is up"
echo "api leg $(date +%Y%m%d_%H%M) out=$RD"

log "== build fork ($T)"
( cd "$T" && make -j6 cuda-spark > "$RD/build.log" 2>&1 && make ds4_test CUDA_ARCH=sm_121 >> "$RD/build.log" 2>&1 && make tests/vision_cont_probe CUDA_ARCH=sm_121 >> "$RD/build.log" 2>&1; echo "rc=$?" >> "$RD/build.log" )
tail -1 "$RD/build.log" | grep -q "rc=0" || die "fork build ($RD/build.log)"
log "   build: $(tail -1 "$RD/build.log") warnings=$(grep -c warning "$RD/build.log")"
( cd "$T" && ./ds4_test --server --vision-cont --vision-compact --vision-cache --vision-prompt > "$RD/units.log" 2>&1; echo "rc=$?" >> "$RD/units.log" )
log "   units: $(tail -1 "$RD/units.log") $(grep -E '^[a-z-]+: ' "$RD/units.log" | tr '\n' ' ')"
tail -1 "$RD/units.log" | grep -q "rc=0" || die "units"

# (The former "cont probe" step is gone: it passed 10 prompts to an 8-prompt probe and had exited 2 on every run,
#  which nothing noticed because nothing read its dump; the seed member runs its own 8-prompt probe.  09-05.)
log "== boot ds4-server (Vision base + sidecar, ctx $CTX, port $PORT, no spec/mtp)"
( cd "$T" && env DS4_CONT_DUMP_SEED_LOGITS="$RD/server_seed.bin" ./ds4-server -m "$VBASE" --vision "$SIDE" -c "$CTX" --host 127.0.0.1 --port "$PORT" --no-spec --no-mtp > "$RD/server.log" 2>&1 < /dev/null & )
for i in $(seq 1 180); do sleep 5; code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null); [ "$code" = "200" ] && break; done
[ "$code" = "200" ] || die "server not ready ($RD/server.log)"
log "   server up after ~$((i*5))s pid=$(pgrep -x ds4-server | tr '\n' ' ') $(grep -c 'vision' "$RD/server.log") vision lines at boot"
C="python3 $HERE/api_client.py"
log "== identity twin, batched lane (streaming chat + token ids)"; $C identity bapi1 batched | tee -a "$OUT"; $C identity bapi2 batched | tee -a "$OUT"
$C twincheck bapi1 bapi2 bapi3 batched | tee -a "$OUT"     # third draw for split twins: first-draw flake (2/3 agree) vs a real split
log "== identity twin, serial lane (non-streaming chat + token ids)"; $C identity api1 serial | tee -a "$OUT"; $C identity api2 serial | tee -a "$OUT"
$C twincheck api1 api2 api3 serial | tee -a "$OUT"
log "== three surfaces, one prompt";           $C surfaces | tee -a "$OUT"
log "== repeat + text after (batched)";        $C repeat batched | tee -a "$OUT"
log "== refusals by name";                     $C refusals | tee -a "$OUT"
log "== concurrency (batched): 7 text + image (twin), 8 text (twin)"
$C concurrency conc_img1 img batched | tee -a "$OUT"; $C concurrency conc_img2 img batched | tee -a "$OUT"
$C concurrency conc_txt1 txt batched | tee -a "$OUT"; $C concurrency conc_txt2 txt batched | tee -a "$OUT"
log "== V4 needles through each surface"
for s in chat anthropic responses; do $C needles $s | tee -a "$OUT"
  ( cd "$T" && python3 tests/vision_needle_score.py --manifest "$NEEDLE/manifest.json" "$RD/needle_$s.replies.jsonl" > "$RD/needle_$s.score.txt" 2>&1 || python3 tests/vision_needle_score.py "$RD/needle_$s.replies.jsonl" > "$RD/needle_$s.score.txt" 2>&1 )
  log "   score $s: $(tail -5 "$RD/needle_$s.score.txt" | tr '\n' '|' | cut -c1-200)"; done
log "== metrics"; curl -s "http://127.0.0.1:$PORT/metrics" > "$RD/metrics.txt"; grep -i "vision" "$RD/metrics.txt" | head -12 | sed 's/^/   /' | tee -a "$OUT"
log "== stop server"; stop_server; log "   servers left: $(pgrep -cx ds4-server)"
log "   tells: image_requests=$(grep -c 'image request images=' "$RD/server.log") visual_admits=$(grep -c 'cont visual admission' "$RD/server.log") lane=$(grep -c 'vision encoder lane' "$RD/server.log") engaged=$(grep -c 'image prefill engaged' "$RD/server.log") kept_whole=$(grep -c 'kept whole' "$RD/server.log") warn=$(grep -ci 'warning' "$RD/server.log") err=$(grep -ci 'error' "$RD/server.log") cont_rejects=$(grep -E '^ds4_cont_admit_rejects_total' "$RD/metrics.txt" | awk '{print $2}') serial_fallbacks=$(grep -c 'served=0 fallback=1' "$RD/server.log")"

log "== compare"
python3 - "$RD" "$ORACLE" <<'PY' | tee -a "$OUT"
import json, sys, os
rd, od = sys.argv[1], sys.argv[2]
EOS = 1
def jl(p):
    return [json.loads(l) for l in open(p) if l.strip().startswith("{")] if os.path.exists(p) else []
def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n
def key(d): return os.path.basename(d["image"]) if d["image"] != "-" else "text#" + str(len(d.get("after", "")) + len(d.get("before", "")))
ora = {}
for d in jl(os.path.join(od, "oracle1.jsonl")):
    g = d["generated"]; g = g[:g.index(EOS)] if EOS in g else g
    ora[os.path.basename(d["image"]) if d["image"] != "-" else "text#%d" % d["prompt_len"]] = (d["prompt_len"], g)
fork = {}
for d in jl(os.path.join(od, "fork1.jsonl")):
    g = d["generated"]; g = g[:g.index(EOS)] if EOS in g else g
    fork[os.path.basename(d["image"]) if d["image"] != "-" else "text#%d" % d["prompt_len"]] = (d["prompt_len"], g)
fails = 0; lane_mixed = 0; flakes = 0
for lane, n1, n2, n3 in (("batched", "bapi1", "bapi2", "bapi3"), ("serial", "api1", "api2", "api3")):
  api1, api2 = jl(os.path.join(rd, n1 + ".jsonl")), jl(os.path.join(rd, n2 + ".jsonl"))
  third = {key(d): d for d in jl(os.path.join(rd, n3 + ".jsonl"))}
  print(f"-- {lane} lane")
  print(f"{'case':26s} {'st':>3s} {'ptok':>5s} {'o-plen':>6s} {'twin':>6s} {'vs oracle lcp/len':>18s} {'vs serial fork':>16s}")
  if len(api1) != 10 or len(api2) != 10: fails += 1
  for a, b in zip(api1, api2):
      k = key(a); ids = a.get("token_ids") or []; ids2 = b.get("token_ids") or []
      twin = "ok" if ids == ids2 else f"lcp{lcp(ids, ids2)}"
      if lane == "serial":
          # return_token_ids is an SSE-only extension (README "Chat-completion SSE accepts return_token_ids"): the
          # non-streaming request ROUTES to the serial lane but its response carries no ids, so the serial twin is
          # the response TEXT (review 09-05, K3 1: the id compare here had been vacuous, both lists empty)
          twin = "ok(text)" if (a.get("text") and a.get("text") == b.get("text")) else "difftext"
      elif not ids or not ids2: twin = "noids"      # the batched lane promises ids: two empty lists are not a twin
      fb1, fb2 = a.get("fallback", False), b.get("fallback", False)
      if fb1 != fb2: twin = "lane-mix"          # one twin refused by the cont lane (memory floor) -> disclosed, not a twin fail
      elif fb1: twin += "/s"                    # both served on the serial lane after a refusal (same lane: still a twin)
      if twin.startswith("lcp") or twin == "difftext":   # a third draw decides: 2 of 3 agree = first-draw near-tie flake (disclosed)
          t = third.get(k); tids = (t or {}).get("token_ids") or []
          agree = (tids == ids or tids == ids2) if lane != "serial" else (t.get("text") in (a.get("text"), b.get("text")))
          if t and t.get("status") == 200 and t.get("fallback", False) == fb1 and agree: twin = "flake"; flakes += 1   # the third draw must be on the SAME lane (review 09-05 F5)
      # text prompts key by prompt_len on the oracle side: match by prompt_tokens
      if k.startswith("text#"):
          cand = [v for kk, v in ora.items() if kk.startswith("text#") and v[0] == a.get("prompt_tokens")]
          o = cand[0] if cand else None
          candf = [v for kk, v in fork.items() if kk.startswith("text#") and v[0] == a.get("prompt_tokens")]
          f = candf[0] if candf else None
      else:
          o = ora.get(k); f = fork.get(k)
      if a["status"] != 200 or b["status"] != 200 or twin.startswith("lcp") or twin in ("noids", "difftext"): fails += 1   # both draws must serve and agree, whatever the lane (review 09-05 F4, K3 1)
      if twin == "lane-mix": lane_mixed += 1
      ocol = "n/a" if not o else f"{lcp(ids, o[1])}/{len(o[1])}" + ("" if ids[:len(o[1])] == o[1] else "*")
      fcol = "n/a" if not f else f"{lcp(ids, f[1])}/{len(f[1])}" + ("" if ids[:len(f[1])] == f[1] else "*")
      oplen = "n/a" if not o else (str(o[0]) + ("" if o[0] == a.get("prompt_tokens") else "!"))
      if o and o[0] != a.get("prompt_tokens"): fails += 1
      print(f"{k:26s} {a['status']:>3d} {str(a.get('prompt_tokens')):>5s} {oplen:>6s} {twin:>6s} {ocol:>18s} {fcol:>16s}")
# concurrency
def rows(n): return jl(os.path.join(rd, n + ".jsonl"))
ci1, ci2, ct1, ct2 = rows("conc_img1"), rows("conc_img2"), rows("conc_txt1"), rows("conc_txt2")
def twin(x, y): return all(r["status"] == 200 and r["token_ids"] == s["token_ids"] for r, s in zip(x, y)) and len(x) == len(y) == 8
print(f"concurrency: img-twin={'IDENTICAL' if twin(ci1, ci2) else 'DIFF'} txt-twin={'IDENTICAL' if twin(ct1, ct2) else 'DIFF'} "
      f"text rows beside image vs beside text: " + " ".join(f"{lcp(r['token_ids'] or [], s['token_ids'] or [])}/{len(s['token_ids'] or [])}" for r, s in zip(ci1[:7], ct1[:7])))
if not twin(ci1, ci2) or not twin(ct1, ct2): fails += 1
sf = json.load(open(os.path.join(rd, "surfaces.json"))) if os.path.exists(os.path.join(rd, "surfaces.json")) else {"verdict": "missing"}
print("surfaces:", sf["verdict"])
if not sf["verdict"].startswith("IDENTICAL"): fails += 1
rp = rows("repeat"); rp_ok = len(rp) == 3 and all(r["status"] == 200 for r in rp)   # every repeat request must serve, whatever the lane (review 09-05 F4)
rp_mixed = rp_ok and rp[0].get("fallback") != rp[1].get("fallback")
rp_same = rp_ok and rp[0]["token_ids"] == rp[1]["token_ids"]
print("repeat:", "LANE-MIXED (disclosed)" if rp_mixed else "IDENTICAL" if rp_same else "DIFF")
if not rp_same and not rp_mixed: fails += 1
fbs = sum(1 for n in ("bapi1", "bapi2", "api1", "api2") for r in jl(os.path.join(rd, n + ".jsonl")) if r.get("fallback")) + sum(1 for r in rp if r.get("fallback")) + sum(1 for s in ("chat", "anthropic", "responses") if sf.get(s, {}).get("fallback"))
print(f"cont-lane refusals -> serial fallbacks among the compared requests: {fbs} (lane-mixed twins: {lane_mixed}); a refusal is the governor's floor protection, disclosed here, never a wiring verdict")
print(f"first-draw twin flakes (a third draw agreed with one of the two): {flakes}; three distinct draws would be a twin FAIL")
rf = rows("refusals"); nok = sum(1 for r in rf if r["ok"]); print(f"refusals: {nok}/{len(rf)}")
if len(rf) == 0 or nok != len(rf): fails += 1   # an empty/absent refusals file is a FAIL, not a vacuous pass (review 09-05, GLM 2)
for s in ("chat", "anthropic", "responses"):
    p = os.path.join(rd, f"needle_{s}.score.txt")
    tot = [l for l in open(p)] if os.path.exists(p) else []
    t = [l.strip() for l in tot if l.startswith("total")]
    print(f"needles {s}: {t[0] if t else 'missing'}")
    if not t or not t[0].startswith("total: 24/24"): fails += 1
print("API_LEG_" + ("PASS" if fails == 0 else f"FAIL ({fails} checks)"))
PY
log "== LEG_DONE"; touch "$SENT"
grep -q "^API_LEG_PASS" "$OUT" && exit 0 || exit 1   # rc follows the verdict (review 09-05, Grok 5)
