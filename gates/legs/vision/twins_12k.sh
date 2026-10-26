#!/usr/bin/env bash
# battery: twins_12k tier=1 stop=1 server=15 driver=33 trigger=always est=5 seq=40
# verdict: ^\[[0-9:]+\] TWINS12K_(PASS|FAIL)
# catches: "the feature perturbed the text path": two 12k orchestrator+subagent serving runs, encoder unbound vs bound, byte-compared on usage, answers and accept trajectories (a reservation that shifts the fit, a render change, an env default drift)
# cmd: bash gates/legs/vision/twins_12k.sh
VERDICT_TAG=TWINS12K; . "$(dirname "$0")/../../lib/verdict.sh"
export DC_GATE_HOST=${RB_SERVER:-${DC_GATE_HOST:-${DS4_BOX15:?}}} BINDIR=${RB_TREE:-${BINDIR:-${DS4_BOX15_TREE:?}}} GGUF=${RB_GGUF:-${GGUF:-${DS4_GGUF_VISION_DIR:?}}} MTP=
ENC=${RB_ENC:-${ENC:-${DS4_VISION_ENCODER:?}}}; OUT=${RB_OUT:-${OUT:-/tmp/twins_12k_$$}}; mkdir -p "$OUT"
DEEP="$DS4_GATES/legs/memory/deep_ctx.sh"
CTX=16384 ORCH_TOK=12000 SUB_TOK=1000 DC_GATE_OUT="$OUT/A" bash "$DEEP" > "$OUT/A.out" 2>&1
CTX=16384 ORCH_TOK=12000 SUB_TOK=1000 DC_GATE_OUT="$OUT/B" EXTRA_ARGS="--vision $ENC" bash "$DEEP" > "$OUT/B.out" 2>&1
python3 -c "import json,sys; f=lambda p: {k:v for k,v in json.load(open(p)).items() if not k.endswith('_s')}; sys.exit(0 if f(sys.argv[1])==f(sys.argv[2]) else 1)" "$OUT/A/orch_result.json" "$OUT/B/orch_result.json" || count_fail "usage differs (orch_result.json, timing keys ignored)"
strip(){ sed -E "s/[0-9]+\.[0-9]+s|[0-9.]+ tok\/s|[0-9.]+ ms\/tok|[0-9]+:[0-9]+:[0-9]+//g" "$1"; }
for f in orch.out sub.out turn2.out; do diff <(strip "$OUT/A/$f") <(strip "$OUT/B/$f") > /dev/null || count_fail "$f differs"; done
a=$(grep -ho "CONT_MTP_ACCEPT[^|]*" "$OUT/A"/*.log | sort | md5sum); b=$(grep -ho "CONT_MTP_ACCEPT[^|]*" "$OUT/B"/*.log | sort | md5sum)
[ "$a" = "$b" ] || count_fail "accept trajectories differ"
finish "usage+answers+accepts identical"
