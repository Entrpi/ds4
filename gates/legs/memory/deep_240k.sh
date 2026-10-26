#!/usr/bin/env bash
# battery: deep_240k tier=1 stop=1 server=15 driver=33 trigger=always est=7 seq=50
# verdict: DEEP-CTX GATE: (ALL STAGES PASS|FAIL)
# catches: the >8192-row comp tiers and the deep bank fit at a 524288-context boot with a ~216k-token needle prompt, subagent and a real second turn (12k never engages them: COMP-RATIO law), plus fork placement at depth
# cmd: bash gates/legs/memory/deep_240k.sh
. "$(dirname "$0")/../../lib/env.sh"
export DC_GATE_HOST=${RB_SERVER:-${DC_GATE_HOST:-${DS4_BOX15:?}}} BINDIR=${RB_TREE:-${BINDIR:-${DS4_BOX15_TREE:?}}} GGUF=${RB_GGUF:-${GGUF:-${DS4_GGUF_VISION_DIR:?}}} MTP=
ENC=${RB_ENC:-${ENC:-${DS4_VISION_ENCODER:?}}}; OUT=${RB_OUT:-${OUT:-/tmp/deep_240k_$$}}
CTX=524288 ORCH_TOK=240000 SUB_TOK=1000 ZEROCONF=1 DC_GATE_OUT="$OUT/deep" EXTRA_ARGS="--vision $ENC" exec bash "$DS4_GATES/legs/memory/deep_ctx.sh"
