#!/usr/bin/env bash
# battery: toolslip tier=2 stop=0 server=15 driver=33 trigger=always est=30 seq=10
# verdict: TOOLCALL-DEPTH GATE: (ALL DEPTHS PASS|FAIL)
# catches: the tool-call protocol slipping with depth: a capture-shaped tool conversation grown to 112k prompt tokens, 2 reasoning arms x 3 seeds x 6 depths, prompt_tokens equal across arms, one shipped-defaults run per release (3l ONE-RUN law)
# cmd: bash gates/legs/serving/toolslip.sh
. "$(dirname "$0")/../../lib/env.sh"
export TG_HOST=${RB_SERVER:-${TG_HOST:-${DS4_BOX15:?}}} BINDIR=${RB_TREE:-${BINDIR:-${DS4_BOX15_TREE:?}}} GGUF=${RB_GGUF:-${GGUF:-${DS4_GGUF_VISION_DIR:?}}} TG_KEEP="${RB_OUT:-${OUT:-/tmp/toolslip_$$}}/jsonl"
exec bash "$DS4_GATES/legs/serving/toolcall_depth.sh"
