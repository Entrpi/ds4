#!/usr/bin/env bash
# battery: toolslip_remoff tier=3 stop=0 server=15 driver=33 trigger="reminder or tool-rendering code changed" est=30 seq=30
# verdict: TOOLCALL-DEPTH GATE: (ALL DEPTHS PASS|FAIL)
# catches: the same depth gate with the tool-call reminder OFF: proves the reminder is an aid, not a crutch hiding a rendering regression
# cmd: bash gates/legs/serving/toolslip_remoff.sh
. "$(dirname "$0")/../../lib/env.sh"
export TG_HOST=${RB_SERVER:-${TG_HOST:-${DS4_BOX15:?}}} BINDIR=${RB_TREE:-${BINDIR:-${DS4_BOX15_TREE:?}}} GGUF=${RB_GGUF:-${GGUF:-${DS4_GGUF_VISION_DIR:?}}} TG_KEEP="${RB_OUT:-${OUT:-/tmp/toolslip_remoff_$$}}/jsonl" TG_EXTRA_ENV="DS4_TOOL_CALL_REMINDER=0"
exec bash "$DS4_GATES/legs/serving/toolcall_depth.sh"
