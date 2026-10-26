# gates/lib/env.sh — every gate sources this first.  It loads the RIG file (the hosts, trees and model
# paths of the machines this checkout drives) and exports the tree locations the gates share.
#
#   DS4_BATTERY_ENV   the rig file; default $REPO/local/battery.env (local/ is gitignored).  The tracked
#                     template is gates/battery.env.example: copy it, fill it in, keep the same keys.
#   DS4_REPO / DS4_GATES / DS4_GATES_LIB   this checkout, its gates/ dir and gates/lib/ (helpers live here)
#
# Rig keys are referenced as ${DS4_X:?} by the gates, so a missing key fails BY NAME at the line that
# needs it ("DS4_BOX33: parameter null or not set"), never silently as an empty host or path.
# Neutral defaults (ports, tolerances, time budgets, cycle shapes) stay in the gates; only facts that
# differ from rig to rig belong in the env file.  Sourcing is idempotent.
[ -n "${DS4_GATES_ENV_LOADED:-}" ] && return 0 2>/dev/null
DS4_GATES_LIB=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DS4_GATES=$(cd "$DS4_GATES_LIB/.." && pwd)
DS4_REPO=$(cd "$DS4_GATES/.." && pwd)
DS4_BATTERY_ENV=${DS4_BATTERY_ENV:-$DS4_REPO/local/battery.env}
if [ -f "$DS4_BATTERY_ENV" ]; then
  # shellcheck disable=SC1090
  . "$DS4_BATTERY_ENV"
elif [ "${DS4_BATTERY_ENV_OPTIONAL:-0}" != 1 ]; then
  echo "$(basename "$0"): no rig file at $DS4_BATTERY_ENV (copy gates/battery.env.example there, or point DS4_BATTERY_ENV at yours)" >&2
fi
# defaults that are conventions, not rig facts (override in the env file if a rig differs)
DS4_BOX_LOCK_DIR=${DS4_BOX_LOCK_DIR:-/tmp/ds4_box_lock}
DS4_CHAIN_REG=${DS4_CHAIN_REG:-$HOME/.ds4_chains}
DS4_BOX15_TREE=${DS4_BOX15_TREE:-${DS4_BOX_TREE:-}}
export DS4_GATES_LIB DS4_GATES DS4_REPO DS4_BATTERY_ENV DS4_BOX_LOCK_DIR DS4_CHAIN_REG DS4_BOX15_TREE
export DS4_BOX33 DS4_BOX15 DS4_BOX33_LAN DS4_BOX15_LAN DS4_BOX_TREE DS4_BOX_SCRATCH DS4_GGUF_DIR DS4_GGUF_VISION_DIR \
       DS4_TEXT_BASE DS4_VISION_BASE DS4_VISION_ENCODER DS4_ORACLE_TREE DS4_VISION_ORACLE_RUN DS4_TEB_DIR DS4_BOX_TREE_B 2>/dev/null
DS4_GATES_ENV_LOADED=1
# rig KEY...: assert the named keys are set (for gates that want the check up front rather than at first use)
rig(){ local v; for v in "$@"; do [ -n "${!v:-}" ] || { echo "$(basename "$0"): rig key $v is unset in $DS4_BATTERY_ENV (see gates/battery.env.example)" >&2; exit 2; }; done; }
