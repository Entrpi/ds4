#!/usr/bin/env bash
# gates/bin/rearm_watchers.sh — after a Claude Code process restart (which orphans every background
# waiter and Monitor), print the exact watchdog + fallback-waiter commands for every registered chain that
# is not finished.  Chains register themselves in ~/.ds4_chains/<name>.env (the orchestrator does it at
# launch; hand chains can, see the format) and are dropped when their sentinel is seen.
# Registry entry (shell assignments):  ALIAS=<driver ssh alias|local>  RD=<output dir>  PGREP=<bracketed pattern>
#   [SERVER=<server alias>] [EXPECT_SERVER=0|1] [AUX_ALIAS=.. AUX_DIR=..] [STALL_MIN=..] [SENT=<sentinel glob>]
#   [CAP_MIN=<fallback waiter cap in minutes>] [NOTE=<free text>] [LOG=<chain log name>] [HB=<heartbeat file glob>]
# Usage: bash gates/bin/rearm_watchers.sh [--prune]   (--prune deletes entries whose sentinel exists)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../lib/env.sh"; REG=$DS4_CHAIN_REG; mkdir -p "$REG"
prune=0; [ "${1:-}" = --prune ] && prune=1
probe(){ local alias=$1 rd=$2 sent=$3 pg=$4
  local cmd="cd '$rd' 2>/dev/null || { echo NODIR; exit 0; }; if ls $sent >/dev/null 2>&1; then echo DONE; else echo LIVE; fi; [ -n '$pg' ] && { pgrep -f '$pg' | wc -l | tr -d ' '; } || echo -1; true"
  if [ "$alias" = local ]; then bash -c "$cmd"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "$alias" "$cmd" 2>/dev/null || echo UNREACHABLE; fi; }
n=0
for f in "$REG"/*.env; do [ -e "$f" ] || continue
  ALIAS=""; RD=""; PGREP=""; SERVER=""; EXPECT_SERVER=0; AUX_ALIAS=""; AUX_DIR=""; STALL_MIN=""; SENT=leg.DONE; CAP_MIN=360; NOTE=""; LOG=leg.log; HB=""
  # shellcheck disable=SC1090
  . "$f"
  name=$(basename "$f" .env); st=$(probe "$ALIAS" "$RD" "$SENT" "$PGREP"); state=$(printf '%s\n' "$st" | head -1); alive=$(printf '%s\n' "$st" | sed -n 2p)
  case "$state" in
    DONE) echo "# $name: DONE (sentinel present)${NOTE:+ — $NOTE}"; [ $prune = 1 ] && rm -f "$f" && echo "#   pruned $f"; continue;;
    NODIR) echo "# $name: output dir missing on $ALIAS ($RD)${NOTE:+ — $NOTE}"; continue;;
    UNREACHABLE) echo "# $name: $ALIAS unreachable, re-arm anyway:";;
    *) echo "# $name: LIVE (driver procs ${alive})${NOTE:+ — $NOTE}";;
  esac
  n=$((n+1))
  env="W_ALIAS=$ALIAS W_RD=$RD W_PGREP='$PGREP' W_SENT='$SENT' W_LOG=$LOG"; [ -n "$HB" ] && env="$env W_HB='$HB'" || env="$env W_HB=none"
  [ -n "$SERVER" ] && env="$env W_SERVER=$SERVER W_EXPECT_SERVER=$EXPECT_SERVER"
  [ -n "$AUX_ALIAS" ] && env="$env W_AUX_ALIAS=$AUX_ALIAS W_AUX_DIR=$AUX_DIR"
  [ -n "$STALL_MIN" ] && env="$env W_STALL_MIN=$STALL_MIN"
  echo "Monitor (persistent): cd $DS4_REPO && $env bash gates/bin/chain_watch.sh"
  if [ "$ALIAS" = local ]; then t="test -e $RD/$SENT"; else t="ssh -o BatchMode=yes -o ConnectTimeout=10 $ALIAS 'ls $RD/$SENT >/dev/null 2>&1'"; fi
  echo "Waiter (background Bash, cap ${CAP_MIN} min): n=0; until $t; do n=\$((n+1)); [ \$n -gt $CAP_MIN ] && { echo 'FALLBACK TIMEOUT $name'; exit 2; }; sleep 60; done; echo '$name DONE'"
  echo
done
[ $n = 0 ] && echo "# no live chains registered in $REG"
exit 0
