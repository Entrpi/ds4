#!/usr/bin/env bash
# battery: memtruth_fast tier=2 stop=0 server=33 driver=15 trigger="admission/census/ledger/serial-lane changed" est=40 seq=10
# verdict: ^\[[0-9:]+\] MT_FAST_(PASS|FAIL)
# catches: admission and census ACCOUNTING regressions at 16k shapes: tranche credit + extension walk, serial lease lifecycle, idle reaper, ctx-aware bank grant + headroom regimes, pool census at reap, --no-serial typed 503s
# cmd: bash gates/legs/memory/memtruth_fast.sh
VERDICT_TAG=MT_FAST; . "$(dirname "$0")/../../lib/verdict.sh"
export R=${RB_SERVER:-${R:-${DS4_BOX33:?}}} BINDIR=${RB_TREE:-${BINDIR:-${DS4_BOX_TREE:?}}}
OUT=${RB_OUT:-${OUT:-/tmp/memtruth_fast_$$}}; mkdir -p "$OUT"
for g in credit_tranche serial_lease serial_reap defaults_derivation pool_truth noserial_optin; do
  OUT="$OUT/$g" bash "$DS4_GATES/legs/memory/$g.sh" > "$OUT/$g.out" 2>&1 && log "$g: PASS" || count_fail "$g (see $OUT/$g.out)"
done
finish "6 gates"
