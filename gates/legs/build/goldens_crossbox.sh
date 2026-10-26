#!/usr/bin/env bash
# battery: goldens_crossbox tier=1 stop=1 server=15 driver=33 trigger=always est=3 seq=30
# verdict: ^\[[0-9:]+\] GOLDENS_XBOX_(PASS|FAIL)
# catches: the recorded top-k/logit goldens (0731 text base + Vision base) replayed on the SECOND box's synced binary: a build, arch or box drift between the two GB10s that the .33 units cannot see
# cmd: bash gates/legs/build/goldens_crossbox.sh
VERDICT_TAG=GOLDENS_XBOX; . "$(dirname "$0")/../../lib/verdict.sh"
R=${RB_SERVER:-${R:-${DS4_BOX15:?}}}; TREE=${RB_TREE:-${BINDIR:-${DS4_BOX15_TREE:?}}}; VBASE=${RB_VBASE:-${VBASE:-${DS4_VISION_BASE:?}}}
OUT=${RB_OUT:-${OUT:-/tmp/goldens_crossbox_$$}}; mkdir -p "$OUT"
for g in "tests/test-vectors/local-golden-0731.vec:" "tests/test-vectors/flash-vision-exp/local-golden-vision-exp.vec:DS4_TEST_MODEL=$VBASE"; do
  fx=${g%%:*}; envx=${g#*:}
  ssh -o BatchMode=yes "$R" "cd $TREE && env $envx DS4_TEST_LOCAL_GOLDEN_FILE=$fx timeout 1800 ./ds4_test --local-golden-vectors" > "$OUT/$(basename "$fx").log" 2>&1
  grep -q "top20_max_abs=0" "$OUT/$(basename "$fx").log" && log "$fx: top20_max_abs=0" || count_fail "$fx: no exact golden (see $OUT/$(basename "$fx").log)"
done
finish "both goldens exact on .$R"
