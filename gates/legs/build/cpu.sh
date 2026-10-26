#!/usr/bin/env bash
# battery: build_cpu tier=0 stop=1 server=mac driver=mac trigger=always est=2 seq=10
# verdict: ^\[[0-9:]+\] BUILD_CPU_(PASS|FAIL)
# catches: a backend-neutral symbol trap: a type or symbol that exists only under a GPU guard, so the CPU shape stops LINKING (the class was latent for five increments while this leg was not first)
# cmd: bash gates/legs/build/cpu.sh
VERDICT_TAG=BUILD_CPU; . "$(dirname "$0")/../../lib/verdict.sh"
OUT=${RB_OUT:-${OUT:-/tmp/build_cpu_$$}}; mkdir -p "$OUT"; cd "$DS4_REPO" || fail "no repo"
make cpu > "$OUT/make.log" 2>&1; rc=$?; w=$(grep -c "warning:" "$OUT/make.log")
[ $rc = 0 ] && [ "$w" = 0 ] && pass "linked warnings=0" || fail "rc=$rc warnings=$w (see $OUT/make.log)"
