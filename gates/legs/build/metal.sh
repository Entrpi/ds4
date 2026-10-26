#!/usr/bin/env bash
# battery: build_metal tier=0 stop=1 server=mac driver=mac trigger=always est=4 seq=20
# verdict: ^\[[0-9:]+\] BUILD_METAL_(PASS|FAIL)
# catches: the Metal shape's quiet build and its unit suite (Metal-only kernels, Apple toolchain warnings) on the workstation
# cmd: bash gates/legs/build/metal.sh
VERDICT_TAG=BUILD_METAL; . "$(dirname "$0")/../../lib/verdict.sh"
OUT=${RB_OUT:-${OUT:-/tmp/build_metal_$$}}; mkdir -p "$OUT"; cd "$DS4_REPO" || fail "no repo"
make -j8 > "$OUT/make.log" 2>&1 && make ds4_test >> "$OUT/make.log" 2>&1; rc=$?; w=$(grep -c "warning:" "$OUT/make.log")
./ds4_test --server > "$OUT/units_server.log" 2>&1; u=$?
[ $rc = 0 ] && [ "$w" = 0 ] && [ $u = 0 ] && pass "warnings=0 units=ok" || fail "rc=$rc warnings=$w units=$u (see $OUT)"
