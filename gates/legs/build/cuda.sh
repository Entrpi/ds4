#!/usr/bin/env bash
# battery: build_cuda tier=0 stop=1 server=33 driver=33 trigger=always est=5 seq=30
# verdict: ^\[[0-9:]+\] BUILD_CUDA_(PASS|FAIL)
# catches: the GB10 CUDA shape's quiet build with the native arch (cuobjdump sm_121a: a generic-arch link is the wrong binary) and the source md5 receipt the cross-box legs compare against
# cmd: bash gates/legs/build/cuda.sh
VERDICT_TAG=BUILD_CUDA; . "$(dirname "$0")/../../lib/verdict.sh"
OUT=${RB_OUT:-${OUT:-/tmp/build_cuda_$$}}; mkdir -p "$OUT"; cd "$DS4_REPO" || fail "no repo"
export PATH=/usr/local/cuda/bin:$PATH
make -j6 cuda-spark ds4_test > "$OUT/make.log" 2>&1; rc=$?; w=$(grep -c "warning:" "$OUT/make.log")
a=$(cuobjdump ds4_cuda.o 2>/dev/null | grep -m1 -o "sm_[0-9a-z]*"); md5sum ds4.c ds4_cuda.cu ds4_server.c > "$OUT/md5.txt"
[ $rc = 0 ] && [ "$w" = 0 ] && [ "$a" = sm_121a ] && pass "warnings=0 arch=$a" || fail "rc=$rc warnings=$w arch=${a:-none} (see $OUT/make.log)"
