#!/usr/bin/env bash
# Step 7 determinism probe.  Runs the same DS4 greedy generation N times
# under both DS4_CUDA_LAYER_GRAPHS=0 and =1 and reports MD5s.  Goal:
# all OFF runs equal; all ON runs equal; OFF == ON (for the parity gate).
#
# Empirical state at b46afc7..3b200c1 (Step 7 commit):
#   - OFF runs: deterministic (single MD5 across N runs).
#   - ON  runs: NON-deterministic across runs (different MD5 each time
#               and != OFF baseline).
#   - CUDA_LAUNCH_BLOCKING=1 doesn't fix the ON-mode non-determinism --
#               ruling out CUDA stream races.  Suspected root cause:
#               cudaPool allocation address dependence within the
#               captured graph (cudaMallocAsync from ggml_cuda_pool
#               inside ds4_mmq_q8_0_dense_vec) interacting with some
#               address-sensitive kernel path.
#
# Usage:  bash tests/cuda_layer_graph_determinism_probe.sh [N] [PROMPT]
#         N      -- run count per mode.  Default 3.
#         PROMPT -- one-shot prompt.  Default a short transformer-attention
#                   explainer.
#
# Run from the ds4 source root after building `make cuda CUDA_ARCH=sm_120`.

set -uo pipefail

N=${1:-3}
PROMPT=${2:-"Explain how a transformer attention mechanism works in three short paragraphs."}

DS4=./ds4
TMPDIR=${TMPDIR:-/tmp}/ds4_determinism_probe.$$
mkdir -p "$TMPDIR"
trap "rm -rf $TMPDIR" EXIT

run_one() {
    local mode_label="$1"
    local mode_env="$2"
    local i="$3"
    local out="$TMPDIR/${mode_label}_${i}.txt"
    pkill -9 -f "$DS4 --cuda" 2>/dev/null || true
    sleep 1
    env $mode_env $DS4 --cuda --temp 0 -n 64 -p "$PROMPT" > "$out" 2>&1
    # Strip ds4: housekeeping; MD5 the actual generation content
    sed -E '/^ds4:/d' "$out" | md5sum | awk '{print $1}'
}

echo "==== DS4_CUDA_LAYER_GRAPHS=0 (baseline) ===="
declare -a off_md5s
for i in $(seq 1 $N); do
    md5=$(run_one off "DS4_CUDA_LAYER_GRAPHS=0" $i)
    off_md5s+=("$md5")
    echo "  run $i: $md5"
done

echo
echo "==== DS4_CUDA_LAYER_GRAPHS=1 (capture path) ===="
declare -a on_md5s
for i in $(seq 1 $N); do
    md5=$(run_one on "DS4_CUDA_LAYER_GRAPHS=1" $i)
    on_md5s+=("$md5")
    echo "  run $i: $md5"
done

echo
echo "==== Determinism report ===="

off_unique=$(printf '%s\n' "${off_md5s[@]}" | sort -u | wc -l)
on_unique=$(printf '%s\n' "${on_md5s[@]}" | sort -u | wc -l)

echo "  OFF unique MD5s: $off_unique / $N"
echo "  ON  unique MD5s: $on_unique / $N"

if [ "$off_unique" -eq 1 ] && [ "$on_unique" -eq 1 ] && [ "${off_md5s[0]}" = "${on_md5s[0]}" ]; then
    echo "  RESULT: PASS -- OFF and ON both deterministic and identical"
    exit 0
elif [ "$off_unique" -eq 1 ] && [ "$on_unique" -eq 1 ]; then
    echo "  RESULT: FAIL/divergence -- OFF and ON each deterministic but differ"
    echo "    OFF: ${off_md5s[0]}"
    echo "    ON:  ${on_md5s[0]}"
    exit 1
elif [ "$off_unique" -eq 1 ] && [ "$on_unique" -gt 1 ]; then
    echo "  RESULT: FAIL/non-determinism -- OFF deterministic, ON varies across runs"
    echo "    OFF: ${off_md5s[0]}"
    echo "    ON values:"
    for m in "${on_md5s[@]}"; do echo "      $m"; done
    exit 2
else
    echo "  RESULT: UNEXPECTED -- both modes show variance"
    exit 3
fi
