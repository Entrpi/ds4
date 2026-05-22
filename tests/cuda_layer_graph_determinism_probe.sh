#!/usr/bin/env bash
# Captured-decode determinism regression backstop.
#
# Runs the same DS4 greedy generation N times under both
# DS4_CUDA_LAYER_GRAPHS=0 (eager) and =1 (per-layer CUDA-graph capture)
# and compares the generated token-id dumps.  PASS = all OFF runs equal,
# all ON runs equal, and OFF == ON (the parity gate).
#
# It compares the SELECTED token-id sequence from the --dump-logprobs
# JSON -- not the console, and not the whole JSON file.  The console
# carries mode-dependent perf lines (capture vs eager run at different
# tok/s), so a console MD5 fakes false FAILs.  The JSON also carries
# per-token logit/logprob floats (%.9g): those drift by FP-tiny
# reduction-order noise that is NOT a decode divergence, so a whole-file
# MD5 fakes false FAILs too.  The greedy argmax token-id sequence is the
# authoritative per-token decode signal.
#
# Step 7 closed this gate: captured decode is bit-identical to eager
# through n=256 on sm_120 (PRO 6000) and sm_121 (GB10).  This script is
# the permanent regression backstop -- run it after any change to the
# decode path, the layer-graph cache, or the scalar substrates.
#
# Why the default is n=256, not a handful of tokens: captured-graph state
# bugs onset at staggered sequence positions, so a short run misses them.
# The three Step 7 root causes surfaced at pos 0 (stale decode token on
# the hash-mode router), pos ~25-49 (frozen compressed-row counter on the
# FP8-KV emit), and pos 128 (the sliding-window-attention saturation
# boundary).  A run shorter than ~256 tokens would have passed while
# still broken.  To localize a *new* divergence, pair this with the
# per-kernel hash dump (DS4_CUDA_LAYER_GRAPHS_HASH_DUMP=1; see the comment
# block above the implementation in ds4_cuda.cu).
#
# Usage:  bash tests/cuda_layer_graph_determinism_probe.sh [N] [PROMPT] [NTOK]
#         N      -- run count per mode.       Default 3.
#         PROMPT -- one-shot prompt.          Default a short explainer.
#         NTOK   -- generated tokens per run. Default 256 (do not lower
#                   below 256 for a real gate -- see above).
#
# Repeated-batch tip: each ds4 invocation reloads the ~87 GB model
# (~40 s).  For large N, bring up ds4_weight_server once (VMM upload,
# scope=base) and point clients at it via DS4_CUDA_WEIGHT_IPC_MANIFEST --
# per-invocation startup drops to ~14 s.  Shut the server down afterward.
#
# Run from the ds4 source root after building `make cuda CUDA_ARCH=sm_120`.

set -uo pipefail

N=${1:-3}
PROMPT=${2:-"Explain how a transformer attention mechanism works in three short paragraphs."}
NTOK=${3:-256}

DS4=./ds4
TMPDIR=${TMPDIR:-/tmp}/ds4_determinism_probe.$$
mkdir -p "$TMPDIR"
trap "rm -rf $TMPDIR" EXIT

run_one() {
    local mode_label="$1"
    local mode_env="$2"
    local i="$3"
    local lp="$TMPDIR/${mode_label}_${i}.json"
    local con="$TMPDIR/${mode_label}_${i}.con"
    pkill -9 -f "$DS4 --cuda" 2>/dev/null || true
    sleep 1
    env $mode_env $DS4 --cuda --temp 0 -n "$NTOK" -p "$PROMPT" \
        --dump-logprobs "$lp" --logprobs-top-k 1 > "$con" 2>&1
    # MD5 the SELECTED token-id sequence only (see the header for why
    # neither the console nor the whole JSON file is compared).
    local ids
    ids=$(grep -oE '"selected":\{"id":-?[0-9]+' "$lp" 2>/dev/null \
          | grep -oE -- '-?[0-9]+$' | tr '\n' ',')
    if [ -z "$ids" ]; then
        # No tokens dumped (ds4 crashed / wrote no JSON). Emit a
        # run-unique marker so empty runs can never collude into a
        # false PASS.
        echo "EMPTY-${mode_label}-${i}"
    else
        printf '%s' "$ids" | md5sum | awk '{print $1}'
    fi
}

echo "==== DS4_CUDA_LAYER_GRAPHS=0 (baseline), n=$NTOK ===="
declare -a off_md5s
for i in $(seq 1 $N); do
    md5=$(run_one off "DS4_CUDA_LAYER_GRAPHS=0" $i)
    off_md5s+=("$md5")
    echo "  run $i: $md5"
done

echo
echo "==== DS4_CUDA_LAYER_GRAPHS=1 (capture path), n=$NTOK ===="
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
