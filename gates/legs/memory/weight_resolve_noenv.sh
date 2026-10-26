#!/usr/bin/env bash
# memgov D3-3b gate: the model-weight resolve call graph reads NO env.
#
# The whole vocabulary the graph consults is snapshotted once by
# cuda_weight_env_read() (the ONLY function in the graph's orbit allowed
# to call getenv); every function below must contain zero getenv calls.
# Pure text check -- runs anywhere the tree lives, no build needed.
#
# Extraction: from the function's definition line (skipping forward
# declarations, which end in ';' before any '{') to the first
# column-0 '}' -- the file's uniform body style.
set -u
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
cd "$(dirname "$0")/../../.."
SRC=ds4_cuda.cu

# The resolve call graph of cuda_model_range_ptr + the per-dispatch
# derived-weight family (see local/docs/v057/d3_scoping_2026-08-14.md
# section 2.3 for provenance).
FNS=(
    cuda_model_range_ptr
    cuda_model_range_resolve
    cuda_model_range_ptr_from_fd
    cuda_model_range_populate_device_copy
    cuda_model_direct_fallback_ptr
    cuda_model_cache_limit_bytes
    cuda_model_arena_chunk_bytes
    cuda_model_arena_alloc
    cuda_vmm_arena_supported
    cuda_vmm_arena_chunk_bytes
    cuda_vmm_arena_alloc
    cuda_unit_materialize_copy
    cuda_stage_copy_to_dev
    cuda_model_stage_read
    cuda_model_copy_chunk_bytes
    cuda_model_discard_source_pages
    cuda_model_drop_file_pages
    cuda_model_load_progress_enabled
    cuda_model_load_progress_note
    cuda_derived_weight_ptr
    cuda_q8_f16_cache_allowed
    cuda_q8_f16_cache_limit_bytes
    cuda_q8_f16_cache_reserve_bytes
    cuda_q8_f16_cache_budget_notice
    cuda_q8_f16_ptr
    cuda_q8_f32_ptr
    cuda_q8_use_dp4a
    cuda_q8_f16_preload_allowed
    cuda_q8_f32_cache_allowed
)

extract_fn() {
    awk -v fn="$1" '
        !infn && $0 ~ ("[ \\*]" fn "\\(") { infn=1; body=0 }
        infn {
            print
            if (!body && index($0, "{")) body = 1
            if (!body && $0 ~ /;[ \t]*$/) infn = 0   # forward declaration
            else if (body && /^}/) exit
        }
    ' "$SRC"
}

fail=0
for fn in "${FNS[@]}"; do
    body=$(extract_fn "$fn")
    if [ -z "$body" ]; then
        echo "FAIL: $fn not found in $SRC (gate list stale?)"
        fail=1
        continue
    fi
    hits=$(printf '%s\n' "$body" | grep -c 'getenv')
    if [ "$hits" -ne 0 ]; then
        echo "FAIL: $fn contains $hits getenv call(s):"
        printf '%s\n' "$body" | grep -n 'getenv' | sed 's/^/    /'
        fail=1
    fi
done

# The snapshot itself must still exist and be the single reader.
if ! grep -q 'static cuda_weight_env_t cuda_weight_env_read' "$SRC"; then
    echo "FAIL: cuda_weight_env_read missing (snapshot deleted?)"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "PASS: ${#FNS[@]} resolve-graph functions, zero getenv"
fi
exit "$fail"
