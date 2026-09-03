/* ds4_vit_common.cuh -- neutral ViT device helpers for the Vision-Exp encoder.
 *
 * Track B inc 2.  These are the eight helpers upstream antirez/ds4's DeepSeek
 * encoder (ds4_deepseek4_vision_gpu.cuh @110afdd) takes from its GLM-5.3
 * vision substrate (ds4_glm53_vision_gpu.cuh, ds4_cuda.cu), re-expressed on
 * the fork's idioms so no GLM symbol enters the tree:
 *   - every launch on ds4_current_stream() (never the default stream);
 *   - weight bytes resolved through cuda_model_range_ptr() (the fork's
 *     registered-map / HBM-cache aware resolver), not upstream's tiered
 *     cuda_resolve_weight_ptr;
 *   - the BF16 GEMM through the fork's global g_cublas after
 *     cuda_cublas_ws_prep() (deterministic split-K padding), with the BF16
 *     activation staging buffer supplied by the CALLER (one per encode,
 *     sized once) instead of the shared cuda_tmp scratch, so an encode can
 *     never grow the scratch that captured decode graphs bake pointers into.
 * Numerics are byte-for-byte upstream's: the same kernels, the same fmaf
 * order, the same cuBLAS call (CUDA_R_16BF inputs, F32 accumulate/output,
 * CUBLAS_GEMM_DEFAULT) -- the bit-exact-vs-upstream gate depends on it.
 *
 * Include from ds4_cuda.cu AFTER warp_sum_f32, cuda_ok, cuda_model_range_ptr,
 * cuda_cublas_ws_prep and g_cublas are defined (same TU, static linkage). */
#ifndef DS4_VIT_COMMON_CUH
#define DS4_VIT_COMMON_CUH

#include <cuda_bf16.h>

__device__ __forceinline__ static float ds4_vit_bf16(const uint16_t *p) {
    return __uint_as_float((uint32_t)(*p) << 16);
}

__device__ __forceinline__ static float ds4_vit_erf_approx(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float a = fabsf(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = (((((1.061405429f * t - 1.453152027f) * t) +
                       1.421413741f) * t - 0.284496736f) * t +
                       0.254829592f) * t;
    return sign * (1.0f - p * expf(-a * a));
}

/* x[i] += bias[i % width] (+ residual[i]) */
__global__ static void ds4_vit_bias_kernel(
        float          *x,
        const uint16_t *bias,
        const float    *residual,
        uint64_t        count,
        uint32_t        width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = x[i] + ds4_vit_bf16(bias + i % width);
    if (residual) v += residual[i];
    x[i] = v;
}

/* Row-wise RMSNorm with a BF16 weight; one block (256 threads) per row. */
__global__ static void ds4_vit_rms_kernel(
        float          *out,
        const float    *x,
        const uint16_t *weight,
        uint32_t        width,
        float           eps) {
    __shared__ float partial[256];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float *xr = x + (uint64_t)row * width;
    float *yr = out + (uint64_t)row * width;
    float sum = 0.0f;
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        sum = fmaf(xr[d], xr[d], sum);
    }
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)width + eps);
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        yr[d] = xr[d] * inv * ds4_vit_bf16(weight + d);
    }
}

/* Full bidirectional attention over `rows` patch tokens, 16 heads x 64,
 * online softmax; grid (rows, 16), block 32 lanes (two 32-wide halves of
 * the head dim per lane).  Scale 1/8 = 1/sqrt(64). */
__global__ static void ds4_vit_attention_kernel(
        float       *out,
        const float *q,
        const float *k,
        const float *v,
        uint32_t     rows) {
    __shared__ float dot[32];
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (row >= rows || head >= 16u || lane >= 32u) return;
    const uint64_t base = (uint64_t)row * 1024u + (uint64_t)head * 64u;
    const float q0 = q[base + lane];
    const float q1 = q[base + lane + 32u];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float max_score = -INFINITY;
    float denom = 0.0f;
    for (uint32_t key_row = 0; key_row < rows; key_row++) {
        const uint64_t kb = (uint64_t)key_row * 1024u + (uint64_t)head * 64u;
        dot[lane] = q0 * k[kb + lane] + q1 * k[kb + lane + 32u];
        __syncthreads();
        for (uint32_t stride = 16u; stride != 0u; stride >>= 1u) {
            if (lane < stride) dot[lane] += dot[lane + stride];
            __syncthreads();
        }
        const float score = dot[0] * 0.125f;
        const float next_max = fmaxf(max_score, score);
        const float old_scale = key_row == 0u ? 0.0f : expf(max_score - next_max);
        const float new_scale = expf(score - next_max);
        denom = denom * old_scale + new_scale;
        acc0 = acc0 * old_scale + new_scale * v[kb + lane];
        acc1 = acc1 * old_scale + new_scale * v[kb + lane + 32u];
        max_score = next_max;
        __syncthreads();
    }
    out[base + lane] = acc0 / denom;
    out[base + lane + 32u] = acc1 / denom;
}

__global__ static void ds4_vit_f32_to_bf16_kernel(
        __nv_bfloat16 *out,
        const float   *x,
        uint64_t       n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(x[i]);
}

/* out[row][col] = sum_i w[col][i] * x[row][i]; grid (ceil(out/8), rows),
 * block 256 = 8 warps, one output column per warp (the n_rows <= 8 path). */
__global__ static void ds4_vit_matvec_bf16_f32_kernel(
        float          *out,
        const uint16_t *weights,
        const float    *x,
        uint32_t        in_dim,
        uint32_t        out_dim) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t col = blockIdx.x * 8u + warp;
    const uint32_t row = blockIdx.y;
    float sum = 0.0f;
    if (col < out_dim) {
        const uint16_t *wrow = weights + (uint64_t)col * in_dim;
        const float *xrow = x + (uint64_t)row * in_dim;
        for (uint32_t i = lane; i < in_dim; i += 32u) {
            const float w = __uint_as_float((uint32_t)wrow[i] << 16);
            sum = fmaf(w, xrow[i], sum);
        }
    }
    sum = warp_sum_f32(sum);
    if (lane == 0u && col < out_dim) {
        out[(uint64_t)row * out_dim + col] = sum;
    }
}

/* Resolve `elements` BF16 weights at `offset` of the registered sidecar map
 * to a device-readable pointer (registered UVA mapping or HBM-cache hit). */
static const uint16_t *ds4_vit_weight(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    elements,
        const char *label) {
    if (!model_map || elements > UINT64_MAX / sizeof(uint16_t) || offset > model_size) return NULL;
    const uint64_t bytes = elements * sizeof(uint16_t);
    if (bytes > model_size - offset) return NULL;
    return (const uint16_t *)cuda_model_range_ptr(model_map, offset, bytes, label);
}

static int ds4_vit_launch_ok(const char *label) {
    return cuda_ok(cudaGetLastError(), label);
}

/* out[n_rows x out_dim] (F32) = x[n_rows x in_dim] (F32) * W[out_dim x in_dim]^T (BF16).
 * xb: caller-owned BF16 staging with room for n_rows*in_dim elements (only
 * read when n_rows > 8; the matvec path needs none). */
static int ds4_vit_matmul_bf16(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows,
        __nv_bfloat16        *xb,
        uint64_t              xb_elements) {
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u || n_rows == 0u ||
        (uint64_t)out_dim > UINT64_MAX / in_dim || weight_offset > model_size) return 0;
    const uint64_t weight_elements = (uint64_t)out_dim * in_dim;
    const uint64_t weight_bytes = weight_elements * sizeof(uint16_t);
    const uint64_t input_elements = (uint64_t)n_rows * in_dim;
    const uint64_t output_elements = (uint64_t)n_rows * out_dim;
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < input_elements * sizeof(float) ||
        out->bytes < output_elements * sizeof(float)) return 0;
    const uint16_t *weights = ds4_vit_weight(model_map, model_size, weight_offset,
                                             weight_elements, "vision BF16 matrix");
    if (!weights) return 0;
    cudaStream_t stream = ds4_current_stream();
    if (n_rows <= 8u) {
        const dim3 grid((out_dim + 7u) / 8u, n_rows, 1u);
        ds4_vit_matvec_bf16_f32_kernel<<<grid, 256u, 0, stream>>>(
                (float *)out->ptr, weights, (const float *)x->ptr, in_dim, out_dim);
        return cuda_ok(cudaGetLastError(), "vision BF16/F32 matvec launch");
    }
    if (!xb || xb_elements < input_elements) return 0;
    /* Review follow-ups (2026-09-03): the fork's cuBLAS handle is a single
     * global that never runs inside a graph capture and, until now, was never
     * bound to a stream (every fork GEMM site is off-capture on the NULL
     * stream).  Refuse under capture, require the handle, and bind it to the
     * current stream around the call (a no-op off-capture, where
     * ds4_current_stream() is the NULL stream too) so an encoder on its own
     * stream (inc 9) cannot race the conversion kernel against the GEMM. */
    if (!g_cublas_ready || !g_cublas) {
        fprintf(stderr, "ds4: vision BF16 GEMM refused: cuBLAS is not initialised\n");
        return 0;
    }
    if (g_layer_graph_capturing_slot || g_cont_graph_capturing_slot) {
        fprintf(stderr, "ds4: vision BF16 GEMM refused under graph capture\n");
        return 0;
    }
    ds4_vit_f32_to_bf16_kernel<<<(unsigned)((input_elements + 255u) / 256u), 256, 0, stream>>>(
            xb, (const float *)x->ptr, input_elements);
    if (!cuda_ok(cudaGetLastError(), "vision BF16 activation conversion launch")) return 0;
    cuda_cublas_ws_prep(stream);
    cudaStream_t prev_stream = NULL;
    if (cublasGetStream(g_cublas, &prev_stream) != CUBLAS_STATUS_SUCCESS) return 0;
    if (prev_stream != stream &&
        cublasSetStream(g_cublas, stream) != CUBLAS_STATUS_SUCCESS) return 0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)out_dim, (int)n_rows, (int)in_dim,
            &alpha,
            weights, CUDA_R_16BF, (int)in_dim,
            xb, CUDA_R_16BF, (int)in_dim,
            &beta,
            out->ptr, CUDA_R_32F, (int)out_dim,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (prev_stream != stream) (void)cublasSetStream(g_cublas, prev_stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "ds4: vision BF16 GEMM failed (cublas status %d)\n", (int)status);
        return 0;
    }
    return 1;
}

#endif /* DS4_VIT_COMMON_CUH */
