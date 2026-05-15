// SPDX-License-Identifier: MIT
// ds4_mmq.cu - host wrapper around llama.cpp's vendored mul_mat_q kernels.
//
// Implements the public ds4_mmq_* entry points and explicitly instantiates
// the mul_mat_q_case<T> template for each quant type the caller needs.
//
// Status:
//   Q8_0 dense ............ implemented, parity-tested against CPU reference
//   Q2_K dense ............ pending (Phase 3)
//   IQ2_XXS dense ......... pending (Phase 3)
//   Q8_0 MoE _id .......... pending (Phase 4)
//   Q2_K MoE _id .......... pending (Phase 4)
//   IQ2_XXS MoE _id ....... pending (Phase 4)

#include "ds4_mmq.h"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdio>

// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------

extern "C" int ds4_mmq_init(int device) {
    if (device < 0) {
        fprintf(stderr, "ds4_mmq_init: invalid device %d\n", device);
        return -1;
    }
    ggml_cuda_set_device(device);
    // Trigger lazy population of the device-info singleton.
    const auto & info = ggml_cuda_info();
    if (info.device_count == 0) {
        fprintf(stderr, "ds4_mmq_init: no CUDA devices found\n");
        return -1;
    }
    if (device >= info.device_count) {
        fprintf(stderr, "ds4_mmq_init: device %d out of range (have %d)\n",
                device, info.device_count);
        return -1;
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Gating: when should the caller choose mmq over dequant+cublas?
//
// Body lifted verbatim from llama.cpp's ggml/src/ggml-cuda/mmq.cu:267-372
// (we do not vendor mmq.cu itself, since its other half talks to ggml_tensor
// and ggml_backend internals we don't carry over).
// ----------------------------------------------------------------------------

static bool ds4_should_use_mmq_impl(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(type); GGML_UNUSED(cc); GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return false;
#endif

    bool mmq_supported;
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }
    if (!mmq_supported) return false;

    if (turing_mma_available(cc)) {
        return true;
    }
    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return true;
#endif

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }
    if (amd_mfma_available(cc)) {
        if (GGML_CUDA_CC_IS_CDNA3(cc)) return true;
        if (n_experts > 64 || ne11 <= 128) return true;
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
            type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) return true;
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) return true;
        return false;
    }
    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            if (n_experts >= 64) return true;
            switch (type) {
                case GGML_TYPE_Q2_K: return ne11 <= 128;
                case GGML_TYPE_Q6_K: return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default: return true;
            }
        }
        return true;
    }
    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

extern "C" int ds4_mmq_should_use(int type_x, int64_t ne11, int64_t n_experts) {
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    const enum ggml_type t = (enum ggml_type) type_x;
    return ds4_should_use_mmq_impl(t, cc, ne11, n_experts) ? 1 : 0;
}

// ----------------------------------------------------------------------------
// Dense Q8_0 entry.
//
// Computes  out[M, N] = (W [M, K]).T @ X [K, N]  with W stored as Q8_0 row-
// major blocks and X / out as F32.  ggml convention: K is the innermost dim
// of every tensor.  Output layout matches what mmq writes internally -
// column-major dst[col*M + row].
//
// Mirrors upstream mmq.cu:154-159 (the no-ids branch) but builds mmq_args
// from plain pointers + shape ints instead of ggml_tensor introspection.
// ----------------------------------------------------------------------------

// Per-device singleton context. Owns the pool for stream-K fixup scratch.
// Phase 4 will make this per-stream as well; for now a single context per
// device is sufficient for the dense path.
namespace {

ggml_backend_cuda_context * get_ctx_for_device(int device) {
    static ggml_backend_cuda_context * cached[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    if (!cached[device]) {
        cached[device] = new ggml_backend_cuda_context(device);
    }
    return cached[device];
}

} // anonymous namespace

extern "C" int ds4_mmq_q8_0_dense(
        const void  * W_q8_0,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        cudaStream_t  stream) {

    if (!W_q8_0 || !X_f32 || !out_f32) {
        fprintf(stderr, "ds4_mmq_q8_0_dense: null pointer\n");
        return -1;
    }
    if (K <= 0 || M <= 0 || N <= 0) {
        fprintf(stderr, "ds4_mmq_q8_0_dense: bad shape M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        // mmq wants K a multiple of QK_K=256 so K-quant super-blocks tile
        // cleanly. Q8_0 alone only needs K % 32 but we standardise on 256.
        fprintf(stderr, "ds4_mmq_q8_0_dense: K=%d must be a multiple of 256\n", K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "ds4_mmq_q8_0_dense: failed to get cuda context for device %d\n", dev);
        return -1;
    }

    // 1. Quantize the F32 activation into the mmq Q8_1 format.
    //    Layout: ne00=K elements per row, ne10_padded = round-up to 512.
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = N;
    const int64_t ne12         = 1;
    const int64_t ne13         = 1;

    // Buffer size matches upstream mmq.cu:128.
    const size_t nbytes_src1_q8_1 =
        ne13 * ne12 * ne11 * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    // Activation strides: dense path is contiguous F32 with K elements per row.
    //   s11 = K  (per-row stride in floats)
    //   s12 = 0  (single channel)
    //   s13 = 0  (single sample)
    quantize_mmq_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        GGML_TYPE_Q8_0, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/ne12, /*ne3=*/ne13,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_mmq_q8_0_dense: quantize failed: %s\n", cudaGetErrorString(err));
        return -2;
    }

    // 2. Build mmq_args for the dense path. Shape conventions follow
    //    upstream mmq.cu:154-159 (the no-ids branch).
    const int64_t s01 = (int64_t)K / QK8_0;        // Q8_0 blocks per weight row
    const int64_t s1  = (int64_t)M;                // stride to next output column
    const int64_t s12 = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12 * s12;

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

    const mmq_args args = {
        /*x=*/(const char *)W_q8_0,
        /*type_x=*/GGML_TYPE_Q8_0,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,    /*nrows_x=*/(int64_t)M,    /*ncols_dst=*/ne11,
        /*stride_row_x=*/s01,/*ncols_y=*/ne11,          /*nrows_dst=*/s1,
        /*nchannels_x=*/1,   /*nchannels_y=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/s12, /*stride_channel_dst=*/0,
        /*nsamples_x=*/1,    /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/s13, /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne11,
    };

    // 3. Launch.
    mul_mat_q_case<GGML_TYPE_Q8_0>(*ctx, args, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_mmq_q8_0_dense: mul_mat_q_case launch failed: %s\n", cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

// Explicit instantiation so the Q8_0 case is forced into this TU. Phase 1
// will add Q2_K and IQ2_XXS; until then those instantiations live nowhere
// and the switch in ggml_cuda_mul_mat_q_switch_type (mmq.cu, not vendored
// here) would link-fail if anyone tried to call them.
template void mul_mat_q_case<GGML_TYPE_Q8_0>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
