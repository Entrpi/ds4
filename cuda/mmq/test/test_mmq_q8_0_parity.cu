// SPDX-License-Identifier: MIT
// test_mmq_q8_0_parity.cu - parity test for ds4_mmq_q8_0_dense vs a
// straightforward CPU dequantize+matmul reference.
//
// Build:
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_120 \
//        -I/path/to/cuda/mmq \
//        test_mmq_q8_0_parity.cu libds4mmq.a -lcudart -lcublas -lcuda \
//        -o test_mmq_q8_0_parity
//
// Run:
//   ./test_mmq_q8_0_parity

#include "ds4_mmq.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int QK8_0 = 32;

// Mirror of llama.cpp's block_q8_0: fp16 scale + 32 int8 quantized values.
struct cpu_block_q8_0 {
    uint16_t d;       // bit pattern of __half scale
    int8_t   qs[QK8_0];
};

static_assert(sizeof(cpu_block_q8_0) == 34, "block_q8_0 must be 34 bytes");

// Convert a uint16_t holding an IEEE 754 half-precision value to float.
// Standalone (no CUDA host fp16 needed).
float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = (h >>  0) & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // subnormal -> normalize
            while ((mant & 0x400) == 0) { mant <<= 1; exp -= 1; }
            exp += 1; mant &= 0x3ff;
            f = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

uint16_t float_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 31) & 0x1u;
    int32_t  exp  = ((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    uint16_t h;
    if (exp >= 31) {
        h = (sign << 15) | (0x1f << 10) | (mant ? 0x200 : 0);  // inf or nan
    } else if (exp <= 0) {
        if (exp < -10) {
            h = sign << 15;  // underflow -> 0
        } else {
            mant |= 0x800000;
            uint32_t shift = 14 - exp;
            uint32_t r = mant >> shift;
            // round-to-nearest-even
            if (mant & (1u << (shift - 1))) r += 1;
            h = (sign << 15) | r;
        }
    } else {
        // round-to-nearest-even
        if (mant & 0x1000) {
            mant += 0x2000;
            if (mant & 0x800000) { mant = 0; exp += 1; }
        }
        h = (sign << 15) | (exp << 10) | (mant >> 13);
    }
    return h;
}

// Quantize a contiguous row of `K` floats to Q8_0 blocks (K must be a
// multiple of 32). Writes (K/32) blocks. Same algorithm as ggml's q8_0
// quantizer in `quantize_q8_0_reference`.
void quantize_row_q8_0_cpu(const float * src, cpu_block_q8_0 * dst, int K) {
    const int nb = K / QK8_0;
    for (int b = 0; b < nb; b++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            const float v = std::fabs(src[b * QK8_0 + j]);
            if (v > amax) amax = v;
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        dst[b].d = float_to_fp16(d);
        for (int j = 0; j < QK8_0; j++) {
            const float x = src[b * QK8_0 + j] * id;
            dst[b].qs[j] = (int8_t) std::lround(std::max(-128.f, std::min(127.f, x)));
        }
    }
}

// Reference dequant + GEMM. Layouts (all row-major C contiguous arrays):
//   W: row-major [M rows, K cols], Q8_0 packed. M*K/32 blocks total. Row i
//      occupies W[i * (K/32) .. i * (K/32) + (K/32)].
//   X: row-major [N rows, K cols], F32. N*K floats total. ggml's
//      convention is "K innermost" - so for a logical [K, N] tensor, we
//      store it as N batches of K contiguous floats. Hence row stride K
//      in our flat array, and column 'col' of the logical [K, N] matrix
//      lives at X[col * K + k] for k in [0, K).
//   Y: column-major output [M rows, N cols]: Y[col * M + row]. This is
//      what mmq's mmq_write_back_* writes (dst[ids_dst[j]*stride + i] with
//      stride=M, j=col, i=row).
void ref_matmul(
        const cpu_block_q8_0 * W, const float * X, float * Y,
        int M, int N, int K) {
    const int nb_per_row = K / QK8_0;
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float acc = 0.0f;
            for (int b = 0; b < nb_per_row; b++) {
                const cpu_block_q8_0 & blk = W[row * nb_per_row + b];
                const float d = fp16_to_float(blk.d);
                for (int j = 0; j < QK8_0; j++) {
                    const int k = b * QK8_0 + j;
                    acc += d * blk.qs[j] * X[col * K + k];
                }
            }
            Y[col * M + row] = acc;
        }
    }
}

bool check_close(const std::vector<float> & got, const std::vector<float> & ref,
                 float abs_tol, float rel_tol, int max_print = 8) {
    int n_bad = 0;
    float worst_abs = 0.0f, worst_rel = 0.0f;
    int worst_i = -1;
    for (size_t i = 0; i < got.size(); i++) {
        const float ag = got[i];
        const float ar = ref[i];
        const float ae = std::fabs(ag - ar);
        const float re = ar != 0.0f ? ae / std::fabs(ar) : (ae > 0 ? INFINITY : 0.0f);
        if (ae > abs_tol && re > rel_tol) {
            if (n_bad < max_print) {
                fprintf(stderr, "  [%zu] got=%.6g ref=%.6g abs=%.3g rel=%.3g\n",
                        i, ag, ar, ae, re);
            }
            n_bad++;
        }
        if (ae > worst_abs) { worst_abs = ae; worst_i = (int)i; }
        if (re > worst_rel) { worst_rel = re; }
    }
    fprintf(stderr, "  worst abs=%.3g  worst rel=%.3g  bad=%d / %zu  (at i=%d)\n",
            worst_abs, worst_rel, n_bad, got.size(), worst_i);
    return n_bad == 0;
}

bool run_one_shape(int M, int N, int K, uint32_t seed) {
    fprintf(stderr, "=== M=%d N=%d K=%d  seed=%u ===\n", M, N, K, seed);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // 1. Random F32 weight matrix [M, K] row-major.
    std::vector<float> W_f32(M * K);
    for (auto & v : W_f32) v = nd(rng);

    // 2. Quantize to Q8_0, blockwise per row.
    const int nb_per_row = K / QK8_0;
    std::vector<cpu_block_q8_0> W_q8(M * nb_per_row);
    for (int row = 0; row < M; row++) {
        quantize_row_q8_0_cpu(&W_f32[row * K], &W_q8[row * nb_per_row], K);
    }

    // 3. Random F32 activation [K, N] row-major.
    std::vector<float> X_f32(K * N);
    for (auto & v : X_f32) v = nd(rng);

    // 4. CPU reference: out[col * M + row] = sum_k W_q8[row, k] * X[k, col]
    std::vector<float> ref_out(M * N, 0.0f);
    ref_matmul(W_q8.data(), X_f32.data(), ref_out.data(), M, N, K);

    // 5. GPU: copy inputs, run ds4_mmq_q8_0_dense, copy output back.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void * dW = nullptr;
    float * dX = nullptr;
    float * dY = nullptr;
    cudaMalloc(&dW, W_q8.size() * sizeof(cpu_block_q8_0));
    cudaMalloc(&dX, X_f32.size() * sizeof(float));
    cudaMalloc(&dY, M * N * sizeof(float));
    cudaMemcpyAsync(dW, W_q8.data(), W_q8.size() * sizeof(cpu_block_q8_0), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX, X_f32.data(), X_f32.size() * sizeof(float),       cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(dY, 0, M * N * sizeof(float), stream);

    int rc = ds4_mmq_q8_0_dense(dW, dX, dY, M, N, K, stream);
    if (rc != 0) {
        fprintf(stderr, "ds4_mmq_q8_0_dense returned %d\n", rc);
        return false;
    }

    std::vector<float> got_out(M * N, 0.0f);
    cudaMemcpyAsync(got_out.data(), dY, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(dW); cudaFree(dX); cudaFree(dY);
    cudaStreamDestroy(stream);

    // 6. Compare.
    // Tolerances: Q8_0 has ~8 bits of precision per element. After
    // accumulating K=256 partial products, expect ~sqrt(K)/127 ≈ 0.13
    // relative error in the worst case. abs_tol scales with sqrt(K).
    const float abs_tol = 0.05f * std::sqrt((float)K);
    const float rel_tol = 0.05f;
    bool ok = check_close(got_out, ref_out, abs_tol, rel_tol);
    fprintf(stderr, "%s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    (void)argc; (void)argv;

    int rc = ds4_mmq_init(0);
    if (rc != 0) {
        fprintf(stderr, "ds4_mmq_init failed: %d\n", rc);
        return 1;
    }

    bool all_ok = true;
    // Small shapes for fast verification. K must be multiple of 256.
    all_ok &= run_one_shape(/*M=*/64,   /*N=*/4,   /*K=*/256,  /*seed=*/0xC0FFEE);
    all_ok &= run_one_shape(/*M=*/128,  /*N=*/8,   /*K=*/512,  /*seed=*/0xDEADBEE);
    all_ok &= run_one_shape(/*M=*/64,   /*N=*/1,   /*K=*/256,  /*seed=*/0x12345);  // single-token decode
    // V4 Flash-ish shapes (attn_q_a: K=4096, M=1024).
    all_ok &= run_one_shape(/*M=*/1024, /*N=*/16,  /*K=*/4096, /*seed=*/0xBAD7E11);

    fprintf(stderr, "===================\n");
    fprintf(stderr, "%s\n", all_ok ? "ALL PASS" : "SOME FAILED");
    return all_ok ? 0 : 1;
}
