#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int M = 4096;
constexpr int N = 2048;
constexpr int K = 8192;

#define HIP_OK(expr) do { \
    hipError_t e_ = (expr); \
    if (e_ != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
        std::exit(1); \
    } \
} while (0)

#define BLAS_OK(expr) do { \
    hipblasStatus_t s_ = (expr); \
    if (s_ != HIPBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s:%d: hipBLAS status %d\n", __FILE__, __LINE__, (int)s_); \
        std::exit(1); \
    } \
} while (0)

__global__ void init_half(half *p, size_t n, unsigned salt) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int v = (int)((i * 17u + salt) % 251u) - 125;
        p[i] = __float2half((float)v / 2048.0f);
    }
}

template <int BLOCK_M, int BLOCK_N>
__global__ void f16_gemm_wmma_tiled(float *c, const half *a, const half *b) {
    static_assert(BLOCK_M % 16 == 0 && BLOCK_N % 16 == 0, "16x16 WMMA tiles");
    constexpr int WAVE_M = BLOCK_M / 16;
    constexpr int WAVE_N = BLOCK_N / 16;
    constexpr int WAVES = WAVE_M * WAVE_N;
    static_assert(WAVES <= 32, "at most 1024 wave32 threads");

    __shared__ half sh_a[BLOCK_M * 16];
    __shared__ half sh_b[16 * BLOCK_N];

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int wave_m = wave % WAVE_M;
    const int wave_n = wave / WAVE_M;
    const int m0 = (int)blockIdx.x * BLOCK_M;
    const int n0 = (int)blockIdx.y * BLOCK_N;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half, rocwmma::col_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    frag_a fa;
    frag_b fb;
    frag_c acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int p = tid; p < BLOCK_M * 8; p += WAVES * 32) {
            const int j = p * 2;
            const int row = j % BLOCK_M;
            const int kk = j / BLOCK_M;
            *reinterpret_cast<unsigned *>(sh_a + j) =
                *reinterpret_cast<const unsigned *>(a + (m0 + row) + (size_t)(k0 + kk) * M);
        }
        for (int p = tid; p < 8 * BLOCK_N; p += WAVES * 32) {
            const int j = p * 2;
            const int kk = j & 15;
            const int col = j >> 4;
            *reinterpret_cast<unsigned *>(sh_b + j) =
                *reinterpret_cast<const unsigned *>(b + (k0 + kk) + (size_t)(n0 + col) * K);
        }
        __syncthreads();
        rocwmma::load_matrix_sync(fa, sh_a + wave_m * 16, BLOCK_M);
        rocwmma::load_matrix_sync(fb, sh_b + wave_n * 16 * 16, 16);
        rocwmma::mma_sync(acc, fa, fb, acc);
        __syncthreads();
    }

    rocwmma::store_matrix_sync(
        c + (m0 + wave_m * 16) + (size_t)(n0 + wave_n * 16) * M,
        acc,
        M,
        rocwmma::mem_col_major);
}

template <int BLOCK_M, int BLOCK_N>
float time_custom(float *c, const half *a, const half *b, int reps) {
    constexpr int waves = (BLOCK_M / 16) * (BLOCK_N / 16);
    const dim3 grid(M / BLOCK_M, N / BLOCK_N, 1);
    const dim3 block(waves * 32, 1, 1);
    for (int i = 0; i < 2; ++i) {
        f16_gemm_wmma_tiled<BLOCK_M, BLOCK_N><<<grid, block>>>(c, a, b);
    }
    HIP_OK(hipGetLastError());
    HIP_OK(hipDeviceSynchronize());
    hipEvent_t start, stop;
    HIP_OK(hipEventCreate(&start));
    HIP_OK(hipEventCreate(&stop));
    HIP_OK(hipEventRecord(start));
    for (int i = 0; i < reps; ++i) {
        f16_gemm_wmma_tiled<BLOCK_M, BLOCK_N><<<grid, block>>>(c, a, b);
    }
    HIP_OK(hipEventRecord(stop));
    HIP_OK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_OK(hipEventElapsedTime(&ms, start, stop));
    HIP_OK(hipEventDestroy(start));
    HIP_OK(hipEventDestroy(stop));
    return ms / reps;
}

float time_hipblas(hipblasHandle_t handle, float *c, const half *a, const half *b, int reps) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int i = 0; i < 2; ++i) {
        BLAS_OK(hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                             M, N, K, &alpha,
                             a, HIP_R_16F, M,
                             b, HIP_R_16F, K,
                             &beta,
                             c, HIP_R_32F, M,
                             HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    }
    HIP_OK(hipDeviceSynchronize());
    hipEvent_t start, stop;
    HIP_OK(hipEventCreate(&start));
    HIP_OK(hipEventCreate(&stop));
    HIP_OK(hipEventRecord(start));
    for (int i = 0; i < reps; ++i) {
        BLAS_OK(hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                             M, N, K, &alpha,
                             a, HIP_R_16F, M,
                             b, HIP_R_16F, K,
                             &beta,
                             c, HIP_R_32F, M,
                             HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    }
    HIP_OK(hipEventRecord(stop));
    HIP_OK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_OK(hipEventElapsedTime(&ms, start, stop));
    HIP_OK(hipEventDestroy(start));
    HIP_OK(hipEventDestroy(stop));
    return ms / reps;
}

void compare(const float *ref_dev, const float *got_dev) {
    const size_t count = (size_t)M * N;
    std::vector<float> ref(count), got(count);
    HIP_OK(hipMemcpy(ref.data(), ref_dev, count * sizeof(float), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(got.data(), got_dev, count * sizeof(float), hipMemcpyDeviceToHost));
    double se = 0.0;
    float max_abs = 0.0f;
    size_t bad = 0;
    for (size_t i = 0; i < count; ++i) {
        const float d = std::fabs(ref[i] - got[i]);
        max_abs = d > max_abs ? d : max_abs;
        se += (double)d * d;
        if (!std::isfinite(got[i]) || d > 0.05f) ++bad;
    }
    std::printf("correctness max_abs=%.6g rmse=%.6g bad_gt_0.05=%zu/%zu\n",
                max_abs, std::sqrt(se / count), bad, count);
}

} // namespace

int main(int argc, char **argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    half *a = nullptr;
    half *b = nullptr;
    float *ref = nullptr;
    float *got = nullptr;
    HIP_OK(hipMalloc(&a, (size_t)M * K * sizeof(half)));
    HIP_OK(hipMalloc(&b, (size_t)K * N * sizeof(half)));
    HIP_OK(hipMalloc(&ref, (size_t)M * N * sizeof(float)));
    HIP_OK(hipMalloc(&got, (size_t)M * N * sizeof(float)));
    init_half<<<((size_t)M * K + 255) / 256, 256>>>(a, (size_t)M * K, 3u);
    init_half<<<((size_t)K * N + 255) / 256, 256>>>(b, (size_t)K * N, 19u);
    HIP_OK(hipGetLastError());

    hipblasHandle_t handle = nullptr;
    BLAS_OK(hipblasCreate(&handle));
    const float blas_ms = time_hipblas(handle, ref, a, b, reps);
    const float m64n64 = time_custom<64, 64>(got, a, b, reps);
    compare(ref, got);
    const float m128n32 = time_custom<128, 32>(got, a, b, reps);
    compare(ref, got);
    const float m128n16 = time_custom<128, 16>(got, a, b, reps);
    compare(ref, got);
    const float m64n32 = time_custom<64, 32>(got, a, b, reps);
    compare(ref, got);
    const float m64n128 = time_custom<64, 128>(got, a, b, reps);
    compare(ref, got);
    const float m128n64 = time_custom<128, 64>(got, a, b, reps);
    compare(ref, got);
    const float m32n128 = time_custom<32, 128>(got, a, b, reps);
    compare(ref, got);

    std::printf("hipblas %.3f ms\n", blas_ms);
    std::printf("wmma m64n64 %.3f ms\n", m64n64);
    std::printf("wmma m128n32 %.3f ms\n", m128n32);
    std::printf("wmma m128n16 %.3f ms\n", m128n16);
    std::printf("wmma m64n32 %.3f ms\n", m64n32);
    std::printf("wmma m64n128 %.3f ms\n", m64n128);
    std::printf("wmma m128n64 %.3f ms\n", m128n64);
    std::printf("wmma m32n128 %.3f ms\n", m32n128);

    BLAS_OK(hipblasDestroy(handle));
    HIP_OK(hipFree(got));
    HIP_OK(hipFree(ref));
    HIP_OK(hipFree(b));
    HIP_OK(hipFree(a));
    return 0;
}
