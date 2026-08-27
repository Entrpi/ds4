#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

#ifndef LAB_M
#define LAB_M 1024
#endif
#ifndef LAB_N
#define LAB_N 2048
#endif
#ifndef LAB_K
#define LAB_K 4096
#endif
#ifndef LAB_BATCH
#define LAB_BATCH 8
#endif

constexpr int M = LAB_M;
constexpr int N = LAB_N;
constexpr int K = LAB_K;
constexpr int BATCH = LAB_BATCH;

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
__global__ void f16_gemm_tn_wmma_batched(float *c, const half *a, const half *b) {
    static_assert(BLOCK_M % 16 == 0 && BLOCK_N % 16 == 0, "16x16 WMMA tiles");
    constexpr int WAVE_M = BLOCK_M / 16;
    constexpr int WAVE_N = BLOCK_N / 16;
    constexpr int WAVES = WAVE_M * WAVE_N;
    static_assert(WAVES <= 32, "at most 1024 wave32 threads");

    // op(A)=A^T is row-major MxK; B and C are column-major.
    __shared__ half sh_a[BLOCK_M * 16];
    __shared__ half sh_b[16 * BLOCK_N];

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int wave_m = wave % WAVE_M;
    const int wave_n = wave / WAVE_M;
    const int m0 = (int)blockIdx.x * BLOCK_M;
    const int n0 = (int)blockIdx.y * BLOCK_N;
    const size_t batch = blockIdx.z;
    a += batch * (size_t)K * M;
    b += batch * (size_t)K * N;
    c += batch * (size_t)M * N;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    frag_a fa;
    frag_b fb;
    frag_c acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int p = tid; p < BLOCK_M * 8; p += WAVES * 32) {
            const int j = p * 2;
            const int kk = j & 15;
            const int row = j >> 4;
            *reinterpret_cast<unsigned *>(sh_a + j) =
                *reinterpret_cast<const unsigned *>(a + (k0 + kk) + (size_t)(m0 + row) * K);
        }
        for (int p = tid; p < 8 * BLOCK_N; p += WAVES * 32) {
            const int j = p * 2;
            const int kk = j & 15;
            const int col = j >> 4;
            *reinterpret_cast<unsigned *>(sh_b + j) =
                *reinterpret_cast<const unsigned *>(b + (k0 + kk) + (size_t)(n0 + col) * K);
        }
        __syncthreads();
        rocwmma::load_matrix_sync(fa, sh_a + wave_m * 16 * 16, 16);
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
    if constexpr (M % BLOCK_M != 0 || N % BLOCK_N != 0 || M < BLOCK_M) {
        return NAN;
    }
    constexpr int waves = (BLOCK_M / 16) * (BLOCK_N / 16);
    const dim3 grid(M / BLOCK_M, N / BLOCK_N, BATCH);
    const dim3 block(waves * 32, 1, 1);
    for (int i = 0; i < 2; ++i) {
        f16_gemm_tn_wmma_batched<BLOCK_M, BLOCK_N><<<grid, block>>>(c, a, b);
    }
    HIP_OK(hipGetLastError());
    HIP_OK(hipDeviceSynchronize());
    hipEvent_t start, stop;
    HIP_OK(hipEventCreate(&start));
    HIP_OK(hipEventCreate(&stop));
    HIP_OK(hipEventRecord(start));
    for (int i = 0; i < reps; ++i) {
        f16_gemm_tn_wmma_batched<BLOCK_M, BLOCK_N><<<grid, block>>>(c, a, b);
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
    auto launch = [&] {
        BLAS_OK(hipblasGemmStridedBatchedEx(
            handle, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K, &alpha,
            a, HIP_R_16F, K, (long long)K * M,
            b, HIP_R_16F, K, (long long)K * N,
            &beta,
            c, HIP_R_32F, M, (long long)M * N,
            BATCH, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    };
    launch();
    launch();
    HIP_OK(hipDeviceSynchronize());
    hipEvent_t start, stop;
    HIP_OK(hipEventCreate(&start));
    HIP_OK(hipEventCreate(&stop));
    HIP_OK(hipEventRecord(start));
    for (int i = 0; i < reps; ++i) launch();
    HIP_OK(hipEventRecord(stop));
    HIP_OK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_OK(hipEventElapsedTime(&ms, start, stop));
    HIP_OK(hipEventDestroy(start));
    HIP_OK(hipEventDestroy(stop));
    return ms / reps;
}

void compare(const float *ref_dev, const float *got_dev) {
    const size_t count = (size_t)BATCH * M * N;
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

template <int BLOCK_M, int BLOCK_N>
void check_variant(const char *name, float *got, const float *ref,
                   const half *a, const half *b, int reps, float *timing) {
    *timing = time_custom<BLOCK_M, BLOCK_N>(got, a, b, reps);
    if (std::isfinite(*timing)) {
        compare(ref, got);
    } else {
        std::printf("correctness %s skipped for M=%d N=%d\n", name, M, N);
    }
}

} // namespace

int main(int argc, char **argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    half *a = nullptr;
    half *b = nullptr;
    float *ref = nullptr;
    float *got = nullptr;
    const size_t a_count = (size_t)BATCH * K * M;
    const size_t b_count = (size_t)BATCH * K * N;
    const size_t c_count = (size_t)BATCH * M * N;
    HIP_OK(hipMalloc(&a, a_count * sizeof(half)));
    HIP_OK(hipMalloc(&b, b_count * sizeof(half)));
    HIP_OK(hipMalloc(&ref, c_count * sizeof(float)));
    HIP_OK(hipMalloc(&got, c_count * sizeof(float)));
    init_half<<<(a_count + 255) / 256, 256>>>(a, a_count, 3u);
    init_half<<<(b_count + 255) / 256, 256>>>(b, b_count, 19u);
    HIP_OK(hipGetLastError());

    hipblasHandle_t handle = nullptr;
    BLAS_OK(hipblasCreate(&handle));
    const float blas_ms = time_hipblas(handle, ref, a, b, reps);
    float m16n64, m16n128, m32n64, m32n128, m64n32, m64n64,
          m64n128, m128n32, m128n64, m256n16, m256n32;
    check_variant<16, 64>("m16n64", got, ref, a, b, reps, &m16n64);
    check_variant<16, 128>("m16n128", got, ref, a, b, reps, &m16n128);
    check_variant<32, 64>("m32n64", got, ref, a, b, reps, &m32n64);
    check_variant<32, 128>("m32n128", got, ref, a, b, reps, &m32n128);
    check_variant<64, 32>("m64n32", got, ref, a, b, reps, &m64n32);
    check_variant<64, 64>("m64n64", got, ref, a, b, reps, &m64n64);
    check_variant<64, 128>("m64n128", got, ref, a, b, reps, &m64n128);
    check_variant<128, 32>("m128n32", got, ref, a, b, reps, &m128n32);
    check_variant<128, 64>("m128n64", got, ref, a, b, reps, &m128n64);
    check_variant<256, 16>("m256n16", got, ref, a, b, reps, &m256n16);
    check_variant<256, 32>("m256n32", got, ref, a, b, reps, &m256n32);

    std::printf("shape M=%d N=%d K=%d batch=%d\n", M, N, K, BATCH);
    std::printf("hipblas %.3f ms\n", blas_ms);
    std::printf("wmma m16n64 %.3f ms\n", m16n64);
    std::printf("wmma m16n128 %.3f ms\n", m16n128);
    std::printf("wmma m32n64 %.3f ms\n", m32n64);
    std::printf("wmma m32n128 %.3f ms\n", m32n128);
    std::printf("wmma m64n32 %.3f ms\n", m64n32);
    std::printf("wmma m64n64 %.3f ms\n", m64n64);
    std::printf("wmma m64n128 %.3f ms\n", m64n128);
    std::printf("wmma m128n32 %.3f ms\n", m128n32);
    std::printf("wmma m128n64 %.3f ms\n", m128n64);
    std::printf("wmma m256n16 %.3f ms\n", m256n16);
    std::printf("wmma m256n32 %.3f ms\n", m256n32);

    BLAS_OK(hipblasDestroy(handle));
    HIP_OK(hipFree(got));
    HIP_OK(hipFree(ref));
    HIP_OK(hipFree(b));
    HIP_OK(hipFree(a));
    return 0;
}
