#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr int M = 24;
constexpr int N = 2048;
constexpr int K = 16384;

#define HIP_OK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); std::exit(1); } } while (0)
#define BLAS_OK(x) do { hipblasStatus_t s = (x); if (s != HIPBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s:%d: hipBLAS %d\n", __FILE__, __LINE__, (int)s); std::exit(1); } } while (0)

__global__ void init_half(half *p, size_t n, unsigned salt) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __float2half(((int)((i * 17u + salt) % 251u) - 125) / 2048.0f);
}

template <int BLOCK_N>
__global__ void tinym_wmma(float *c, const half *a_rowmajor, const half *b_colmajor) {
    constexpr int BLOCK_M = 32;
    constexpr int WAVE_M = 2;
    constexpr int WAVE_N = BLOCK_N / 16;
    constexpr int WAVES = WAVE_M * WAVE_N;
    static_assert(WAVES <= 32, "wave limit");
    __shared__ half sh_a[BLOCK_M * 16];
    __shared__ half sh_b[16 * BLOCK_N];
    __shared__ float sh_c[BLOCK_M * BLOCK_N];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int wave_m = wave & 1;
    const int wave_n = wave >> 1;
    const int n0 = blockIdx.x * BLOCK_N;

    using fa_t = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half, rocwmma::col_major>;
    using fb_t = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half, rocwmma::col_major>;
    using fc_t = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    fa_t fa; fb_t fb; fc_t acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int j = tid; j < BLOCK_M * 16; j += WAVES * 32) {
            const int row = j & 31;
            const int kk = j >> 5;
            sh_a[j] = row < M ? a_rowmajor[(size_t)row * K + k0 + kk] : __float2half(0.0f);
        }
        for (int p = tid; p < 8 * BLOCK_N; p += WAVES * 32) {
            const int j = p * 2;
            const int kk = j & 15;
            const int col = j >> 4;
            *reinterpret_cast<uint32_t *>(sh_b + j) =
                *reinterpret_cast<const uint32_t *>(b_colmajor + k0 + kk + (size_t)(n0 + col) * K);
        }
        __syncthreads();
        rocwmma::load_matrix_sync(fa, sh_a + wave_m * 16, BLOCK_M);
        rocwmma::load_matrix_sync(fb, sh_b + wave_n * 16 * 16, 16);
        rocwmma::mma_sync(acc, fa, fb, acc);
        __syncthreads();
    }
    rocwmma::store_matrix_sync(sh_c + wave_m * 16 + wave_n * 16 * BLOCK_M,
                               acc, BLOCK_M, rocwmma::mem_col_major);
    __syncthreads();
    for (int i = tid; i < M * BLOCK_N; i += WAVES * 32) {
        const int col = i / M;
        const int row = i - col * M;
        c[row + (size_t)(n0 + col) * M] = sh_c[row + col * BLOCK_M];
    }
}

template <int BN>
float time_custom(float *c, const half *a, const half *b, int reps) {
    constexpr int waves = 2 * (BN / 16);
    for (int i = 0; i < 2; ++i) tinym_wmma<BN><<<N / BN, waves * 32>>>(c, a, b);
    HIP_OK(hipDeviceSynchronize());
    hipEvent_t evt_start, evt_stop; HIP_OK(hipEventCreate(&evt_start)); HIP_OK(hipEventCreate(&evt_stop)); HIP_OK(hipEventRecord(evt_start));
    for (int i = 0; i < reps; ++i) tinym_wmma<BN><<<N / BN, waves * 32>>>(c, a, b);
    HIP_OK(hipEventRecord(evt_stop)); HIP_OK(hipEventSynchronize(evt_stop)); float ms = 0; HIP_OK(hipEventElapsedTime(&ms, evt_start, evt_stop));
    HIP_OK(hipEventDestroy(evt_start)); HIP_OK(hipEventDestroy(evt_stop)); return ms / reps;
}

float time_blas(hipblasHandle_t h, float *c, const half *a, const half *b, int reps) {
    const float alpha = 1, beta = 0;
    for (int i = 0; i < 2; ++i) BLAS_OK(hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K,
        &alpha, a, HIP_R_16F, K, b, HIP_R_16F, K, &beta, c, HIP_R_32F, M,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    HIP_OK(hipDeviceSynchronize());
    hipEvent_t evt_start, evt_stop; HIP_OK(hipEventCreate(&evt_start)); HIP_OK(hipEventCreate(&evt_stop)); HIP_OK(hipEventRecord(evt_start));
    for (int i = 0; i < reps; ++i) BLAS_OK(hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K,
        &alpha, a, HIP_R_16F, K, b, HIP_R_16F, K, &beta, c, HIP_R_32F, M,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    HIP_OK(hipEventRecord(evt_stop)); HIP_OK(hipEventSynchronize(evt_stop)); float ms = 0; HIP_OK(hipEventElapsedTime(&ms, evt_start, evt_stop));
    HIP_OK(hipEventDestroy(evt_start)); HIP_OK(hipEventDestroy(evt_stop)); return ms / reps;
}

void compare(const float *ref_d, const float *got_d) {
    std::vector<float> ref((size_t)M*N), got((size_t)M*N);
    HIP_OK(hipMemcpy(ref.data(), ref_d, ref.size()*sizeof(float), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(got.data(), got_d, got.size()*sizeof(float), hipMemcpyDeviceToHost));
    double se=0; float ma=0; size_t bad=0;
    for (size_t i=0;i<ref.size();++i) { float d=std::fabs(ref[i]-got[i]); ma=fmaxf(ma,d); se+=(double)d*d; bad += !std::isfinite(got[i]) || d>0.05f; }
    std::printf("maxabs %.8g rmse %.8g bad %zu/%zu\n", ma, std::sqrt(se/ref.size()), bad, ref.size());
}
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    half *a=nullptr,*b=nullptr; float *ref=nullptr,*got=nullptr;
    HIP_OK(hipMalloc(&a,(size_t)M*K*sizeof(half))); HIP_OK(hipMalloc(&b,(size_t)K*N*sizeof(half)));
    HIP_OK(hipMalloc(&ref,(size_t)M*N*sizeof(float))); HIP_OK(hipMalloc(&got,(size_t)M*N*sizeof(float)));
    init_half<<<((size_t)M*K+255)/256,256>>>(a,(size_t)M*K,3); init_half<<<((size_t)K*N+255)/256,256>>>(b,(size_t)K*N,19);
    hipblasHandle_t h=nullptr; BLAS_OK(hipblasCreate(&h));
    const float blas=time_blas(h,ref,a,b,reps);
    const float n64=time_custom<64>(got,a,b,reps); compare(ref,got);
    const float n128=time_custom<128>(got,a,b,reps); compare(ref,got);
    const float n256=time_custom<256>(got,a,b,reps); compare(ref,got);
    std::printf("hipblas %.3f ms\nwmma n64 %.3f ms\nwmma n128 %.3f ms\nwmma n256 %.3f ms\n",blas,n64,n128,n256);
    BLAS_OK(hipblasDestroy(h)); HIP_OK(hipFree(got)); HIP_OK(hipFree(ref)); HIP_OK(hipFree(b)); HIP_OK(hipFree(a));
}
