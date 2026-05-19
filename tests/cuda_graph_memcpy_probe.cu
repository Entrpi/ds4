/*
 * Empirical probe for the load-bearing CUDA Graph capture semantic:
 *
 *   Does a captured cudaMemcpyAsync(d_dst, h_src, n, H2D, stream) node
 *   record the source by ADDRESS (so mutations to *h_src between replays
 *   propagate to d_dst) or by VALUE (so mutations are ignored)?
 *
 * The device-side-scalars + captured-memcpy design relies on the ADDRESS
 * semantic. NVIDIA docs do not explicitly commit to it, but PyTorch / vLLM /
 * llama.cpp all rely on it through their tensor abstractions.
 *
 * This probe captures one graph that:
 *   1. cudaMemcpyAsync host -> device (4 bytes)
 *   2. kernel reads *d_src, writes to *d_dst (with optional doubling)
 *
 * Then mutates the host buffer across two replays and asserts the kernel
 * output reflects the mutated value. Prints PASS or FAIL with concrete
 * numbers. Exit code: 0 on PASS, 1 on FAIL, 2 on infrastructure error.
 *
 * Build: nvcc -O2 -std=c++17 -arch=sm_120 cuda_graph_memcpy_probe.cu -o probe
 * Run:   ./probe
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "FAIL infra: %s -> %s\n", #call, cudaGetErrorString(_e)); \
        return 2; \
    } \
} while (0)

__global__ static void read_and_double_kernel(const uint32_t *src, uint32_t *dst) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *dst = (*src) * 2u;
    }
}

int main(void) {
    /* Init: device, pinned host buffer, device buffers. */
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    uint32_t *h_src = NULL;
    CHECK(cudaHostAlloc((void**)&h_src, sizeof(uint32_t), cudaHostAllocDefault));
    *h_src = 0xdeadbeefu;  /* sentinel - replaced before capture */

    uint32_t *d_src = NULL;
    uint32_t *d_dst = NULL;
    CHECK(cudaMalloc((void**)&d_src, sizeof(uint32_t)));
    CHECK(cudaMalloc((void**)&d_dst, sizeof(uint32_t)));

    /* Capture: H2D memcpy + doubling kernel. */
    *h_src = 7u;  /* value at capture time */
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CHECK(cudaMemcpyAsync(d_src, h_src, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    read_and_double_kernel<<<1, 1, 0, stream>>>(d_src, d_dst);
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
    CHECK(cudaGraphDestroy(graph));

    /* Replay 1: use the value baked at capture. */
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    uint32_t got1 = 0;
    CHECK(cudaMemcpy(&got1, d_dst, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    fprintf(stderr, "replay1: host_src=%u expected=14 got=%u\n", *h_src, got1);

    /* Mutate host buffer between replays. */
    *h_src = 100u;

    /* Replay 2: if the captured memcpy is ADDRESS-bound, dst should be 200. */
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    uint32_t got2 = 0;
    CHECK(cudaMemcpy(&got2, d_dst, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    fprintf(stderr, "replay2: host_src=%u expected=200 got=%u\n", *h_src, got2);

    /* Mutate again to be thorough. */
    *h_src = 42u;
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    uint32_t got3 = 0;
    CHECK(cudaMemcpy(&got3, d_dst, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    fprintf(stderr, "replay3: host_src=%u expected=84 got=%u\n", *h_src, got3);

    /* Verdict. */
    int pass = (got1 == 14u && got2 == 200u && got3 == 84u);
    if (pass) {
        printf("PASS: captured memcpy is ADDRESS-bound. Design is sound.\n");
    } else if (got1 == 14u && got2 == 14u && got3 == 14u) {
        printf("FAIL: captured memcpy is VALUE-bound. Mutations ignored. Pivot to option 2.\n");
    } else {
        printf("FAIL: unexpected results. got1=%u got2=%u got3=%u\n", got1, got2, got3);
    }

    /* Cleanup. */
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFreeHost(h_src));
    CHECK(cudaStreamDestroy(stream));
    return pass ? 0 : 1;
}
