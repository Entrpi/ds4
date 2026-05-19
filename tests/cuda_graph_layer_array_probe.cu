/*
 * Layer-array probe for the R6 substrate (see plan doc section 15).
 *
 * Validates the load-bearing pattern that the Step 4b per-layer scalar
 * substrate depends on:
 *
 *   R6a -- A captured graph contains N kernel nodes whose pointer args
 *          are derived as `&d_array[il]` for il in 0..N-1.  Each kernel
 *          reads from its own offset on every replay.  The base pointer
 *          d_array is stable for the session; the per-element pointer
 *          baked into each node is also stable (different per node, but
 *          the same across replays of that node).  This is the array-
 *          of-43 design's core mechanic: per-layer captured graphs each
 *          hold &g_layer_dev[il] in their arg list, indexed by il.
 *
 *   R6b -- A SINGLE captured memcpy at the start of the graph moves the
 *          full N-entry host array to d_array.  Subsequent replays with
 *          mutated host contents propagate through the captured memcpy
 *          (address-bound semantic, already validated by the memcpy
 *          probe at cfbe1b8).  Per-layer kernels later in the graph
 *          read the freshly-propagated values via their baked &d_array[il]
 *          arg.  Two host buffers are NOT required for correctness here:
 *          the captured memcpy's source address is baked at capture time,
 *          so it always reads the SAME host buffer (we mutate its
 *          contents between replays).  Double-buffering is a CPU-side
 *          defensive pattern for hypothetical async-sampling futures and
 *          does not affect capture semantics.
 *
 * This probe also serves as a no-regression check for the existing
 * cudaMemcpyAsync address-bound semantic (cfbe1b8) and the R1
 * base+scalar pattern (b0a09fc) on the N>1 case.
 *
 * Probe layout (mirrors fragment_probe.cu shape; ~200 lines):
 *
 *   N        = 8 layers (small analogue of 43; same semantic, fast probe)
 *   STRIDE   = 4 uint32 per row
 *   h_array  : pinned host array of N entries (one uint32 multiplier each)
 *   d_array  : device array of N entries; address baked into kernel args
 *   d_src    : 1 row of STRIDE input sentinels
 *   d_dst    : N rows of STRIDE output
 *
 * One outer captured graph contains, in order:
 *
 *   1. cudaMemcpyAsync(d_array, h_array, N*sizeof(uint32_t), H2D, stream)
 *      -- propagates the next replay's per-layer multipliers via the
 *      address-bound captured-memcpy semantic.
 *
 *   2. For each il in 0..N-1:
 *      multiply_row_kernel<<<...>>>(
 *          d_dst + il*STRIDE,           // baked per-layer destination
 *          d_src,                       // shared input
 *          &d_array[il],                // BAKED PER-LAYER POINTER
 *                                       // -- the R6 substrate analogue
 *          STRIDE);
 *
 * Replay 1: h_array[il] = il + 1.  Expect d_dst[il][k] == d_src[k] * (il+1).
 * Replay 2: h_array[il] = il + 100.  Expect d_dst[il][k] == d_src[k] * (il+100).
 * Replay 3: h_array[il] = 17 (uniform).  Expect d_dst[il][k] == d_src[k] * 17.
 *
 * Exit codes:
 *   0 -- PASS, R6a and R6b both hold
 *   1 -- FAIL, per-layer baked offset or host-source propagation wrong
 *   2 -- FAIL, infrastructure (CUDA error, alloc failure, etc.)
 *
 * Build: nvcc -O2 -std=c++17 -arch=sm_120 cuda_graph_layer_array_probe.cu \
 *        -o layer_array_probe
 * Run:   ./layer_array_probe
 */
#include <cuda_runtime.h>
#include <cstdint>
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

/* Per-"layer" kernel.  Reads `*scalar` (a baked pointer into the per-layer
 * array) at execution time and multiplies the row.  This is the structural
 * analogue of fp8_kv_quantize_row_kernel reading ls->comp_row from a per-
 * layer baked pointer in the array-of-43 substrate. */
__global__ static void multiply_row_kernel(uint32_t *dst,
                                           const uint32_t *src,
                                           const uint32_t *scalar,
                                           uint32_t stride) {
    uint32_t m = *scalar;
    uint32_t i = threadIdx.x;
    if (i < stride) {
        dst[i] = src[i] * m;
    }
}

int main(void) {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "device: %s (sm_%d%d)\n",
            prop.name, prop.major, prop.minor);

    const uint32_t N      = 8;   /* "layers" */
    const uint32_t STRIDE = 4;   /* uint32 per row */

    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    /* Pinned host array.  R2 / address-bound captured memcpy requires a
     * pinned source so the captured async H2D doesn't silently fall back
     * to a synchronous copy under ThreadLocal capture. */
    uint32_t *h_array = NULL;
    CHECK(cudaHostAlloc((void **)&h_array, N * sizeof(uint32_t),
                        cudaHostAllocDefault));
    for (uint32_t il = 0; il < N; ++il) h_array[il] = 0xdeadbeefu;

    /* Device array.  Address is baked into each kernel's `scalar` arg
     * as `&d_array[il]`. */
    uint32_t *d_array = NULL;
    CHECK(cudaMalloc((void **)&d_array, N * sizeof(uint32_t)));

    /* Sentinel input. */
    uint32_t h_src[STRIDE] = { 11u, 13u, 17u, 19u };  /* primes for clarity */
    uint32_t *d_src = NULL;
    CHECK(cudaMalloc((void **)&d_src, STRIDE * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_src, h_src, STRIDE * sizeof(uint32_t),
                     cudaMemcpyHostToDevice));

    /* Output: N rows of STRIDE uint32. */
    uint32_t *d_dst = NULL;
    CHECK(cudaMalloc((void **)&d_dst, N * STRIDE * sizeof(uint32_t)));
    CHECK(cudaMemset(d_dst, 0, N * STRIDE * sizeof(uint32_t)));

    /* ---------- Outer capture ---------- */
    /* Initial h_array contents are inert (will be overwritten before
     * first replay).  The captured memcpy bakes the source pointer
     * h_array, not its contents. */
    for (uint32_t il = 0; il < N; ++il) h_array[il] = 0u;

    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    /* (1) Single captured memcpy moves the full N-entry array.  R6 design
     *     uses one memcpy per token regardless of N=43. */
    CHECK(cudaMemcpyAsync(d_array, h_array, N * sizeof(uint32_t),
                          cudaMemcpyHostToDevice, stream));

    /* (2) N captured kernel nodes, each with a baked per-layer pointer
     *     &d_array[il].  This is the array-of-43 substrate's access
     *     pattern.  In ds4 each per-layer cached cudaGraphExec_t holds
     *     exactly the il-th node's baked pointer; here we capture all N
     *     into a single graph for the probe's compactness, but the
     *     semantics are identical: each baked pointer is per-layer
     *     stable, and the kernel reads through it at execution time. */
    for (uint32_t il = 0; il < N; ++il) {
        multiply_row_kernel<<<1, STRIDE, 0, stream>>>(
                d_dst + (uint64_t)il * STRIDE,
                d_src,
                &d_array[il],  /* BAKED per-layer pointer */
                STRIDE);
        CHECK(cudaGetLastError());
    }

    cudaGraph_t graph = NULL;
    CHECK(cudaStreamEndCapture(stream, &graph));
    if (graph == NULL) {
        fprintf(stderr, "FAIL infra: EndCapture produced no graph\n");
        return 2;
    }
    cudaGraphExec_t exec = NULL;
    CHECK(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
    CHECK(cudaGraphDestroy(graph));

    /* Helper: assert d_dst[il][k] == h_src[k] * h_array[il] for all il,k. */
    auto verify = [&](const char *tag) -> int {
        uint32_t got[N * STRIDE];
        CHECK(cudaMemcpy(got, d_dst, N * STRIDE * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost));
        int pass = 1;
        for (uint32_t il = 0; il < N; ++il) {
            for (uint32_t k = 0; k < STRIDE; ++k) {
                uint32_t expect = h_src[k] * h_array[il];
                if (got[il * STRIDE + k] != expect) {
                    if (pass) {
                        fprintf(stderr,
                                "%s FAIL: dst[il=%u][k=%u] = %u, expected %u "
                                "(h_array[%u]=%u, h_src[%u]=%u)\n",
                                tag, il, k, got[il * STRIDE + k], expect,
                                il, h_array[il], k, h_src[k]);
                    }
                    pass = 0;
                }
            }
        }
        fprintf(stderr, "%s: %s\n", tag, pass ? "ok" : "BAD");
        return pass;
    };

    /* ---------- Replay 1: h_array[il] = il + 1 ---------- */
    for (uint32_t il = 0; il < N; ++il) h_array[il] = il + 1u;
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    int pass1 = verify("replay1 (h_array[il]=il+1)");

    /* ---------- Replay 2: h_array[il] = il + 100 ---------- */
    for (uint32_t il = 0; il < N; ++il) h_array[il] = il + 100u;
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    int pass2 = verify("replay2 (h_array[il]=il+100)");

    /* ---------- Replay 3: h_array[il] = 17 (uniform) ---------- */
    for (uint32_t il = 0; il < N; ++il) h_array[il] = 17u;
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    int pass3 = verify("replay3 (h_array[il]=17 uniform)");

    /* ---------- Verdict ---------- */
    int all_pass = pass1 && pass2 && pass3;
    if (all_pass) {
        printf("PASS: per-layer baked pointer + captured memcpy correctly\n"
               "      propagates per-element host writes to per-element\n"
               "      kernel reads across replays.  Array-of-N substrate\n"
               "      design (R6 / Step 4b) is sound on this device.\n");
    } else {
        printf("FAIL: pass1=%d pass2=%d pass3=%d\n", pass1, pass2, pass3);
        printf("      R6 substrate assumption violated.  The array-of-43\n"
               "      design must be revised before Step 4b lands.\n");
    }

    /* Cleanup */
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_array));
    CHECK(cudaFreeHost(h_array));
    CHECK(cudaStreamDestroy(stream));
    return all_pass ? 0 : 1;
}
