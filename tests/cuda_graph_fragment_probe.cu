/*
 * Fragment-capture probe for the two harder load-bearing semantics in the
 * full-layer-graph design (see local/docs/ds4_full_layer_graph_capture_plan.html
 * sections R1 and R3):
 *
 *   R1: Can a captured kernel compute its destination address as
 *       base + (*device_scalar) * stride and have each replay write to
 *       the row indexed by the current scalar value?
 *
 *   R3: Can an "inner helper" that normally opens its own
 *       cudaStreamBeginCapture/EndCapture be safely converted to a
 *       bypass branch that skips its own begin/end when an outer
 *       capture is active on the current thread?
 *
 * Both are load-bearing for the layer-graph design. R1 is how we eliminate
 * the comp_row_view / index_row_view transient pointer baking problem.
 * R3 is how the existing Step 8/8.2 dense and routed-MoE inner captures
 * coexist with a wider per-layer outer capture without hitting nested
 * cudaStreamBeginCapture (which is illegal).
 *
 * Probe layout:
 *
 *   d_src   : 4 uint32 sentinels (one row's worth of data)
 *   d_dst   : 3 rows x 4 uint32 destination ring buffer
 *   h_row   : pinned host scalar (the "row index" analogue of s->comp_row)
 *   d_row   : device-side mirror of h_row
 *   d_mid   : scratch buffer that the "inner helper" writes (doubled src)
 *
 * One outer captured graph contains, in order:
 *
 *   1. cudaMemcpyAsync(d_row, h_row, 4, H2D, outer_stream)  -- propagates
 *      the next replay's row index via the load-bearing address-bound
 *      captured-memcpy semantic.
 *   2. inner_helper(d_mid, d_src) -- a routine that normally opens its
 *      own capture+instantiate+launch around a "doubling" kernel. With
 *      outer_capture_active() true, it must skip its own Begin/End and
 *      let the kernel launch be captured into the outer graph instead.
 *   3. emit_row_kernel<<<...>>>(d_dst, d_mid, d_row, stride) -- reads
 *      *d_row at execution time, computes dst = d_dst + (*d_row)*stride,
 *      writes d_mid into that row. This is the kernel-side analogue of
 *      the rewritten ds4_gpu_dsv4_fp8_kv_quantize_kernel.
 *
 * The probe then replays the graph TWICE with different host row values:
 *
 *   Replay 1 (h_row=0): row 0 of d_dst becomes 2*sentinel; rows 1,2 zero.
 *   Replay 2 (h_row=1): row 1 of d_dst becomes 2*sentinel; row 0 still
 *                       holds replay-1's writes; row 2 still zero.
 *
 * Plus a standalone test of inner_helper outside any outer capture to
 * confirm the existing capture path still works unchanged.
 *
 * Exit codes:
 *   0 -- PASS, both assertions hold
 *   1 -- FAIL, captured-row write or bypass behavior wrong
 *   2 -- FAIL, infrastructure (CUDA error, alloc failure, etc.)
 *
 * Build: nvcc -O2 -std=c++17 -arch=sm_120 cuda_graph_fragment_probe.cu \
 *        -o fragment_probe
 * Run:   ./fragment_probe
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

/* Mimics ds4_cuda.cu's thread-local outer-capture stream. ds4 uses the
 * same pattern (t_ds4_capture_stream) and inner captures must consult it
 * before opening their own Begin/End. */
static thread_local cudaStream_t t_outer_capture_stream = (cudaStream_t)0;

static inline cudaStream_t outer_capture_stream(void) {
    return t_outer_capture_stream;
}
static inline int outer_capture_active(void) {
    return t_outer_capture_stream != (cudaStream_t)0;
}
static inline void outer_capture_set(cudaStream_t s) {
    t_outer_capture_stream = s;
}

/* The "doubling" kernel that the inner helper drives. Under outer
 * capture it gets launched directly onto the outer stream; under
 * standalone, the inner helper captures a tiny graph around it. */
__global__ static void double_kernel(uint32_t *dst, const uint32_t *src, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * 2u;
}

/* The emit-row kernel: reads *d_row at execution time, writes one row
 * of d_dst from d_mid. This is the load-bearing R1 case -- if CUDA
 * captures the launch in a way that defers the base+scalar arithmetic
 * to replay time, this works. If somehow the captured kernel baked the
 * value of *d_row at capture time, replays would all write row 0. */
__global__ static void emit_row_kernel(uint32_t *d_dst,
                                       const uint32_t *d_mid,
                                       const uint32_t *d_row,
                                       uint32_t stride) {
    uint32_t row = *d_row;
    uint32_t i = threadIdx.x;
    if (i < stride) {
        d_dst[row * stride + i] = d_mid[i];
    }
}

/* Inner helper. Mirrors the shape of ds4_cuda.cu's Step 8 / 8.2 inner
 * captures: under no outer capture it opens its own Begin/End around the
 * kernel and instantiates/launches a private graph. Under outer
 * capture it bypasses its own Begin/End and launches the kernel directly
 * onto the outer stream so it gets captured into the outer graph.
 *
 * Returns 0 on success, nonzero on infrastructure failure. */
static int inner_helper(uint32_t *d_dst,
                        const uint32_t *d_src,
                        uint32_t n,
                        cudaStream_t fallback_stream) {
    cudaStream_t s;
    if (outer_capture_active()) {
        /* R3 bypass: outer capture is active. Skip our own Begin/End;
         * launch the kernel on the outer stream so it lands in the
         * outer graph naturally. */
        s = outer_capture_stream();
        double_kernel<<<1, n, 0, s>>>(d_dst, d_src, n);
        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            fprintf(stderr, "FAIL infra: inner double_kernel bypass launch -> %s\n",
                    cudaGetErrorString(le));
            return 2;
        }
        return 0;
    }
    /* Standalone path: open our own private capture, instantiate, launch.
     * This is what the existing Step 8/8.2 inner captures do today. */
    s = fallback_stream;
    cudaError_t ge = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (ge != cudaSuccess) {
        fprintf(stderr, "FAIL infra: inner cudaStreamBeginCapture -> %s\n",
                cudaGetErrorString(ge));
        return 2;
    }
    double_kernel<<<1, n, 0, s>>>(d_dst, d_src, n);
    cudaGraph_t g;
    ge = cudaStreamEndCapture(s, &g);
    if (ge != cudaSuccess) {
        fprintf(stderr, "FAIL infra: inner cudaStreamEndCapture -> %s\n",
                cudaGetErrorString(ge));
        return 2;
    }
    cudaGraphExec_t e;
    ge = cudaGraphInstantiate(&e, g, NULL, NULL, 0);
    if (ge != cudaSuccess) {
        fprintf(stderr, "FAIL infra: inner cudaGraphInstantiate -> %s\n",
                cudaGetErrorString(ge));
        cudaGraphDestroy(g);
        return 2;
    }
    cudaGraphDestroy(g);
    ge = cudaGraphLaunch(e, s);
    if (ge != cudaSuccess) {
        fprintf(stderr, "FAIL infra: inner cudaGraphLaunch -> %s\n",
                cudaGetErrorString(ge));
        cudaGraphExecDestroy(e);
        return 2;
    }
    cudaGraphExecDestroy(e);
    return 0;
}

int main(void) {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "device: %s (sm_%d%d)\n",
            prop.name, prop.major, prop.minor);

    cudaStream_t outer_stream;
    CHECK(cudaStreamCreateWithFlags(&outer_stream, cudaStreamNonBlocking));
    cudaStream_t inner_fallback;
    CHECK(cudaStreamCreateWithFlags(&inner_fallback, cudaStreamNonBlocking));

    /* Pinned host scalar (R2 / address-bound captured memcpy semantic
     * requires pinned source for the async copy not to silently
     * fall back to synchronous). */
    uint32_t *h_row = NULL;
    CHECK(cudaHostAlloc((void **)&h_row, sizeof(uint32_t),
                        cudaHostAllocDefault));
    *h_row = 0xdeadbeefu;  /* will be overwritten before capture */

    /* Device-side row scalar (the analogue of g_decode_dev->comp_row). */
    uint32_t *d_row = NULL;
    CHECK(cudaMalloc((void **)&d_row, sizeof(uint32_t)));

    /* Sentinel input: 4 uint32 with recognizable bit patterns. */
    const uint32_t STRIDE = 4;
    const uint32_t NROWS  = 3;
    uint32_t h_src[STRIDE] = { 0xa5a50000u, 0xa5a50001u,
                               0xa5a50002u, 0xa5a50003u };
    uint32_t *d_src = NULL;
    CHECK(cudaMalloc((void **)&d_src, STRIDE * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_src, h_src, STRIDE * sizeof(uint32_t),
                     cudaMemcpyHostToDevice));

    /* Scratch buffer that the inner helper writes (2 * src). */
    uint32_t *d_mid = NULL;
    CHECK(cudaMalloc((void **)&d_mid, STRIDE * sizeof(uint32_t)));
    CHECK(cudaMemset(d_mid, 0, STRIDE * sizeof(uint32_t)));

    /* Destination ring buffer: NROWS rows, each STRIDE uint32. */
    uint32_t *d_dst = NULL;
    CHECK(cudaMalloc((void **)&d_dst, NROWS * STRIDE * sizeof(uint32_t)));
    CHECK(cudaMemset(d_dst, 0, NROWS * STRIDE * sizeof(uint32_t)));

    /* Expected per-row value after a replay that targets that row:
     * each lane gets 2 * h_src[lane]. */
    uint32_t expected_row[STRIDE];
    for (uint32_t i = 0; i < STRIDE; ++i) {
        expected_row[i] = h_src[i] * 2u;
    }

    /* ---------- Outer capture ---------- */
    *h_row = 0u;  /* value baked into the graph as a starting point */

    outer_capture_set(outer_stream);
    CHECK(cudaStreamBeginCapture(outer_stream,
                                 cudaStreamCaptureModeThreadLocal));

    /* (1) Captured memcpy: host scalar -> device scalar. Address-bound
     *     so future *h_row updates propagate at the next replay. */
    CHECK(cudaMemcpyAsync(d_row, h_row, sizeof(uint32_t),
                          cudaMemcpyHostToDevice, outer_stream));

    /* (2) Inner helper under outer capture: must bypass its own
     *     Begin/End and let double_kernel land in the outer graph. */
    int ih_rc = inner_helper(d_mid, d_src, STRIDE, inner_fallback);
    if (ih_rc != 0) {
        cudaStreamEndCapture(outer_stream, NULL);
        outer_capture_set((cudaStream_t)0);
        return ih_rc;
    }

    /* (3) Emit-row kernel: reads *d_row at execution time, writes
     *     d_mid into d_dst[row]. This is the R1 base+scalar pattern. */
    emit_row_kernel<<<1, STRIDE, 0, outer_stream>>>(d_dst, d_mid, d_row, STRIDE);
    CHECK(cudaGetLastError());

    cudaGraph_t graph = NULL;
    CHECK(cudaStreamEndCapture(outer_stream, &graph));
    outer_capture_set((cudaStream_t)0);
    if (graph == NULL) {
        fprintf(stderr, "FAIL infra: EndCapture produced no graph\n");
        return 2;
    }
    cudaGraphExec_t exec = NULL;
    CHECK(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
    CHECK(cudaGraphDestroy(graph));

    /* ---------- Replay 1 at row=0 ---------- */
    *h_row = 0u;
    CHECK(cudaGraphLaunch(exec, outer_stream));
    CHECK(cudaStreamSynchronize(outer_stream));

    uint32_t got_dst[NROWS * STRIDE];
    CHECK(cudaMemcpy(got_dst, d_dst,
                     NROWS * STRIDE * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));

    int pass = 1;
    /* Row 0 should equal expected_row; rows 1 and 2 still zero. */
    for (uint32_t i = 0; i < STRIDE; ++i) {
        if (got_dst[0 * STRIDE + i] != expected_row[i]) pass = 0;
        if (got_dst[1 * STRIDE + i] != 0u) pass = 0;
        if (got_dst[2 * STRIDE + i] != 0u) pass = 0;
    }
    fprintf(stderr,
            "replay1 row=0: dst[0]={%08x %08x %08x %08x} dst[1]={%08x...} dst[2]={%08x...} %s\n",
            got_dst[0], got_dst[1], got_dst[2], got_dst[3],
            got_dst[4], got_dst[8],
            pass ? "ok" : "BAD");

    /* ---------- Replay 2 at row=1 ---------- */
    *h_row = 1u;
    CHECK(cudaGraphLaunch(exec, outer_stream));
    CHECK(cudaStreamSynchronize(outer_stream));

    CHECK(cudaMemcpy(got_dst, d_dst,
                     NROWS * STRIDE * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));

    int pass2 = 1;
    /* Row 0 retains replay-1 writes; row 1 now equals expected_row;
     * row 2 still zero. */
    for (uint32_t i = 0; i < STRIDE; ++i) {
        if (got_dst[0 * STRIDE + i] != expected_row[i]) pass2 = 0;
        if (got_dst[1 * STRIDE + i] != expected_row[i]) pass2 = 0;
        if (got_dst[2 * STRIDE + i] != 0u) pass2 = 0;
    }
    fprintf(stderr,
            "replay2 row=1: dst[0]={%08x...} dst[1]={%08x %08x %08x %08x} dst[2]={%08x...} %s\n",
            got_dst[0],
            got_dst[4], got_dst[5], got_dst[6], got_dst[7],
            got_dst[8],
            pass2 ? "ok" : "BAD");

    /* ---------- Replay 3 at row=2 (a row we never previously touched) ---- */
    *h_row = 2u;
    CHECK(cudaGraphLaunch(exec, outer_stream));
    CHECK(cudaStreamSynchronize(outer_stream));

    CHECK(cudaMemcpy(got_dst, d_dst,
                     NROWS * STRIDE * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));

    int pass3 = 1;
    for (uint32_t i = 0; i < STRIDE; ++i) {
        if (got_dst[0 * STRIDE + i] != expected_row[i]) pass3 = 0;
        if (got_dst[1 * STRIDE + i] != expected_row[i]) pass3 = 0;
        if (got_dst[2 * STRIDE + i] != expected_row[i]) pass3 = 0;
    }
    fprintf(stderr,
            "replay3 row=2: dst[2]={%08x %08x %08x %08x} %s\n",
            got_dst[8], got_dst[9], got_dst[10], got_dst[11],
            pass3 ? "ok" : "BAD");

    /* ---------- Standalone inner-helper test (R3 no-regression) ----- */
    /* Outside any outer capture, inner_helper should take its own
     * capture+instantiate+launch path and write 2*src into a fresh
     * destination buffer. */
    uint32_t *d_solo = NULL;
    CHECK(cudaMalloc((void **)&d_solo, STRIDE * sizeof(uint32_t)));
    CHECK(cudaMemset(d_solo, 0, STRIDE * sizeof(uint32_t)));

    if (outer_capture_active()) {
        fprintf(stderr, "FAIL infra: outer_capture_active() true outside capture\n");
        return 2;
    }
    int solo_rc = inner_helper(d_solo, d_src, STRIDE, inner_fallback);
    if (solo_rc != 0) return solo_rc;
    CHECK(cudaStreamSynchronize(inner_fallback));

    uint32_t got_solo[STRIDE];
    CHECK(cudaMemcpy(got_solo, d_solo, STRIDE * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));

    int pass_solo = 1;
    for (uint32_t i = 0; i < STRIDE; ++i) {
        if (got_solo[i] != expected_row[i]) pass_solo = 0;
    }
    fprintf(stderr, "standalone inner: solo={%08x %08x %08x %08x} %s\n",
            got_solo[0], got_solo[1], got_solo[2], got_solo[3],
            pass_solo ? "ok" : "BAD");

    /* ---------- Verdict ---------- */
    int all_pass = pass && pass2 && pass3 && pass_solo;
    if (all_pass) {
        printf("PASS: base+scalar emit-row write and inner-capture bypass both work.\n"
               "      Design assumptions R1 and R3 hold on this device.\n");
    } else {
        printf("FAIL: pass=%d pass2=%d pass3=%d pass_solo=%d.\n", pass, pass2, pass3, pass_solo);
        if (!pass || !pass2 || !pass3) {
            printf("      R1 violated: captured kernel did NOT read *d_row at replay time.\n"
                   "      Design must reframe row indexing.\n");
        }
        if (!pass_solo) {
            printf("      R3 standalone path regressed: inner helper failed outside outer capture.\n");
        }
    }

    /* Cleanup */
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaFree(d_solo));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_mid));
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_row));
    CHECK(cudaFreeHost(h_row));
    CHECK(cudaStreamDestroy(inner_fallback));
    CHECK(cudaStreamDestroy(outer_stream));
    return all_pass ? 0 : 1;
}
