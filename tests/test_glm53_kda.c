#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void require_close(const char *what, float actual, float expected, float tolerance) {
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "%s: got %.9g, expected %.9g (tolerance %.9g)\n",
                what, actual, expected, tolerance);
        exit(1);
    }
}

static uint16_t f32_to_bf16(float value) {
    union { float f; uint32_t u; } bits = { .f = value };
    const uint32_t rounding = 0x7fffu + ((bits.u >> 16) & 1u);
    return (uint16_t)((bits.u + rounding) >> 16);
}

static float bf16_to_f32(uint16_t value) {
    union { uint32_t u; float f; } bits = { .u = (uint32_t)value << 16 };
    return bits.f;
}

int main(void) {
    enum {
        D = 128,
        HEADS = 2,
        PROJECTION = HEADS * D,
        TOKENS = 3,
        Q_CONV_OFFSET = 0,
        K_CONV_OFFSET = 4096,
        V_CONV_OFFSET = 8192,
        A_LOG_OFFSET = 12288,
        DT_BIAS_OFFSET = 16384,
        NORM_OFFSET = 20480,
        POOL_NORM_OFFSET = 22528,
        POOL_BIAS_OFFSET = 24576,
        POOL_APE_OFFSET = 28672,
        BF16_OFFSET = 32768,
        BF16_IN = 64,
        BF16_OUT = 64,
        BF16_ROWS = 16,
        MODEL_BYTES = 65536,
    };

    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    float *q_conv = (float *)(model + Q_CONV_OFFSET);
    float *k_conv = (float *)(model + K_CONV_OFFSET);
    float *v_conv = (float *)(model + V_CONV_OFFSET);
    float *a_log = (float *)(model + A_LOG_OFFSET);
    float *dt_bias = (float *)(model + DT_BIAS_OFFSET);
    float *norm = (float *)(model + NORM_OFFSET);
    for (uint32_t channel = 0; channel < PROJECTION; channel++) {
        q_conv[channel * 4u + 3u] = 1.0f;
        k_conv[channel * 4u + 3u] = 1.0f;
        v_conv[channel * 4u + 3u] = 1.0f;
        dt_bias[channel] = 0.0f;
    }
    for (uint32_t head = 0; head < HEADS; head++) a_log[head] = 0.0f;
    for (uint32_t d = 0; d < D; d++) norm[d] = 1.0f;
    float *pool_norm = (float *)(model + POOL_NORM_OFFSET);
    float *pool_bias = (float *)(model + POOL_BIAS_OFFSET);
    uint16_t *pool_ape = (uint16_t *)(model + POOL_APE_OFFSET);
    for (uint32_t d = 0; d < D; d++) {
        pool_norm[d] = 0.75f + 0.002f * (float)d;
        pool_bias[d] = -0.1f + 0.001f * (float)d;
        for (uint32_t r = 0; r < 4u; r++) {
            pool_ape[r * D + d] = f32_to_bf16(
                0.03f * (float)r - 0.0005f * (float)d);
        }
    }

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES), "model map registration");

    uint16_t *bf16_weights = (uint16_t *)(model + BF16_OFFSET);
    for (uint32_t o = 0; o < BF16_OUT; o++) {
        for (uint32_t i = 0; i < BF16_IN; i++) {
            const float value = 0.002f * (float)((int)(o % 11u) - 5) +
                                0.001f * (float)((int)(i % 13u) - 6);
            bf16_weights[o * BF16_IN + i] = f32_to_bf16(value);
        }
    }
    float bf16_input[BF16_ROWS * BF16_IN];
    float bf16_expected[BF16_ROWS * BF16_OUT];
    for (uint32_t row = 0; row < BF16_ROWS; row++) {
        for (uint32_t i = 0; i < BF16_IN; i++) {
            bf16_input[row * BF16_IN + i] =
                0.02f * (float)((int)(i % 17u) - 8) + 0.005f * (float)row;
        }
        for (uint32_t o = 0; o < BF16_OUT; o++) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < BF16_IN; i++) {
                sum += bf16_to_f32(bf16_weights[o * BF16_IN + i]) *
                       bf16_input[row * BF16_IN + i];
            }
            bf16_expected[row * BF16_OUT + o] = sum;
        }
    }
    ds4_gpu_tensor *bf16_x = ds4_gpu_tensor_alloc(sizeof(bf16_input));
    ds4_gpu_tensor *bf16_out = ds4_gpu_tensor_alloc(sizeof(bf16_expected));
    require_ok(bf16_x && bf16_out, "BF16 tensor allocation");
    require_ok(ds4_gpu_tensor_write(bf16_x, 0, bf16_input, sizeof(bf16_input)),
               "BF16 input write");
    require_ok(ds4_gpu_glm53_matmul_bf16(
        bf16_out, model, MODEL_BYTES, BF16_OFFSET,
        BF16_IN, BF16_OUT, bf16_x, 1), "BF16 decode matmul");
    float bf16_actual[BF16_ROWS * BF16_OUT];
    require_ok(ds4_gpu_tensor_read(bf16_out, 0, bf16_actual,
                                   BF16_OUT * sizeof(float)),
               "BF16 decode output read");
    for (uint32_t i = 0; i < BF16_OUT; i++)
        require_close("BF16 decode matmul", bf16_actual[i], bf16_expected[i], 2e-6f);
    require_ok(ds4_gpu_glm53_matmul_bf16(
        bf16_out, model, MODEL_BYTES, BF16_OFFSET,
        BF16_IN, BF16_OUT, bf16_x, BF16_ROWS), "BF16 prefill matmul");
    require_ok(ds4_gpu_tensor_read(bf16_out, 0, bf16_actual, sizeof(bf16_actual)),
               "BF16 prefill output read");
    for (uint32_t i = 0; i < BF16_ROWS * BF16_OUT; i++)
        require_close("BF16 prefill matmul", bf16_actual[i], bf16_expected[i], 2e-4f);

    enum { COMPACT_LORA = 512, COMPACT_TOKENS = 2, COMPACT_CAP = 4 };
    float compact_norm[COMPACT_TOKENS * COMPACT_LORA];
    float compact_raw[COMPACT_TOKENS * COMPACT_LORA];
    for (uint32_t i = 0; i < COMPACT_TOKENS * COMPACT_LORA; i++) {
        compact_norm[i] = 0.001f * (float)((int)(i % 101u) - 50);
        compact_raw[i] = compact_norm[i] + 1.0f;
    }
    ds4_gpu_tensor *compact_norm_gpu =
        ds4_gpu_tensor_alloc(sizeof(compact_norm));
    ds4_gpu_tensor *compact_raw_gpu =
        ds4_gpu_tensor_alloc(sizeof(compact_raw));
    ds4_gpu_tensor *compact_cache_gpu = ds4_gpu_tensor_alloc(
        (uint64_t)COMPACT_CAP * COMPACT_LORA * sizeof(float));
    ds4_gpu_tensor *zero_rope_cache_gpu = ds4_gpu_tensor_alloc(1u);
    require_ok(compact_norm_gpu && compact_raw_gpu && compact_cache_gpu &&
               zero_rope_cache_gpu, "zero-RoPE compact tensor allocation");
    require_ok(ds4_gpu_tensor_write(compact_norm_gpu, 0, compact_norm,
                                    sizeof(compact_norm)),
               "zero-RoPE compact norm write");
    require_ok(ds4_gpu_tensor_write(compact_raw_gpu, 0, compact_raw,
                                    sizeof(compact_raw)),
               "zero-RoPE compact raw write");
    require_ok(ds4_gpu_glm_store_compact_kv_tensor(
        compact_cache_gpu, zero_rope_cache_gpu,
        compact_norm_gpu, compact_raw_gpu,
        1, COMPACT_TOKENS, COMPACT_CAP,
        COMPACT_LORA, COMPACT_LORA, 0, false),
        "GLM-5.3 zero-RoPE compact KV store");
    float compact_actual[COMPACT_CAP * COMPACT_LORA];
    require_ok(ds4_gpu_tensor_read(
        compact_cache_gpu,
        (uint64_t)COMPACT_LORA * sizeof(float),
        compact_actual,
        sizeof(compact_norm)),
        "zero-RoPE compact cache read");
    for (uint32_t i = 0; i < COMPACT_TOKENS * COMPACT_LORA; i++) {
        require_close("zero-RoPE compact KV", compact_actual[i],
                      compact_norm[i], 0.0f);
    }
    ds4_gpu_tensor_free(zero_rope_cache_gpu);
    ds4_gpu_tensor_free(compact_cache_gpu);
    ds4_gpu_tensor_free(compact_raw_gpu);
    ds4_gpu_tensor_free(compact_norm_gpu);

    enum { POOL = 4, POOL_TOKENS = 11, POOL_CAP = 16, POOL_COUNT = 4 };
    float pool_raw[POOL_TOKENS * D];
    float pool_gate_values[POOL_TOKENS * D];
    for (uint32_t t = 0; t < POOL_TOKENS; t++) {
        for (uint32_t d = 0; d < D; d++) {
            pool_raw[t * D + d] =
                0.01f * (float)t + 0.002f * (float)((int)(d % 19u) - 9);
            pool_gate_values[t * D + d] =
                -0.2f + 0.07f * (float)t - 0.001f * (float)(d % 23u);
        }
    }
    ds4_gpu_tensor *pool_cache =
        ds4_gpu_tensor_alloc((uint64_t)POOL_COUNT * D * sizeof(float));
    const uint64_t pool_tail_bytes =
        (uint64_t)POOL * D * sizeof(float);
    ds4_gpu_tensor *pool_tail_k =
        ds4_gpu_tensor_alloc(2u * pool_tail_bytes);
    ds4_gpu_tensor *pool_tail_gate =
        ds4_gpu_tensor_view(pool_tail_k, pool_tail_bytes, pool_tail_bytes);
    ds4_gpu_tensor *pool_raw_gpu =
        ds4_gpu_tensor_alloc(8u * D * sizeof(float));
    ds4_gpu_tensor *pool_gate_gpu =
        ds4_gpu_tensor_alloc(8u * D * sizeof(float));
    require_ok(pool_cache && pool_tail_k && pool_tail_gate &&
               pool_raw_gpu && pool_gate_gpu, "pool tensor allocation");
    require_ok(ds4_gpu_tensor_fill_f32(pool_cache, 0.0f,
                                       (uint64_t)POOL_COUNT * D),
               "pool cache clear");
    require_ok(ds4_gpu_tensor_fill_f32(pool_tail_k, 0.0f,
                                       (uint64_t)POOL * D),
               "pool K tail clear");
    require_ok(ds4_gpu_tensor_fill_f32(pool_tail_gate, 0.0f,
                                       (uint64_t)POOL * D),
               "pool gate tail clear");
    const uint32_t pool_chunks[] = {3, 8};
    uint32_t pool_pos = 0;
    for (uint32_t c = 0;
         c < sizeof(pool_chunks) / sizeof(pool_chunks[0]); c++) {
        const uint32_t rows = pool_chunks[c];
        require_ok(ds4_gpu_tensor_write(pool_raw_gpu, 0,
                                        pool_raw + (uint64_t)pool_pos * D,
                                        (uint64_t)rows * D * sizeof(float)),
                   "pool raw write");
        require_ok(ds4_gpu_tensor_write(pool_gate_gpu, 0,
                                        pool_gate_values + (uint64_t)pool_pos * D,
                                        (uint64_t)rows * D * sizeof(float)),
                   "pool gate write");
        require_ok(ds4_gpu_glm53_indexer_pool_update_tensor(
            pool_cache, pool_tail_k, pool_tail_gate,
            pool_raw_gpu, pool_gate_gpu,
            model, MODEL_BYTES, POOL_NORM_OFFSET, POOL_BIAS_OFFSET, POOL_APE_OFFSET,
            pool_pos, rows, POOL_CAP, D, POOL, 1e-6f, false),
            "GLM-5.3 indexer pool update");
        pool_pos += rows;
    }
    float pool_actual[POOL_COUNT * D];
    require_ok(ds4_gpu_tensor_read(pool_cache, 0, pool_actual,
                                   sizeof(pool_actual)), "pool cache read");
    for (uint32_t p = 0; p < 2u; p++) {
        float means[POOL], invs[POOL];
        for (uint32_t r = 0; r < POOL; r++) {
            const float *row = pool_raw + (uint64_t)(p * POOL + r) * D;
            float sum = 0.0f;
            for (uint32_t d = 0; d < D; d++) sum += row[d];
            means[r] = sum / (float)D;
            float ss = 0.0f;
            for (uint32_t d = 0; d < D; d++) {
                const float delta = row[d] - means[r];
                ss += delta * delta;
            }
            invs[r] = 1.0f / sqrtf(ss / (float)D + 1e-6f);
        }
        for (uint32_t d = 0; d < D; d++) {
            float logits[POOL], max_logit = -FLT_MAX, denom = 0.0f;
            for (uint32_t r = 0; r < POOL; r++) {
                logits[r] = pool_gate_values[(uint64_t)(p * POOL + r) * D + d] +
                    bf16_to_f32(pool_ape[r * D + d]);
                if (logits[r] > max_logit) max_logit = logits[r];
            }
            for (uint32_t r = 0; r < POOL; r++) {
                logits[r] = expf(logits[r] - max_logit);
                denom += logits[r];
            }
            float expected_pool = 0.0f;
            for (uint32_t r = 0; r < POOL; r++) {
                const float value =
                    (pool_raw[(uint64_t)(p * POOL + r) * D + d] - means[r]) *
                    invs[r] * pool_norm[d] + pool_bias[d];
                expected_pool += logits[r] / denom * value;
            }
            require_close("GLM-5.3 pool", pool_actual[p * D + d],
                          expected_pool, 2e-5f);
        }
    }

    enum { SCORE_ROWS = 3, SCORE_TOKENS = 5, SCORE_POS0 = 4 };
    float score_q[SCORE_TOKENS * HEADS * D];
    float score_weights[SCORE_TOKENS * HEADS];
    float score_cache[SCORE_ROWS * D];
    for (uint32_t t = 0; t < SCORE_TOKENS; t++) {
        for (uint32_t h = 0; h < HEADS; h++) {
            score_weights[t * HEADS + h] =
                0.25f + 0.1f * (float)t - 0.05f * (float)h;
            for (uint32_t d = 0; d < D; d++) {
                score_q[((uint64_t)t * HEADS + h) * D + d] =
                    0.01f * (float)(t + 1u) +
                    0.02f * (float)h +
                    0.0001f * (float)d;
            }
        }
    }
    for (uint32_t row = 0; row < SCORE_ROWS; row++) {
        for (uint32_t d = 0; d < D; d++) {
            score_cache[row * D + d] =
                0.03f * (float)(row + 1u) - 0.0002f * (float)d;
        }
    }
    ds4_gpu_tensor *score_q_gpu = ds4_gpu_tensor_alloc(sizeof(score_q));
    ds4_gpu_tensor *score_weights_gpu =
        ds4_gpu_tensor_alloc(sizeof(score_weights));
    ds4_gpu_tensor *score_cache_gpu = ds4_gpu_tensor_alloc(sizeof(score_cache));
    ds4_gpu_tensor *scores_gpu = ds4_gpu_tensor_alloc(
        (uint64_t)SCORE_TOKENS * SCORE_ROWS * sizeof(float));
    require_ok(score_q_gpu && score_weights_gpu && score_cache_gpu && scores_gpu,
               "grouped scorer tensor allocation");
    require_ok(ds4_gpu_tensor_write(score_q_gpu, 0, score_q, sizeof(score_q)),
               "grouped scorer Q write");
    require_ok(ds4_gpu_tensor_write(score_weights_gpu, 0, score_weights,
                                    sizeof(score_weights)),
               "grouped scorer weights write");
    require_ok(ds4_gpu_tensor_write(score_cache_gpu, 0, score_cache,
                                    sizeof(score_cache)),
               "grouped scorer cache write");
    const float score_scale = 0.125f;
    require_ok(ds4_gpu_glm53_indexer_scores_batch_tensor(
        scores_gpu, score_q_gpu, score_weights_gpu, score_cache_gpu,
        SCORE_ROWS, SCORE_TOKENS, SCORE_POS0, POOL,
        HEADS, D, score_scale, false), "GLM-5.3 grouped indexer scores");
    float scores_actual[SCORE_TOKENS * SCORE_ROWS];
    require_ok(ds4_gpu_tensor_read(scores_gpu, 0, scores_actual,
                                   sizeof(scores_actual)),
               "grouped scorer output read");
    for (uint32_t t = 0; t < SCORE_TOKENS; t++) {
        const uint32_t visible = (SCORE_POS0 + t + 1u) / POOL;
        for (uint32_t row = 0; row < SCORE_ROWS; row++) {
            const float actual_score = scores_actual[t * SCORE_ROWS + row];
            if (row >= visible) {
                if (!isinf(actual_score) || actual_score >= 0.0f) {
                    fprintf(stderr,
                            "grouped scorer row %u token %u should be hidden\n",
                            row, t);
                    return 1;
                }
                continue;
            }
            float expected_score = 0.0f;
            for (uint32_t h = 0; h < HEADS; h++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; d++) {
                    dot += score_q[((uint64_t)t * HEADS + h) * D + d] *
                           score_cache[row * D + d];
                }
                expected_score += score_weights[t * HEADS + h] * dot;
            }
            require_close("GLM-5.3 grouped indexer score", actual_score,
                          expected_score * score_scale, 2e-5f);
        }
    }
    ds4_gpu_tensor_free(scores_gpu);
    ds4_gpu_tensor_free(score_cache_gpu);
    ds4_gpu_tensor_free(score_weights_gpu);
    ds4_gpu_tensor_free(score_q_gpu);

    enum { SELECTED_POOLS = 512, INDEX_TOPK = 2048, SELECT_ROWS = 5,
           SELECT_WIDTH = 2051 };
    uint32_t pool_ids[SELECT_ROWS * SELECTED_POOLS];
    for (uint32_t t = 0; t < SELECT_ROWS; t++) {
        for (uint32_t i = 0; i < SELECTED_POOLS; i++) {
            pool_ids[t * SELECTED_POOLS + i] = SELECTED_POOLS - 1u - i;
        }
    }
    ds4_gpu_tensor *pool_ids_gpu = ds4_gpu_tensor_alloc(sizeof(pool_ids));
    ds4_gpu_tensor *raw_ids_gpu =
        ds4_gpu_tensor_alloc((uint64_t)SELECT_ROWS * SELECT_WIDTH * sizeof(uint32_t));
    require_ok(pool_ids_gpu && raw_ids_gpu, "pool selection tensor allocation");
    require_ok(ds4_gpu_tensor_write(pool_ids_gpu, 0, pool_ids, sizeof(pool_ids)),
               "pool selection write");
    require_ok(ds4_gpu_glm53_expand_pool_selection_tensor(
        raw_ids_gpu, pool_ids_gpu, SELECT_ROWS, INDEX_TOPK,
        SELECTED_POOLS, INDEX_TOPK, POOL, SELECT_WIDTH),
        "pool selection expansion");
    uint32_t raw_ids[SELECT_ROWS * SELECT_WIDTH];
    require_ok(ds4_gpu_tensor_read(raw_ids_gpu, 0, raw_ids, sizeof(raw_ids)),
               "pool selection read");
    for (uint32_t t = 0; t < SELECT_ROWS; t++) {
        for (uint32_t i = 0; i < INDEX_TOPK; i++) {
            const uint32_t p = pool_ids[t * SELECTED_POOLS + i / POOL];
            if (raw_ids[t * SELECT_WIDTH + i] != p * POOL + i % POOL) {
                fprintf(stderr, "pool expansion mismatch at row %u slot %u\n", t, i);
                return 1;
            }
        }
        const uint32_t visible = INDEX_TOPK + t + 1u;
        const uint32_t tail_count = visible % POOL;
        for (uint32_t i = 0; i < POOL - 1u; i++) {
            const uint32_t expected_id =
                i < tail_count ? visible - tail_count + i : UINT32_MAX;
            if (raw_ids[t * SELECT_WIDTH + INDEX_TOPK + i] != expected_id) {
                fprintf(stderr, "pool tail mismatch at row %u slot %u\n", t, i);
                return 1;
            }
        }
    }
    ds4_gpu_tensor_free(raw_ids_gpu);
    ds4_gpu_tensor_free(pool_ids_gpu);
    ds4_gpu_tensor_free(pool_gate_gpu);
    ds4_gpu_tensor_free(pool_raw_gpu);
    ds4_gpu_tensor_free(pool_tail_gate);
    ds4_gpu_tensor_free(pool_tail_k);
    ds4_gpu_tensor_free(pool_cache);

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *beta = ds4_gpu_tensor_alloc(HEADS * sizeof(float));
    ds4_gpu_tensor *output_gate = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *conv = ds4_gpu_tensor_alloc(9u * PROJECTION * sizeof(float));
    ds4_gpu_tensor *state = ds4_gpu_tensor_alloc((uint64_t)HEADS * D * D * sizeof(float));
    require_ok(q && k && v && gate && beta && output_gate && out && conv && state,
               "decode tensor allocation");

    float ones[PROJECTION], zeros[PROJECTION], beta_zero[HEADS];
    for (uint32_t i = 0; i < PROJECTION; i++) {
        ones[i] = 1.0f;
        zeros[i] = 0.0f;
    }
    for (uint32_t i = 0; i < HEADS; i++) beta_zero[i] = 0.0f;
    require_ok(ds4_gpu_tensor_write(q, 0, ones, sizeof(ones)), "Q write");
    require_ok(ds4_gpu_tensor_write(k, 0, ones, sizeof(ones)), "K write");
    require_ok(ds4_gpu_tensor_write(v, 0, ones, sizeof(ones)), "V write");
    require_ok(ds4_gpu_tensor_write(gate, 0, zeros, sizeof(zeros)), "gate write");
    require_ok(ds4_gpu_tensor_write(output_gate, 0, zeros, sizeof(zeros)), "output gate write");
    require_ok(ds4_gpu_tensor_write(beta, 0, beta_zero, sizeof(beta_zero)), "beta write");
    require_ok(ds4_gpu_tensor_fill_f32(conv, 0.0f, 9u * PROJECTION), "conv clear");
    require_ok(ds4_gpu_tensor_fill_f32(state, 0.0f, (uint64_t)HEADS * D * D), "state clear");
    require_ok(ds4_gpu_glm53_kda_decode(
        out, conv, state, q, k, v, gate, beta, output_gate,
        model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
        A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
        HEADS, 1, -5.0f, 1e-5f), "KDA decode");
    float actual[PROJECTION];
    require_ok(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)), "output read");
    const float silu_one = 1.0f / (1.0f + expf(-1.0f));
    const float raw = 0.5f * silu_one / sqrtf((float)D);
    const float expected = 0.5f * raw / sqrtf(raw * raw + 1e-5f);
    for (uint32_t i = 0; i < PROJECTION; i++)
        require_close("KDA decode", actual[i], expected, 2e-5f);

    float qs[TOKENS * PROJECTION], ks[TOKENS * PROJECTION];
    float vs[TOKENS * PROJECTION], gates[TOKENS * PROJECTION];
    float output_gates[TOKENS * PROJECTION], betas[TOKENS * HEADS];
    for (uint32_t t = 0; t < TOKENS; t++) {
        for (uint32_t h = 0; h < HEADS; h++)
            betas[t * HEADS + h] = -0.25f + 0.2f * (float)t + 0.1f * (float)h;
        for (uint32_t d = 0; d < PROJECTION; d++) {
            const uint32_t i = t * PROJECTION + d;
            qs[i] = 0.1f + 0.002f * (float)(d % 17u) + 0.03f * (float)t;
            ks[i] = -0.08f + 0.001f * (float)(d % 23u) + 0.02f * (float)t;
            vs[i] = 0.05f - 0.0015f * (float)(d % 13u) + 0.04f * (float)t;
            gates[i] = -0.2f + 0.003f * (float)(d % 11u);
            output_gates[i] = 0.15f - 0.002f * (float)(d % 7u);
        }
    }

    require_ok(ds4_gpu_tensor_fill_f32(conv, 0.0f, 9u * PROJECTION), "decode conv reset");
    require_ok(ds4_gpu_tensor_fill_f32(state, 0.0f, (uint64_t)HEADS * D * D), "decode state reset");
    float decode_outputs[TOKENS * PROJECTION];
    for (uint32_t t = 0; t < TOKENS; t++) {
        const uint32_t off = t * PROJECTION;
        require_ok(ds4_gpu_tensor_write(q, 0, qs + off, PROJECTION * sizeof(float)), "decode Q write");
        require_ok(ds4_gpu_tensor_write(k, 0, ks + off, PROJECTION * sizeof(float)), "decode K write");
        require_ok(ds4_gpu_tensor_write(v, 0, vs + off, PROJECTION * sizeof(float)), "decode V write");
        require_ok(ds4_gpu_tensor_write(gate, 0, gates + off, PROJECTION * sizeof(float)), "decode gate write");
        require_ok(ds4_gpu_tensor_write(output_gate, 0, output_gates + off, PROJECTION * sizeof(float)), "decode output gate write");
        require_ok(ds4_gpu_tensor_write(beta, 0, betas + t * HEADS, HEADS * sizeof(float)), "decode beta write");
        require_ok(ds4_gpu_glm53_kda_decode(
            out, conv, state, q, k, v, gate, beta, output_gate,
            model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
            A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
            HEADS, 1, -5.0f, 1e-5f), "consistency decode");
        require_ok(ds4_gpu_tensor_read(out, 0, decode_outputs + off,
                                       PROJECTION * sizeof(float)), "decode output read");
    }

    ds4_gpu_tensor *pq = ds4_gpu_tensor_alloc(sizeof(qs));
    ds4_gpu_tensor *pk = ds4_gpu_tensor_alloc(sizeof(ks));
    ds4_gpu_tensor *pv = ds4_gpu_tensor_alloc(sizeof(vs));
    ds4_gpu_tensor *pg = ds4_gpu_tensor_alloc(sizeof(gates));
    ds4_gpu_tensor *poutput_gate = ds4_gpu_tensor_alloc(sizeof(output_gates));
    ds4_gpu_tensor *pbeta = ds4_gpu_tensor_alloc(sizeof(betas));
    ds4_gpu_tensor *pout = ds4_gpu_tensor_alloc(sizeof(qs));
    ds4_gpu_tensor *pconv = ds4_gpu_tensor_alloc(9u * PROJECTION * sizeof(float));
    ds4_gpu_tensor *pstate = ds4_gpu_tensor_alloc((uint64_t)HEADS * D * D * sizeof(float));
    require_ok(pq && pk && pv && pg && poutput_gate && pbeta && pout && pconv && pstate,
               "prefill tensor allocation");
    require_ok(ds4_gpu_tensor_write(pq, 0, qs, sizeof(qs)), "prefill Q write");
    require_ok(ds4_gpu_tensor_write(pk, 0, ks, sizeof(ks)), "prefill K write");
    require_ok(ds4_gpu_tensor_write(pv, 0, vs, sizeof(vs)), "prefill V write");
    require_ok(ds4_gpu_tensor_write(pg, 0, gates, sizeof(gates)), "prefill gate write");
    require_ok(ds4_gpu_tensor_write(poutput_gate, 0, output_gates, sizeof(output_gates)), "prefill output gate write");
    require_ok(ds4_gpu_tensor_write(pbeta, 0, betas, sizeof(betas)), "prefill beta write");
    require_ok(ds4_gpu_tensor_fill_f32(pconv, 0.0f, 9u * PROJECTION), "prefill conv clear");
    require_ok(ds4_gpu_tensor_fill_f32(pstate, 0.0f, (uint64_t)HEADS * D * D), "prefill state clear");
    require_ok(ds4_gpu_glm53_kda_prefill(
        pout, pconv, pstate, pq, pk, pv, pg, pbeta, poutput_gate,
        model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
        A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
        HEADS, TOKENS, -5.0f, 1e-5f), "KDA prefill");
    float prefill_outputs[TOKENS * PROJECTION];
    require_ok(ds4_gpu_tensor_read(pout, 0, prefill_outputs, sizeof(prefill_outputs)),
               "prefill output read");
    for (uint32_t i = 0; i < TOKENS * PROJECTION; i++)
        require_close("KDA prefill/decode", prefill_outputs[i], decode_outputs[i], 5e-5f);

    ds4_gpu_tensor_free(pstate);
    ds4_gpu_tensor_free(pconv);
    ds4_gpu_tensor_free(pout);
    ds4_gpu_tensor_free(pbeta);
    ds4_gpu_tensor_free(poutput_gate);
    ds4_gpu_tensor_free(pg);
    ds4_gpu_tensor_free(pv);
    ds4_gpu_tensor_free(pk);
    ds4_gpu_tensor_free(pq);
    ds4_gpu_tensor_free(state);
    ds4_gpu_tensor_free(conv);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(output_gate);
    ds4_gpu_tensor_free(beta);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(bf16_out);
    ds4_gpu_tensor_free(bf16_x);
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    puts("GLM-5.3 KDA GPU tests: PASS");
    return 0;
}
