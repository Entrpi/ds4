#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#ifndef DS4_NO_GPU
#include "../ds4_gpu.h"
#include "../ds4_image.h"
#include <math.h>

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static char *test_save_env(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    size_t len = strlen(value);
    char *copy = malloc(len + 1);
    TEST_ASSERT(copy != NULL);
    if (!copy) return NULL;
    memcpy(copy, value, len + 1);
    return copy;
}

static void test_restore_env(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}

static ds4_engine *test_open_engine(bool quality) {
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = test_model_path(),
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = quality,
    };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0);
    return engine;
}

static ds4_engine *test_get_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    *slot = test_open_engine(quality);
    return *slot;
}

static void test_close_engines(void) {
    ds4_engine_close(test_engine_fast);
    ds4_engine_close(test_engine_quality);
    test_engine_fast = NULL;
    test_engine_quality = NULL;
}

static void test_close_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    ds4_engine_close(*slot);
    *slot = NULL;
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float test_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void test_fill_q8_0_weights(uint8_t *weights,
                                   uint32_t in_dim,
                                   uint32_t out_dim) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 17u + k * 23u + (o ^ k) * 3u) % 67u) - 33;
                vals[i] = (float)v / 96.0f;
                float av = fabsf(vals[i]);
                if (av > amax) amax = av;
            }
            const uint16_t scale_bits = test_float_to_f16(amax / 127.0f);
            const float scale = test_f16_to_f32(scale_bits);
            memcpy(row + b * 34u, &scale_bits, sizeof(scale_bits));
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                int q = scale != 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[i] = (int8_t)q;
            }
        }
    }
}

static void test_metal_f16_matvec_fast_nr0_4(void) {
    /*
     * This is the short regression for the long-context repetition failure.
     * Decode uses one-token F16 matvecs for several DS4 projections; the fast
     * nr0=4 variant must be numerically equivalent to the plain kernel.
     */
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 512;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        ds4_gpu_unregister_model_map(weights_raw);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        ds4_gpu_unregister_model_map(weights_raw);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            ref += w * x_host[i];
        }
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    ds4_gpu_unregister_model_map(weights_raw);
    free(weights_raw);
}

static void test_metal_f16_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((o * 11u + i * 13u + (o ^ i) * 5u) % 61u) - 30;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16((float)v / 96.0f);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        ds4_gpu_unregister_model_map(weights_raw);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        ds4_gpu_unregister_model_map(weights_raw);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 7u + i * 17u + (t ^ i) * 3u) % 73u) - 36;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                          in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float ref = 0.0f;
            for (uint32_t i = 0; i < in_dim; i++) {
                ref += test_f16_to_f32(weights[(uint64_t)o * in_dim + i]) *
                       x_host[(uint64_t)t * in_dim + i];
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    ds4_gpu_unregister_model_map(weights_raw);
    free(weights_raw);
}

static void test_metal_q8_0_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint8_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights, in_dim, out_dim);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        ds4_gpu_unregister_model_map(weights_raw);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        ds4_gpu_unregister_model_map(weights_raw);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 19u + i * 7u + (t ^ i)) % 71u) - 35;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(out, weights_raw, weight_alloc, 0,
                                           in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const uint8_t *row = weights + (uint64_t)o * row_bytes;
            float ref = 0.0f;
            for (uint32_t b = 0; b < in_dim / 32u; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = test_f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32; i++) {
                    ref += scale * (float)qs[i] *
                           x_host[(uint64_t)t * in_dim + b * 32u + i];
                }
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    ds4_gpu_unregister_model_map(weights_raw);
    free(weights_raw);
}

static void test_metal_kernel_group(void) {
    test_metal_f16_matvec_fast_nr0_4();
    test_metal_f16_prefill_matmul();
    test_metal_q8_0_prefill_matmul();
}

static void test_metal_short_prefill_ratio4(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    const int tokens[] = {
        ds4_token_user(engine),
        ds4_token_assistant(engine),
        ds4_token_eos(engine),
    };
    for (size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
        TEST_ASSERT(tokens[i] >= 0);
        if (tokens[i] < 0) return;
    }

    for (size_t n = 1; n <= 3; n++) {
        ds4_tokens prompt = {0};
        for (size_t i = 0; i < n; i++) {
            ds4_tokens_push(&prompt, tokens[i]);
        }
        TEST_ASSERT(prompt.len == (int)n);

        ds4_session *session = NULL;
        TEST_ASSERT(ds4_session_create(&session, engine, 2048) == 0);
        if (!session) {
            ds4_tokens_free(&prompt);
            return;
        }

        char err[160] = {0};
        const int rc = ds4_session_sync(session, &prompt, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "ds4-test: short prefill failed for %zu token(s): %s\n",
                    n, err);
        }
        TEST_ASSERT(rc == 0);

        ds4_session_free(session);
        ds4_tokens_free(&prompt);
    }
}

static char *test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *s = malloc((size_t)len + 1);
    if (!s) {
        fclose(fp);
        return NULL;
    }
    size_t nread = fread(s, 1, (size_t)len, fp);
    fclose(fp);
    if (nread != (size_t)len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

typedef struct {
    const char *name;
    int number;
} test_long_fact;

static const test_long_fact test_long_facts[] = {
    {"Bob", 34},
    {"Alice", 52},
    {"Clara", 71},
    {"Diego", 93},
    {"Elena", 16},
    {"Felix", 88},
    {"Greta", 47},
    {"Hugo", 29},
    {"Iris", 64},
    {"Jonas", 12},
    {"Kira", 81},
    {"Leo", 39},
    {"Marta", 76},
    {"Nadia", 23},
    {"Owen", 58},
    {"Priya", 97},
};

static bool test_is_name_boundary(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || !(isalnum(uc) || c == '_');
}

static bool test_parse_assignment_value(const char *p, int *value) {
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '=') return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (!isdigit((unsigned char)*p)) return false;

    int v = 0;
    while (isdigit((unsigned char)*p)) {
        v = v * 10 + (*p - '0');
        p++;
    }
    *value = v;
    return true;
}

static bool test_output_has_fact(const char *text, const test_long_fact *fact) {
    const size_t name_len = strlen(fact->name);
    const char *p = text;
    bool saw_wrong_assignment = false;
    int wrong_value = -1;

    while ((p = strstr(p, fact->name)) != NULL) {
        const bool before_ok = p == text || test_is_name_boundary(p[-1]);
        const bool after_ok = test_is_name_boundary(p[name_len]) ||
                              p[name_len] == ' ' ||
                              p[name_len] == '\t' ||
                              p[name_len] == '=';
        if (before_ok && after_ok) {
            int value = 0;
            if (test_parse_assignment_value(p + name_len, &value)) {
                if (value == fact->number) return true;
                saw_wrong_assignment = true;
                wrong_value = value;
            }
        }
        p += name_len;
    }

    if (saw_wrong_assignment) {
        fprintf(stderr,
                "ds4-test: long-context wrong assignment for %s: got %d expected %d\n",
                fact->name, wrong_value, fact->number);
    } else {
        fprintf(stderr,
                "ds4-test: long-context missing assignment for %s=%d\n",
                fact->name, fact->number);
    }
    return false;
}

static int test_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool test_hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    while (*hex && !isspace((unsigned char)*hex)) {
        int hi = test_hex_digit(hex[0]);
        int lo = test_hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static bool test_token_bytes_equal(ds4_engine *engine, int token,
                                   const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = ds4_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static void test_long_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 8192 == 0) {
        fprintf(stderr, "ds4-test: long-context prefill %d/%d\n", current, total);
    }
}

static void test_long_story_fact_recall(void) {
    const char *prompt_path = getenv("DS4_TEST_LONG_PROMPT");
    if (!prompt_path || !prompt_path[0]) {
        prompt_path = "tests/long_context_story_prompt.txt";
    }
    char *prompt_text = test_read_file(prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        free(prompt_text);
        return;
    }

    ds4_tokens prompt = {0};
    ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    TEST_ASSERT(prompt.len > 30000);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 100000) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        free(prompt_text);
        return;
    }

    char err[160];
    ds4_session_set_progress(session, test_long_prefill_progress, NULL);
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    ds4_session_set_progress(session, NULL, NULL);

    buf out = {0};
    uint64_t rng = 12345;
    int generated = 0;
    bool decode_ok = true;
    for (; generated < 350; generated++) {
        int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);

        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    const char *text = out.ptr ? out.ptr : "";
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(generated > 0);
    for (size_t i = 0; i < sizeof(test_long_facts) / sizeof(test_long_facts[0]); i++) {
        TEST_ASSERT(test_output_has_fact(text, &test_long_facts[i]));
    }

    buf_free(&out);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    free(prompt_text);
}

#define TEST_VEC_MAX_STEPS 16
#define TEST_VEC_MAX_TOP 32
#define TEST_VEC_MAX_TOKEN_BYTES 128

typedef struct {
    unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} test_vec_top;

typedef struct {
    unsigned char selected[TEST_VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    test_vec_top top[TEST_VEC_MAX_TOP];
} test_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    test_vec_step steps[TEST_VEC_MAX_STEPS];
} test_vec_case;

static char *test_trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static bool test_read_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            TEST_ASSERT(vc->nsteps > 0 && vc->nsteps <= TEST_VEC_MAX_STEPS);
            return true;
        }
        TEST_ASSERT(!"unexpected line before vector case");
    }
    return false;
}

static bool test_fill_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;

        if (!strncmp(p, "step ", 5)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                TEST_ASSERT(!"bad vector step line");
                return false;
            }
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(ntop >= 0 && ntop <= TEST_VEC_MAX_TOP);
            vc->steps[step_index].ntop = ntop;
            TEST_ASSERT(test_hex_to_bytes(hex,
                                          vc->steps[step_index].selected,
                                          TEST_VEC_MAX_TOKEN_BYTES,
                                          &vc->steps[step_index].selected_len));
            top_index = 0;
            continue;
        }

        if (!strncmp(p, "top ", 4)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index < vc->steps[step_index].ntop);
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) {
                TEST_ASSERT(!"bad vector top line");
                return false;
            }
            test_vec_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            TEST_ASSERT(test_hex_to_bytes(hex, top->bytes,
                                          TEST_VEC_MAX_TOKEN_BYTES, &top->len));
            continue;
        }

        TEST_ASSERT(!"unexpected vector line");
        return false;
    }

    TEST_ASSERT(!"unterminated vector case");
    return false;
}

static bool test_logprob_vector_step_disabled(const test_vec_case *vc, int step_index) {
    /*
     * Excluded steps are skipped but teacher-forced to the official token
     * so the remaining steps still check the API trajectory.  No active
     * step exclusions against the 2026-08-03 fixtures: the former
     * short_code_completion step-1 exclusion (inc-14a distribution shift,
     * last re-adjudicated 2026-07-31) died with the pre-0731 fixture --
     * that case now records a single step.  Future adjudications add
     * (vc->id, step) pairs here; DS4_TEST_NO_STEP_EXCLUSIONS=1 runs the
     * unexcluded battery for re-adjudication legs.
     */
    (void)vc; (void)step_index;
    if (getenv("DS4_TEST_NO_STEP_EXCLUSIONS")) return false;
    return false;
}

static void test_logprob_vector_case(ds4_engine *engine, const test_vec_case *vc) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);

    ds4_token_score scores[20];
    for (int i = 0; i < vc->nsteps; i++) {
        const test_vec_step *step = &vc->steps[i];
        int nscore = ds4_session_top_logprobs(session, scores, 20);
        int token = ds4_session_argmax(session);
        if (test_logprob_vector_step_disabled(vc, i)) {
            int forced = -1;
            for (int j = 0; j < nscore; j++) {
                if (scores[j].id >= 0 &&
                    test_token_bytes_equal(engine, scores[j].id,
                                           step->selected, step->selected_len)) {
                    forced = scores[j].id;
                    break;
                }
            }
            fprintf(stderr,
                    "ds4-test: vector %s step %d excluded, teacher-forcing "
                    "official token id=%d\n", vc->id, i, forced);
            TEST_ASSERT(forced >= 0);
            if (forced >= 0) token = forced;
            if (i + 1 < vc->nsteps) {
                TEST_ASSERT(ds4_session_eval(session, token, err, sizeof(err)) == 0);
            }
            continue;
        }
        if (!test_token_bytes_equal(engine, token, step->selected, step->selected_len)) {
            fprintf(stderr, "ds4-test: vector %s step %d selected token mismatch\n",
                    vc->id, i);
            /* Adjudication data: the local top-4 with logprobs vs the
             * fixture's expected leader -- a near-zero top-2 gap marks a
             * knife-edge fixture step (exclusion candidate), a wide gap a
             * real distribution shift. */
            fprintf(stderr, "ds4-test:   expected '%.*s', local argmax id=%d\n",
                    (int)step->selected_len, step->selected, token);
            for (int j = 0; j < nscore && j < 4; j++)
                fprintf(stderr, "ds4-test:   local top%d id=%d logprob=%.5f\n",
                        j, scores[j].id, scores[j].logprob);
            TEST_ASSERT(false);
        }

        for (int t = 0; t < step->ntop; t++) {
            bool found = false;
            float local_lp = 0.0f;
            for (int j = 0; j < nscore; j++) {
                if (scores[j].id < 0) continue;
                if (test_token_bytes_equal(engine, scores[j].id,
                                           step->top[t].bytes,
                                           step->top[t].len)) {
                    found = true;
                    local_lp = scores[j].logprob;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "ds4-test: vector %s step %d official top token missing locally\n",
                        vc->id, i);
                TEST_ASSERT(false);
            } else if (fabsf(local_lp - step->top[t].logprob) > 4.0f) {
                fprintf(stderr,
                        "ds4-test: vector %s step %d logprob delta too high: local=%g official=%g\n",
                        vc->id, i, local_lp, step->top[t].logprob);
                TEST_ASSERT(false);
            }
        }

        if (i + 1 < vc->nsteps) {
            TEST_ASSERT(ds4_session_eval(session, token, err, sizeof(err)) == 0);
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static bool test_logprob_vector_case_disabled(const test_vec_case *vc) {
    /*
     * No active case exclusions against the 2026-08-03 fixtures (official
     * DeepSeek provider via OpenRouter, 0731 model -- provenance in
     * tests/test-vectors/manifest.json).  The former long_memory_archive
     * exclusion (Hadamard+FP4 indexer divergence vs the pre-0731 API) was
     * re-adjudicated against the fresh fixture and passes.
     */
    (void)vc;
    return false;
}

static void test_official_logprob_vectors(void) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    setenv("DS4_METAL_PREFILL_CHUNK", "2048", 1);
    if (getenv("DS4_TEST_LOGPROB_AUTO_METAL") == NULL) {
        setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    } else {
        unsetenv("DS4_METAL_DISABLE_METAL4");
    }
    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_vec_case vc;
    while (test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (test_logprob_vector_case_disabled(&vc)) {
            fprintf(stderr, "ds4-test: vector %s skipped (API/official graph mismatch)\n",
                    vc.id);
            continue;
        }
        fprintf(stderr, "ds4-test: vector %s\n", vc.id);
        test_logprob_vector_case(engine, &vc);
    }
    ds4_engine_close(engine);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

static void test_logits_topk(const float *logits, int n, int *out, int k);
static bool test_topk_contains(const int *top, int k, int id);

#define TEST_LOCAL_GOLDEN_MAX_TOP 128

typedef struct {
    int id;
    float logit;
} test_local_golden_top;

typedef struct {
    char id[96];
    char mode[16];
    char prompt_path[512];
    int ctx;
    int frontier;
    int ntop;
    test_local_golden_top top[TEST_LOCAL_GOLDEN_MAX_TOP];
} test_local_golden_case;

static bool test_read_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    memset(tc, 0, sizeof(*tc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %15s %d %d %511s %d",
                   tc->id, tc->mode, &tc->ctx, &tc->frontier,
                   tc->prompt_path, &tc->ntop) == 6) {
            TEST_ASSERT(tc->ctx > tc->frontier);
            TEST_ASSERT(tc->frontier > 0);
            TEST_ASSERT(tc->ntop > 0 && tc->ntop <= TEST_LOCAL_GOLDEN_MAX_TOP);
            return true;
        }
        TEST_ASSERT(!"unexpected line before local golden case");
        return false;
    }
    return false;
}

static bool test_fill_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    int seen = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) {
            TEST_ASSERT(seen == tc->ntop);
            return seen == tc->ntop;
        }
        int rank = -1;
        int id = -1;
        float logit = 0.0f;
        if (sscanf(p, "top %d %d %f", &rank, &id, &logit) != 3) {
            TEST_ASSERT(!"bad local golden top line");
            return false;
        }
        TEST_ASSERT(rank == seen);
        TEST_ASSERT(seen < tc->ntop);
        if (seen >= tc->ntop) return false;
        tc->top[seen].id = id;
        tc->top[seen].logit = logit;
        seen++;
    }
    TEST_ASSERT(!"unterminated local golden case");
    return false;
}

static int test_local_golden_overlap(const test_local_golden_case *tc,
                                     const int *cand_top,
                                     int n) {
    int overlap = 0;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        if (test_topk_contains(cand_top, n, tc->top[i].id)) overlap++;
    }
    return overlap;
}

static float test_local_golden_max_abs(const test_local_golden_case *tc,
                                       const float *cand_logits,
                                       int n) {
    float max_abs = 0.0f;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        const int id = tc->top[i].id;
        if (id < 0) continue;
        const float abs_delta = fabsf(cand_logits[id] - tc->top[i].logit);
        if (abs_delta > max_abs) max_abs = abs_delta;
    }
    return max_abs;
}

static void test_local_golden_case_run(ds4_engine *engine,
                                       const test_local_golden_case *tc) {
    char *prompt_text = test_read_file(tc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    if (!strcmp(tc->mode, "text")) {
        ds4_tokenize_text(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "rendered")) {
        ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "chat")) {
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    } else {
        TEST_ASSERT(!"unknown local golden prompt mode");
    }
    free(prompt_text);
    TEST_ASSERT(prompt.len >= tc->frontier);
    if (prompt.len < tc->frontier) {
        ds4_tokens_free(&prompt);
        return;
    }

    ds4_tokens prefix = {
        .v = prompt.v,
        .len = tc->frontier,
        .cap = tc->frontier,
    };

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);

    const int vocab = ds4_engine_vocab_size(engine);
    float *cand_logits = malloc((size_t)vocab * sizeof(cand_logits[0]));
    TEST_ASSERT(cand_logits != NULL);
    if (cand_logits &&
        ds4_session_copy_logits(session, cand_logits, vocab) == vocab) {
        int cand_top[TEST_LOCAL_GOLDEN_MAX_TOP];
        const int ntop = tc->ntop < TEST_LOCAL_GOLDEN_MAX_TOP ?
                         tc->ntop : TEST_LOCAL_GOLDEN_MAX_TOP;
        test_logits_topk(cand_logits, vocab, cand_top, ntop);

        const char *record_path = getenv("DS4_TEST_LOCAL_GOLDEN_RECORD");
        if (record_path && record_path[0]) {
            /*
             * Re-base mode: emit this run's top list in local-golden.vec
             * format instead of checking the stale fixture.  Only valid
             * after an adjudicated numerics re-base with the quality gates
             * green; verify by re-running the check against the recorded
             * file before installing it.
             */
            FILE *out = fopen(record_path, "a");
            TEST_ASSERT(out != NULL);
            if (out) {
                fprintf(out, "case %s %s %d %d %s %d\n", tc->id, tc->mode,
                        tc->ctx, tc->frontier, tc->prompt_path, ntop);
                for (int r = 0; r < ntop; r++) {
                    fprintf(out, "top %d %d %.9g\n",
                            r, cand_top[r], cand_logits[cand_top[r]]);
                }
                fprintf(out, "end\n");
                fclose(out);
            }
            fprintf(stderr, "ds4-test: local golden %s recorded to %s\n",
                    tc->id, record_path);
            free(cand_logits);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            return;
        }

        const int top5_overlap = test_local_golden_overlap(tc, cand_top, 5);
        const int top20_overlap = test_local_golden_overlap(tc, cand_top, 20);
        const int top64_overlap = test_local_golden_overlap(tc, cand_top, 64);
        const float top20_max_abs =
            test_local_golden_max_abs(tc, cand_logits, 20);

        fprintf(stderr,
                "ds4-test: local golden %s top1 ref=%d cand=%d "
                "top5_overlap=%d/5 top20_overlap=%d/20 top64_overlap=%d/64 "
                "top20_max_abs=%g\n",
                tc->id, tc->top[0].id, cand_top[0],
                top5_overlap, top20_overlap, top64_overlap, top20_max_abs);

        /*
         * This is intentionally tolerant: it is meant to catch substantial
         * backend drift (wrong tiling, skipped work, bad dispatch), not tiny
         * floating-point differences from otherwise sane kernel changes.
         */
        TEST_ASSERT(cand_top[0] == tc->top[0].id);
        TEST_ASSERT(top5_overlap >= 4);
        TEST_ASSERT(top20_overlap >= 15);
        TEST_ASSERT(top64_overlap >= 40);
        TEST_ASSERT(top20_max_abs <= 8.0f);
    } else {
        TEST_ASSERT(false);
    }

    free(cand_logits);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static void test_local_golden_vectors(void) {
    const char *path = getenv("DS4_TEST_LOCAL_GOLDEN_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/local-golden.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    char *saved_moe_tile_max = test_save_env("DS4_METAL_MOE_TILE_MAX");
    setenv("DS4_METAL_PREFILL_CHUNK", "4096", 1);
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    unsetenv("DS4_METAL_MOE_TILE_MAX");

    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_local_golden_case tc;
    while (test_read_local_golden_case(fp, &tc)) {
        if (!test_fill_local_golden_case(fp, &tc)) break;
        test_local_golden_case_run(engine, &tc);
    }

    ds4_engine_close(engine);
    test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

#define TEST_MPP_EQ_MAX_CASES 8
#define TEST_MPP_EQ_TOPK 20
#define TEST_MPP_EQ_TOP5 5
#define TEST_MPP_EQ_DELTAS 5

typedef struct {
    char id[96];
    int ctx;
    int vocab_size;
    int gen_steps;
    ds4_tokens prompt;
    float *ref_logits;
    int ref_gen[TEST_VEC_MAX_STEPS];
    int ref_gen_len;
} test_mpp_eq_case;

typedef struct {
    int ref_top1;
    int cand_top1;
    int overlap;
    int top5_overlap;
    int max_rank_delta;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
    bool same_top1;
    bool pass;
} test_mpp_eq_result;

typedef struct {
    const char *label;
    int cases;
    int capture_failures;
    int logits_failures;
    int greedy_failures;
    int top1_mismatches;
    int min_overlap;
    int min_top5_overlap;
    int worst_rank_delta;
    float worst_rms;
    float worst_max_abs;
    float worst_top20_max_abs;
} test_mpp_eq_summary;

static void test_mpp_eq_case_free(test_mpp_eq_case *tc) {
    if (!tc) return;
    ds4_tokens_free(&tc->prompt);
    free(tc->ref_logits);
    memset(tc, 0, sizeof(*tc));
}

static void test_logits_topk(const float *logits, int n, int *out, int k) {
    for (int i = 0; i < k; i++) out[i] = -1;
    for (int id = 0; id < n; id++) {
        const float v = logits[id];
        if (!isfinite(v)) continue;
        for (int j = 0; j < k; j++) {
            if (out[j] < 0 || v > logits[out[j]]) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j] = id;
                break;
            }
        }
    }
}

static bool test_topk_contains(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return true;
    }
    return false;
}

static int test_topk_rank(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return i;
    }
    return -1;
}

static void test_note_delta(int *ids, float *ref_vals, float *cand_vals,
                            float *abs_vals, int id, float ref, float cand) {
    const float abs_delta = fabsf(cand - ref);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        if (ids[i] < 0 || abs_delta > abs_vals[i]) {
            for (int j = TEST_MPP_EQ_DELTAS - 1; j > i; j--) {
                ids[j] = ids[j - 1];
                ref_vals[j] = ref_vals[j - 1];
                cand_vals[j] = cand_vals[j - 1];
                abs_vals[j] = abs_vals[j - 1];
            }
            ids[i] = id;
            ref_vals[i] = ref;
            cand_vals[i] = cand;
            abs_vals[i] = abs_delta;
            return;
        }
    }
}

static float test_top_union_max_abs(const float *ref, const float *cand,
                                    const int *ref_top, const int *cand_top, int k) {
    float max_abs = 0.0f;
    for (int i = 0; i < k; i++) {
        if (ref_top[i] >= 0) {
            const float d = fabsf(cand[ref_top[i]] - ref[ref_top[i]]);
            if (d > max_abs) max_abs = d;
        }
        if (cand_top[i] >= 0 && !test_topk_contains(ref_top, k, cand_top[i])) {
            const float d = fabsf(cand[cand_top[i]] - ref[cand_top[i]]);
            if (d > max_abs) max_abs = d;
        }
    }
    return max_abs;
}

/*
 * Metal4/TensorOps equivalence is a smoke test, not a demand for bitwise local
 * logits.  Tensor kernels change precision and reduction order, so the useful
 * invariant here is: no NaNs, same first greedy token, and same short greedy
 * continuation.  Larger logit drift is still printed so it can be compared with
 * official API-vector and long-context recall gates.
 */
static test_mpp_eq_result test_compare_mpp_logits(const test_mpp_eq_case *tc,
                                                  const float *cand_logits,
                                                  bool assert_thresholds) {
    int ref_top[TEST_MPP_EQ_TOPK];
    int cand_top[TEST_MPP_EQ_TOPK];
    test_logits_topk(tc->ref_logits, tc->vocab_size, ref_top, TEST_MPP_EQ_TOPK);
    test_logits_topk(cand_logits, tc->vocab_size, cand_top, TEST_MPP_EQ_TOPK);

    int overlap = 0;
    int top5_overlap = 0;
    int max_rank_delta = 0;
    for (int i = 0; i < TEST_MPP_EQ_TOPK; i++) {
        const int cand_rank = test_topk_rank(cand_top, TEST_MPP_EQ_TOPK, ref_top[i]);
        if (ref_top[i] >= 0 && cand_rank >= 0) {
            overlap++;
            const int rank_delta = abs(cand_rank - i);
            if (rank_delta > max_rank_delta) max_rank_delta = rank_delta;
        }
        if (i < TEST_MPP_EQ_TOP5 &&
            ref_top[i] >= 0 &&
            test_topk_contains(cand_top, TEST_MPP_EQ_TOP5, ref_top[i])) {
            top5_overlap++;
        }
    }

    double sumsq = 0.0;
    float max_abs = 0.0f;
    int nonfinite = 0;
    int delta_ids[TEST_MPP_EQ_DELTAS];
    float delta_ref[TEST_MPP_EQ_DELTAS];
    float delta_cand[TEST_MPP_EQ_DELTAS];
    float delta_abs[TEST_MPP_EQ_DELTAS];
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        delta_ids[i] = -1;
        delta_ref[i] = 0.0f;
        delta_cand[i] = 0.0f;
        delta_abs[i] = 0.0f;
    }

    for (int i = 0; i < tc->vocab_size; i++) {
        if (!isfinite(tc->ref_logits[i]) || !isfinite(cand_logits[i])) {
            nonfinite++;
            continue;
        }
        const float delta = cand_logits[i] - tc->ref_logits[i];
        const float abs_delta = fabsf(delta);
        if (abs_delta > max_abs) max_abs = abs_delta;
        sumsq += (double)delta * (double)delta;
        test_note_delta(delta_ids, delta_ref, delta_cand, delta_abs,
                        (int)i, tc->ref_logits[i], cand_logits[i]);
    }

    const float rms = (float)sqrt(sumsq / (double)tc->vocab_size);
    const float top_abs = test_top_union_max_abs(tc->ref_logits, cand_logits,
                                                 ref_top, cand_top, TEST_MPP_EQ_TOPK);
    const bool same_top1 = ref_top[0] >= 0 && ref_top[0] == cand_top[0];
    test_mpp_eq_result result = {
        .ref_top1 = ref_top[0],
        .cand_top1 = cand_top[0],
        .overlap = overlap,
        .top5_overlap = top5_overlap,
        .max_rank_delta = max_rank_delta,
        .nonfinite = nonfinite,
        .rms = rms,
        .max_abs = max_abs,
        .top20_max_abs = top_abs,
        .same_top1 = same_top1,
        .pass = nonfinite == 0 && same_top1,
    };

    fprintf(stderr,
            "ds4-test: Tensor equivalence %s top1 ref=%d cand=%d top5_overlap=%d/%d overlap=%d/%d max_rank_delta=%d rms=%g max_abs=%g top20_max_abs=%g\n",
            tc->id, ref_top[0], cand_top[0],
            top5_overlap, TEST_MPP_EQ_TOP5,
            overlap, TEST_MPP_EQ_TOPK,
            max_rank_delta, rms, max_abs, top_abs);
    fprintf(stderr, "ds4-test: Tensor equivalence %s largest deltas:", tc->id);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS && delta_ids[i] >= 0; i++) {
        fprintf(stderr, " id=%d ref=%g cand=%g abs=%g",
                delta_ids[i], delta_ref[i], delta_cand[i], delta_abs[i]);
    }
    fputc('\n', stderr);

    if (assert_thresholds) {
        TEST_ASSERT(nonfinite == 0);
        TEST_ASSERT(same_top1);
    }
    return result;
}

static bool test_mpp_capture(ds4_engine *engine, const test_mpp_eq_case *tc,
                             float *logits, int *gen, int *gen_len) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    int n = 0;
    while (ok && n < tc->gen_steps) {
        const int token = ds4_session_argmax(session);
        gen[n++] = token;
        if (n < tc->gen_steps && ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            ok = false;
            TEST_ASSERT(false);
        }
    }
    *gen_len = n;

    ds4_session_free(session);
    return ok;
}

static bool test_mpp_eq_case_selected(const char *id) {
    const char *filter = getenv("DS4_TEST_MPP_EQ_CASE");
    if (!filter || !filter[0]) return true;

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", filter);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        tok = test_trim_line(tok);
        if (tok[0] && strstr(id, tok)) return true;
    }
    return false;
}

static int test_load_mpp_cases(ds4_engine *engine, test_mpp_eq_case *cases, int cap) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return 0;

    int ncase = 0;
    test_vec_case vc;
    while (ncase < cap && test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (!test_mpp_eq_case_selected(vc.id)) continue;
        char *prompt_text = test_read_file(vc.prompt_path);
        TEST_ASSERT(prompt_text != NULL);
        if (!prompt_text) continue;

        test_mpp_eq_case *tc = &cases[ncase++];
        snprintf(tc->id, sizeof(tc->id), "%s", vc.id);
        tc->ctx = vc.ctx;
        tc->vocab_size = ds4_engine_vocab_size(engine);
        tc->gen_steps = vc.nsteps < TEST_VEC_MAX_STEPS ? vc.nsteps : TEST_VEC_MAX_STEPS;
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &tc->prompt);
        free(prompt_text);
        TEST_ASSERT(tc->prompt.len > 0);
    }
    fclose(fp);
    return ncase;
}

static void test_mpp_summary_init(test_mpp_eq_summary *summary, const char *label) {
    memset(summary, 0, sizeof(*summary));
    summary->label = label;
    summary->min_overlap = TEST_MPP_EQ_TOPK;
    summary->min_top5_overlap = TEST_MPP_EQ_TOP5;
}

static void test_mpp_summary_note_logits(test_mpp_eq_summary *summary,
                                         const test_mpp_eq_result *result) {
    if (!result->pass) summary->logits_failures++;
    if (!result->same_top1) summary->top1_mismatches++;
    if (result->overlap < summary->min_overlap) summary->min_overlap = result->overlap;
    if (result->top5_overlap < summary->min_top5_overlap) {
        summary->min_top5_overlap = result->top5_overlap;
    }
    if (result->max_rank_delta > summary->worst_rank_delta) {
        summary->worst_rank_delta = result->max_rank_delta;
    }
    if (result->rms > summary->worst_rms) summary->worst_rms = result->rms;
    if (result->max_abs > summary->worst_max_abs) summary->worst_max_abs = result->max_abs;
    if (result->top20_max_abs > summary->worst_top20_max_abs) {
        summary->worst_top20_max_abs = result->top20_max_abs;
    }
}

static void test_mpp_summary_print(const test_mpp_eq_summary *summary) {
    fprintf(stderr,
            "ds4-test: Tensor summary route=%s cases=%d capture_fail=%d logits_fail=%d greedy_fail=%d top1_mismatch=%d min_top5_overlap=%d/%d min_overlap=%d/%d worst_rank_delta=%d worst_rms=%g worst_max_abs=%g worst_top20_max_abs=%g\n",
            summary->label,
            summary->cases,
            summary->capture_failures,
            summary->logits_failures,
            summary->greedy_failures,
            summary->top1_mismatches,
            summary->min_top5_overlap,
            TEST_MPP_EQ_TOP5,
            summary->min_overlap,
            TEST_MPP_EQ_TOPK,
            summary->worst_rank_delta,
            summary->worst_rms,
            summary->worst_max_abs,
            summary->worst_top20_max_abs);
}

static void test_run_mpp_candidate(const char *label,
                                   test_mpp_eq_case *cases,
                                   int ncase) {
    fprintf(stderr, "ds4-test: Tensor equivalence candidate route=%s\n", label);
    test_mpp_eq_summary summary;
    test_mpp_summary_init(&summary, label);
    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        const int vocab_size = ncase > 0 ? cases[0].vocab_size : 0;
        float *cand_logits = malloc((size_t)vocab_size * sizeof(cand_logits[0]));
        TEST_ASSERT(cand_logits != NULL);
        if (cand_logits) {
            for (int i = 0; i < ncase; i++) {
                test_mpp_eq_case *tc = &cases[i];
                if (!tc->ref_logits) continue;
                int cand_gen[TEST_VEC_MAX_STEPS] = {0};
                int cand_gen_len = 0;
                if (!test_mpp_capture(cand_engine, tc, cand_logits, cand_gen, &cand_gen_len)) {
                    summary.capture_failures++;
                    continue;
                }
                summary.cases++;
                test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_logits, true);
                test_mpp_summary_note_logits(&summary, &result);
                TEST_ASSERT(cand_gen_len == tc->ref_gen_len);
                if (cand_gen_len != tc->ref_gen_len) summary.greedy_failures++;
                for (int j = 0; j < tc->ref_gen_len && j < cand_gen_len; j++) {
                    if (cand_gen[j] != tc->ref_gen[j]) {
                        fprintf(stderr,
                                "ds4-test: Tensor equivalence %s greedy token mismatch step=%d ref=%d cand=%d\n",
                                tc->id, j, tc->ref_gen[j], cand_gen[j]);
                        summary.greedy_failures++;
                    }
                    TEST_ASSERT(cand_gen[j] == tc->ref_gen[j]);
                }
            }
            free(cand_logits);
        }
        ds4_engine_close(cand_engine);
    }
    test_mpp_summary_print(&summary);
}

static void test_metal_mpp_equivalence(void) {
    test_close_engines();

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);

    test_run_mpp_candidate("auto", cases, ncase);

    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static void test_tool_call_quality_one(bool quality) {
    ds4_engine *engine = test_get_engine(quality);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_tool_call_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(ds4_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    buf text = {0};
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    for (int i = 0; i < r.max_tokens; i++) {
        int token = ds4_session_sample(session, r.temperature, r.top_k,
                                       r.top_p, r.min_p, &rng);
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr ? text.ptr : "", &saw_tool_start, &saw_tool_end, NULL);
        if (saw_tool_end) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr ? text.ptr : "",
                                             false, &content, &reasoning, &calls);
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    ds4_session_free(session);
    request_free(&r);
}

static void test_tool_call_quality(void) {
    fprintf(stderr, "ds4-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "ds4-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}

#endif


/* ===================== Track B inc 3: image block token stream + checkpoint lineage =====================
 * Box-free and engine-free: the public row builder takes the four BF16 sentinel
 * rows as inputs, the span validator takes a token stream, and the A7 identity
 * rules are pure functions over a ds4_vision_lineage.  What needs a model (the
 * chat-template append, the CUDA prefill) is gated on .33. */
static uint16_t test_f32_to_bf16(float f) { uint32_t b; memcpy(&b, &f, 4); return (uint16_t)(b >> 16); }

static void test_vision_stub_embedding(ds4_vision_embedding *emb, uint32_t gh, uint32_t gw, uint8_t seed) {
    memset(emb, 0, sizeof(*emb));
    emb->layout = DS4_VISION_LAYOUT_DEEPSEEK4_NATURAL;
    emb->grid_height = gh; emb->grid_width = gw; emb->token_count = gh * gw;
    emb->data = malloc((size_t)emb->token_count * 4096u * sizeof(float));
    TEST_ASSERT(emb->data != NULL);
    for (uint32_t r = 0; r < emb->token_count; r++)
        for (uint32_t d = 0; d < 4096u; d++) emb->data[(size_t)r * 4096u + d] = (float)(r * 1000u + d);
    memset(emb->fingerprint, seed, sizeof(emb->fingerprint));
    memset(emb->identity, (uint8_t)(seed + 1), sizeof(emb->identity));
}

static void test_vision_prompt_group(void) {
    uint16_t *sidecar = malloc((size_t)4 * 4096 * sizeof(uint16_t));
    TEST_ASSERT(sidecar != NULL);
    for (int t = 0; t < 4; t++)
        for (int d = 0; d < 4096; d++) sidecar[t * 4096 + d] = test_f32_to_bf16((float)(100 * (t + 1) + (d % 7)));
    const uint16_t *const sentinel[5] = { sidecar, sidecar + 4096, NULL, sidecar + 2 * 4096, sidecar + 3 * 4096 };
    char err[256];

    /* (a) one block at prefixes 0..3: ids follow the layout, IMAGE rows follow perm[],
     *     sentinel rows carry the sidecar vectors, the span parses as one block. */
    for (int prefix = 0; prefix < 4; prefix++) {
        ds4_tokens toks = {0};
        for (int i = 0; i < prefix; i++) ds4_tokens_push(&toks, 5 + i);
        ds4_deepseek4_image_layout want = {0};
        TEST_ASSERT(ds4_deepseek4_image_layout_build(&want, 3, 4, (uint32_t)prefix, err, sizeof(err)));
        ds4_vision_embedding emb; test_vision_stub_embedding(&emb, 3, 4, 0x11);
        ds4_vision_span span = {0};
        TEST_ASSERT(ds4_vision_prompt_append_rows(&toks, &span, &emb, sentinel, err, sizeof(err)) == 1);
        TEST_ASSERT(emb.data == NULL && span.embedding.data != NULL);   /* consumed into the span */
        TEST_ASSERT(span.token_start == (uint32_t)prefix);
        TEST_ASSERT(span.embedding.token_count == want.token_count);
        TEST_ASSERT(toks.len == prefix + (int)want.token_count);
        {   /* exactly one START and one END; alignment PAD rows only at the edges;
             * IMAGE and NEWLINE rows strictly between them */
            int fs = -1, le = -1, n_start = 0, n_end = 0;
            for (uint32_t r = 0; r < want.token_count; r++) {
                if (want.types[r] == DS4_DEEPSEEK4_IMAGE_START) { n_start++; if (fs < 0) fs = (int)r; }
                if (want.types[r] == DS4_DEEPSEEK4_IMAGE_END) { n_end++; le = (int)r; }
            }
            TEST_ASSERT(n_start == 1 && n_end == 1 && fs >= 0 && le > fs);
            for (uint32_t r = 0; r < want.token_count; r++) {
                const uint8_t ty = want.types[r];
                if ((int)r < fs || (int)r > le) TEST_ASSERT(ty == DS4_DEEPSEEK4_IMAGE_PAD);
                else if ((int)r > fs && (int)r < le) TEST_ASSERT(ty == DS4_DEEPSEEK4_IMAGE || ty == DS4_DEEPSEEK4_IMAGE_NEWLINE || ty == DS4_DEEPSEEK4_IMAGE_PAD);
            }
            TEST_ASSERT(fs == (int)(3u - (uint32_t)prefix % 4u));   /* compress_pad leading PADs put START on the last slot of its 4-row group */
            TEST_ASSERT(((uint32_t)prefix + (uint32_t)fs) % 4u == 3u);
        }
        uint32_t image_pos = 0;
        for (uint32_t r = 0; r < want.token_count; r++) {
            TEST_ASSERT(toks.v[prefix + (int)r] == (int)(DS4_VISION_ID_BASE + want.types[r]));
            const float *row = span.embedding.data + (size_t)r * 4096u;
            if (want.types[r] == DS4_DEEPSEEK4_IMAGE) {
                const uint32_t src = want.perm[image_pos++];
                TEST_ASSERT(row[0] == (float)(src * 1000u) && row[4095] == (float)(src * 1000u + 4095u));
            } else {
                const int slot = want.types[r] == DS4_DEEPSEEK4_IMAGE_START ? 0 :
                                 want.types[r] == DS4_DEEPSEEK4_IMAGE_PAD ? 1 :
                                 want.types[r] == DS4_DEEPSEEK4_IMAGE_NEWLINE ? 2 : 3;
                TEST_ASSERT(row[0] == (float)(100 * (slot + 1)) && row[6] == (float)(100 * (slot + 1) + 6));
            }
        }
        TEST_ASSERT(image_pos == 12u);
        uint32_t cursor = 0, bs = 0, is = 0, ie = 0;
        TEST_ASSERT(ds4_deepseek4_next_image_span(toks.v + prefix, want.token_count, DS4_VISION_ID_BASE,
                                                  &cursor, &bs, &is, &ie) == 1);
        TEST_ASSERT(bs == 0 && cursor == want.token_count && ie + 1u == cursor);
        TEST_ASSERT(span.embedding.identity[0] == 0x12 && span.embedding.fingerprint[0] == 0x11);
        TEST_ASSERT(ds4_vision_spans_validate(&toks, &span, 1, err, sizeof(err)) == 1);
        ds4_deepseek4_image_layout_free(&want);
        ds4_vision_embedding_free(&span.embedding);
        ds4_tokens_free(&toks);
    }

    /* (b) two blocks back to back validate; overlap and a cut block are refused by name */
    {
        ds4_tokens toks = {0};
        ds4_tokens_push(&toks, 7);
        ds4_vision_embedding emb[2]; ds4_vision_span span[2] = {{0}, {0}};
        test_vision_stub_embedding(&emb[0], 2, 2, 0x21);
        test_vision_stub_embedding(&emb[1], 3, 3, 0x31);
        TEST_ASSERT(ds4_vision_prompt_append_rows(&toks, &span[0], &emb[0], sentinel, err, sizeof(err)) == 1);
        ds4_tokens_push(&toks, 9);
        TEST_ASSERT(ds4_vision_prompt_append_rows(&toks, &span[1], &emb[1], sentinel, err, sizeof(err)) == 1);
        TEST_ASSERT(span[1].token_start == span[0].token_start + span[0].embedding.token_count + 1u);
        TEST_ASSERT(ds4_vision_spans_validate(&toks, span, 2, err, sizeof(err)) == 1);
        ds4_vision_span bad[2] = { span[0], span[1] };
        bad[1].token_start = span[0].token_start;
        TEST_ASSERT(ds4_vision_spans_validate(&toks, bad, 2, err, sizeof(err)) == 0);
        TEST_ASSERT(strstr(err, "overlapping") != NULL);
        ds4_vision_span cut = span[0]; cut.embedding.token_count -= 1u;
        TEST_ASSERT(ds4_vision_spans_validate(&toks, &cut, 1, err, sizeof(err)) == 0);
        TEST_ASSERT(strstr(err, "complete DeepSeek image block") != NULL);
        ds4_vision_span shifted = span[0]; shifted.token_start += 1u;   /* starts on a PAD/IMAGE row */
        TEST_ASSERT(ds4_vision_spans_validate(&toks, &shifted, 1, err, sizeof(err)) == 0);
        ds4_vision_embedding_free(&span[0].embedding); ds4_vision_embedding_free(&span[1].embedding);
        ds4_tokens_free(&toks);
    }

    /* (c) A7 lineage rules: same span+identity matches, a different image at the
     *     same span does not, an extra image inside the checkpoint does not, the
     *     LCP walk is capped at the block start, a rewind into a block demands
     *     invalidation, a rewind before it forgets it. */
    {
        ds4_vision_lineage l = {0};
        ds4_vision_span same = {0}; same.token_start = 10; same.embedding.token_count = 20; same.embedding.data = (float *)1;
        memset(same.embedding.identity, 0xAA, 32);
        TEST_ASSERT(ds4_vision_lineage_store(&l, &same, 1) == 1 && l.n == 1 && l.v[0].token_start == 10);
        TEST_ASSERT(ds4_vision_lineage_prefix_matches(&l, true, 40, &same, 1));
        TEST_ASSERT(ds4_vision_lineage_prefix_matches(&l, false, 0, NULL, 0));   /* invalid checkpoint matches anything */
        ds4_vision_span other = same; memset(other.embedding.identity, 0xBB, 32);
        TEST_ASSERT(!ds4_vision_lineage_prefix_matches(&l, true, 40, &other, 1));
        ds4_vision_span moved = same; moved.token_start = 11;
        TEST_ASSERT(!ds4_vision_lineage_prefix_matches(&l, true, 40, &moved, 1));
        TEST_ASSERT(!ds4_vision_lineage_prefix_matches(&l, true, 40, NULL, 0));     /* text prompt over an image checkpoint */
        ds4_vision_span two[2] = { same, same }; two[1].token_start = 35;           /* second image INSIDE the checkpoint */
        TEST_ASSERT(!ds4_vision_lineage_prefix_matches(&l, true, 40, two, 2));
        two[1].token_start = 40;                                                    /* appended after it: fine */
        TEST_ASSERT(ds4_vision_lineage_prefix_matches(&l, true, 40, two, 2));
        TEST_ASSERT(ds4_vision_lineage_prefix_cap(&l, 40) == 10);
        TEST_ASSERT(ds4_vision_lineage_prefix_cap(&l, 10) == 10);
        TEST_ASSERT(ds4_vision_lineage_prefix_cap(&l, 3) == 3);
        TEST_ASSERT(ds4_vision_lineage_rewind(&l, 35) == true && l.n == 1);    /* after the block: keeps it */
        TEST_ASSERT(ds4_vision_lineage_rewind(&l, 20) == false);               /* inside: caller invalidates */
        TEST_ASSERT(ds4_vision_lineage_rewind(&l, 10) == true && l.n == 0);    /* before: forgets it */
        TEST_ASSERT(ds4_vision_lineage_store(&l, NULL, 0) == 1 && l.n == 0 && l.v == NULL);
        ds4_vision_lineage_clear(&l);
    }

    /* (d) refusals of the row builder */
    {
        ds4_tokens toks = {0};
        ds4_vision_embedding emb; ds4_vision_span span = {0};
        test_vision_stub_embedding(&emb, 2, 2, 0x41);
        emb.layout = 0;
        TEST_ASSERT(ds4_vision_prompt_append_rows(&toks, &span, &emb, sentinel, err, sizeof(err)) == 0);
        TEST_ASSERT(strstr(err, "layout") != NULL && emb.data != NULL && toks.len == 0);
        emb.layout = DS4_VISION_LAYOUT_DEEPSEEK4_NATURAL;
        const uint16_t *const missing[5] = { sidecar, NULL, NULL, sidecar, sidecar };
        TEST_ASSERT(ds4_vision_prompt_append_rows(&toks, &span, &emb, missing, err, sizeof(err)) == 0);
        TEST_ASSERT(strstr(err, "sentinels") != NULL && toks.len == 0);
        ds4_vision_embedding_free(&emb);
        ds4_tokens_free(&toks);
    }
    free(sidecar);
    printf("vision-prompt: ok\n");
}

/* Track B inc 8 (cache safety): a piece source that needs no model.  Token 7
 * renders the raw marker bytes so the escape can be pinned; everything else
 * renders "t<id>". */
static char *test_cache_piece(void *ud, int token, size_t *len) {
    (void)ud;
    char *s = malloc(24);
    TEST_ASSERT(s != NULL);
    if (token == 7) { memcpy(s, "\036DS4_IMAGE_\037", 12); s[12] = '\0'; *len = 12; return s; }
    const int n = snprintf(s, 24, "t%d", token);
    *len = (size_t)n;
    return s;
}

static void test_vision_cache_group(void) {
    enum { NV = 100 };   /* the stub vocabulary: ids >= NV are image rows */
    const int S0 = NV + 0, S1 = NV + 1, S2 = NV + 2, S4 = NV + 4;
    char err[256];
    size_t len = 0;
    ds4_vision_identity idA = { 1, 4, {0} }, idB = { 1, 4, {0} };
    memset(idA.identity, 0xA1, 32);
    memset(idB.identity, 0xB2, 32);
    char markA[DS4_KVSTORE_IMAGE_MARKER_LEN + 1], markB[DS4_KVSTORE_IMAGE_MARKER_LEN + 1];
    ds4_kvstore_image_marker(idA.identity, markA);
    ds4_kvstore_image_marker(idB.identity, markB);
    TEST_ASSERT(strlen(markA) == DS4_KVSTORE_IMAGE_MARKER_LEN && markA[0] == '\036' && markA[DS4_KVSTORE_IMAGE_MARKER_LEN - 1] == '\037');
    TEST_ASSERT(memcmp(markA, "\036DS4_IMAGE_a1a1", 15) == 0 && memcmp(markB, "\036DS4_IMAGE_b2b2", 15) == 0);

    /* (a) text streams render exactly as before (byte-identical, no lineage). */
    {
        const int toks[3] = { 1, 2, 3 };
        char *t = ds4_kvstore_render_text_core(test_cache_piece, NULL, toks, 3, NV, NULL, 0, &len, err, sizeof(err));
        TEST_ASSERT(t && len == 6 && memcmp(t, "t1t2t3", 6) == 0);
        free(t);
    }
    /* (b) an image block renders as ONE identity marker: same image -> same key,
     *     different image -> different key (and a different sha), never the empty
     *     string the lineage-less renderer used to produce for synthetic ids. */
    {
        const int toks[6] = { 1, S0, S2, S4, S1, 2 };   /* [t1][START IMAGE END PAD][t2] shape-wise: any synthetic run */
        char *ta = ds4_kvstore_render_text_core(test_cache_piece, NULL, toks, 6, NV, &idA, 1, &len, err, sizeof(err));
        TEST_ASSERT(ta && len == 2 + DS4_KVSTORE_IMAGE_MARKER_LEN + 2);
        TEST_ASSERT(memcmp(ta, "t1", 2) == 0 && memcmp(ta + 2, markA, DS4_KVSTORE_IMAGE_MARKER_LEN) == 0 &&
                    memcmp(ta + 2 + DS4_KVSTORE_IMAGE_MARKER_LEN, "t2", 2) == 0);
        size_t lenb = 0;
        char *tb = ds4_kvstore_render_text_core(test_cache_piece, NULL, toks, 6, NV, &idB, 1, &lenb, err, sizeof(err));
        TEST_ASSERT(tb && lenb == len && memcmp(ta, tb, len) != 0);
        char sha_a[41], sha_b[41];
        ds4_kvstore_sha1_bytes_hex(ta, len, sha_a);
        ds4_kvstore_sha1_bytes_hex(tb, lenb, sha_b);
        TEST_ASSERT(strcmp(sha_a, sha_b) != 0);
        TEST_ASSERT(ds4_kvstore_text_has_image_marker(ta, len));
        TEST_ASSERT(!ds4_kvstore_text_has_image_marker("t1t2t3", 6));
        TEST_ASSERT(ds4_kvstore_text_lineage_matches(ta, len, &idA, 1));
        TEST_ASSERT(!ds4_kvstore_text_lineage_matches(ta, len, &idB, 1));
        TEST_ASSERT(!ds4_kvstore_text_lineage_matches(ta, len, NULL, 0));      /* an image key over a text lineage */
        TEST_ASSERT(!ds4_kvstore_text_lineage_matches("t1t2", 4, &idA, 1));   /* a text key over an image lineage */
        TEST_ASSERT(ds4_kvstore_text_lineage_matches("t1t2", 4, NULL, 0));
        /* the second copy of the same block, in order */
        ds4_vision_identity two[2] = { idA, idB };
        two[1].token_start = 7;
        const int toks2[11] = { 1, S0, S2, S4, S1, 2, 3, S0, S2, S4, S1 };
        size_t len2 = 0;
        char *t2 = ds4_kvstore_render_text_core(test_cache_piece, NULL, toks2, 11, NV, two, 2, &len2, err, sizeof(err));
        TEST_ASSERT(t2 && len2 == 2 * DS4_KVSTORE_IMAGE_MARKER_LEN + 6);
        TEST_ASSERT(ds4_kvstore_text_lineage_matches(t2, len2, two, 2));
        ds4_vision_identity swapped[2] = { two[1], two[0] };
        swapped[0].token_start = 1; swapped[1].token_start = 7;
        TEST_ASSERT(!ds4_kvstore_text_lineage_matches(t2, len2, swapped, 2));   /* order matters */
        free(ta); free(tb); free(t2);
    }
    /* (c) refusals by name: image rows without lineage; lineage over text rows;
     *     lineage past the stream; lineage the stream never reaches. */
    {
        const int toks[6] = { 1, S0, S2, S4, S1, 2 };
        TEST_ASSERT(ds4_kvstore_render_text_core(test_cache_piece, NULL, toks, 6, NV, NULL, 0, &len, err, sizeof(err)) == NULL);
        TEST_ASSERT(strstr(err, "no lineage entry") != NULL && len == 0);
        ds4_vision_identity bad = idA; bad.token_start = 0;             /* covers t1 */
        TEST_ASSERT(ds4_kvstore_render_text_core(test_cache_piece, NULL, toks, 6, NV, &bad, 1, &len, err, sizeof(err)) == NULL);
        TEST_ASSERT(strstr(err, "covers text rows") != NULL);
        bad = idA; bad.token_count = 6;                                  /* past the end */
        TEST_ASSERT(ds4_kvstore_render_text_core(test_cache_piece, NULL, toks, 6, NV, &bad, 1, &len, err, sizeof(err)) == NULL);
        TEST_ASSERT(strstr(err, "exceeds") != NULL);
        bad = idA; bad.token_start = 9;                                  /* never reached (and the rows go unkeyed) */
        TEST_ASSERT(ds4_kvstore_render_text_core(test_cache_piece, NULL, toks, 6, NV, &bad, 1, &len, err, sizeof(err)) == NULL);
        const int text_only[2] = { 1, 2 };
        bad = idA; bad.token_start = 5;
        TEST_ASSERT(ds4_kvstore_render_text_core(test_cache_piece, NULL, text_only, 2, NV, &bad, 1, &len, err, sizeof(err)) == NULL);
        TEST_ASSERT(strstr(err, "beyond") != NULL);
    }
    /* (d) a marker cannot be forged from token text: 0x1E/0x1F are escaped, the
     *     scanner ignores escape pairs, and the escaped form is still injective. */
    {
        const int toks[3] = { 1, 7, 2 };
        char *t = ds4_kvstore_render_text_core(test_cache_piece, NULL, toks, 3, NV, NULL, 0, &len, err, sizeof(err));
        TEST_ASSERT(t && len == 2 + 14 + 2);
        TEST_ASSERT(memcmp(t + 2, "\036\036DS4_IMAGE_\036\037", 14) == 0);
        TEST_ASSERT(!ds4_kvstore_text_has_image_marker(t, len));
        TEST_ASSERT(ds4_kvstore_text_lineage_matches(t, len, NULL, 0));
        size_t elen = 0;
        char *e = ds4_kvstore_escape_text("a\036b\037c", 5, &elen);
        TEST_ASSERT(e && elen == 7 && memcmp(e, "a\036\036b\036\037c", 7) == 0);
        free(e);
        free(t);
    }
    /* (e) a byte-LCP never splits a marker or an escape pair. */
    {
        char text[2 + DS4_KVSTORE_IMAGE_MARKER_LEN + 2 + 1];
        memcpy(text, "ab", 2);
        memcpy(text + 2, markA, DS4_KVSTORE_IMAGE_MARKER_LEN);
        memcpy(text + 2 + DS4_KVSTORE_IMAGE_MARKER_LEN, "cd", 3);
        const size_t tl = 2 + DS4_KVSTORE_IMAGE_MARKER_LEN + 2;
        TEST_ASSERT(ds4_kvstore_lcp_marker_cap(text, tl, 1) == 1);
        TEST_ASSERT(ds4_kvstore_lcp_marker_cap(text, tl, 2) == 2);
        TEST_ASSERT(ds4_kvstore_lcp_marker_cap(text, tl, 2 + 10) == 2);                                  /* inside the marker */
        TEST_ASSERT(ds4_kvstore_lcp_marker_cap(text, tl, 2 + DS4_KVSTORE_IMAGE_MARKER_LEN - 1) == 2);   /* one byte short */
        TEST_ASSERT(ds4_kvstore_lcp_marker_cap(text, tl, 2 + DS4_KVSTORE_IMAGE_MARKER_LEN) == 2 + DS4_KVSTORE_IMAGE_MARKER_LEN);
        TEST_ASSERT(ds4_kvstore_lcp_marker_cap(text, tl, tl) == tl);
        TEST_ASSERT(ds4_kvstore_lcp_marker_cap(text, tl, tl + 5) == tl);
        TEST_ASSERT(ds4_kvstore_lcp_marker_cap("a\036\036b", 4, 2) == 1);   /* an escape pair split */
        TEST_ASSERT(ds4_kvstore_lcp_marker_cap("a\036\036b", 4, 3) == 3);
    }
    /* (f) the exact-prefix + text-suffix splice refuses a suffix that carries an
     *     image block (before touching any tokenizer). */
    {
        ds4_tokens prefix = {0}, out = {0};
        ds4_tokens_push(&prefix, 5);
        ds4_tokens_push(&out, 9);
        TEST_ASSERT(ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(NULL, &prefix, markA, &out) == 0);
        TEST_ASSERT(out.len == 0);
        ds4_tokens_free(&prefix);
        ds4_tokens_free(&out);
    }
    /* (g) payload lineage (v4): round trip; a v3 payload with image rows is
     *     refused; a v4 list must describe exactly the image rows. */
    {
        const int toks[11] = { 1, S0, S2, S4, S1, 2, 3, S0, S2, S4, S1 };
        ds4_vision_identity two[2] = { idA, idB };
        two[1].token_start = 7;
        TEST_ASSERT(ds4_session_payload_lineage_bytes(2) == 4 + 2 * DS4_SESSION_PAYLOAD_LINEAGE_ENTRY_BYTES);
        FILE *fp = tmpfile();
        TEST_ASSERT(fp != NULL);
        TEST_ASSERT(ds4_session_payload_write_lineage(fp, two, 2, err, sizeof(err)) == 0);
        TEST_ASSERT((uint64_t)ftell(fp) == ds4_session_payload_lineage_bytes(2));
        rewind(fp);
        uint64_t remaining = ds4_session_payload_lineage_bytes(2);
        ds4_vision_lineage got = {0};
        TEST_ASSERT(ds4_session_payload_read_lineage(fp, 4, &remaining, toks, 11, NV, &got, err, sizeof(err)) == 0);
        TEST_ASSERT(got.n == 2 && remaining == 0);
        TEST_ASSERT(got.v[0].token_start == 1 && got.v[0].token_count == 4 && memcmp(got.v[0].identity, idA.identity, 32) == 0);
        TEST_ASSERT(got.v[1].token_start == 7 && got.v[1].token_count == 4 && memcmp(got.v[1].identity, idB.identity, 32) == 0);
        ds4_vision_lineage_clear(&got);
        /* the same bytes against a stream where the second block is text: refused */
        rewind(fp);
        remaining = ds4_session_payload_lineage_bytes(2);
        const int toks_text_tail[11] = { 1, S0, S2, S4, S1, 2, 3, 4, 5, 6, 8 };
        const int pure_text[7] = { 1, 2, 3, 4, 5, 6, 8 };   /* no image rows at all */
        TEST_ASSERT(ds4_session_payload_read_lineage(fp, 4, &remaining, toks_text_tail, 11, NV, &got, err, sizeof(err)) != 0);
        TEST_ASSERT(strstr(err, "describe") != NULL && got.n == 0);
        /* v3 semantics: nothing is read; image rows in the stream are refused by name */
        rewind(fp);
        remaining = 99;
        TEST_ASSERT(ds4_session_payload_read_lineage(fp, 3, &remaining, toks, 11, NV, &got, err, sizeof(err)) != 0);
        TEST_ASSERT(strstr(err, "predates") != NULL && remaining == 99);
        TEST_ASSERT(ds4_session_payload_read_lineage(fp, 3, &remaining, pure_text, 7, NV, &got, err, sizeof(err)) == 0);
        TEST_ASSERT(got.n == 0 && remaining == 99);
        fclose(fp);
        /* an empty v4 list over a stream with image rows: refused (does not cover) */
        fp = tmpfile();
        TEST_ASSERT(fp != NULL);
        TEST_ASSERT(ds4_session_payload_write_lineage(fp, NULL, 0, err, sizeof(err)) == 0);
        rewind(fp);
        remaining = 4;
        TEST_ASSERT(ds4_session_payload_read_lineage(fp, 4, &remaining, toks, 11, NV, &got, err, sizeof(err)) != 0);
        TEST_ASSERT(strstr(err, "cover") != NULL);
        rewind(fp);
        remaining = 4;
        TEST_ASSERT(ds4_session_payload_read_lineage(fp, 4, &remaining, pure_text, 7, NV, &got, err, sizeof(err)) == 0 && got.n == 0);
        /* a truncated list */
        rewind(fp);
        remaining = 2;
        TEST_ASSERT(ds4_session_payload_read_lineage(fp, 4, &remaining, pure_text, 7, NV, &got, err, sizeof(err)) != 0);
        fclose(fp);
        ds4_vision_lineage_clear(&got);
    }
    /* (h) reference spans: data-less spans are fine inside the held prefix and
     *     refused (by block) where the sync would have to prefill them. */
    {
        ds4_vision_span ref[2] = {0};
        ref[0].token_start = 1;  ref[0].embedding.token_count = 4;  ref[0].embedding.data = NULL;
        ref[1].token_start = 20; ref[1].embedding.token_count = 4;  ref[1].embedding.data = (float *)1;
        TEST_ASSERT(ds4_vision_spans_require_data(ref, 2, 5, err, sizeof(err)) == 1);    /* block 0 ends at 5 == resume */
        TEST_ASSERT(ds4_vision_spans_require_data(ref, 2, 30, err, sizeof(err)) == 1);
        TEST_ASSERT(ds4_vision_spans_require_data(ref, 2, 4, err, sizeof(err)) == 0);    /* one row short */
        TEST_ASSERT(strstr(err, "block 0 @1") != NULL && strstr(err, "re-encode") != NULL);
        TEST_ASSERT(ds4_vision_spans_require_data(ref, 2, 0, err, sizeof(err)) == 0);
        ref[1].embedding.data = NULL;
        TEST_ASSERT(ds4_vision_spans_require_data(ref, 2, 24, err, sizeof(err)) == 1);
        TEST_ASSERT(ds4_vision_spans_require_data(ref, 2, 23, err, sizeof(err)) == 0 && strstr(err, "block 1 @20") != NULL);
        TEST_ASSERT(ds4_vision_spans_require_data(NULL, 0, 0, err, sizeof(err)) == 1);
    }
    /* (i) the capped token walk passes through a block only where the incoming
     *     spans reproduce it. */
    {
        ds4_vision_lineage l = {0};
        ds4_vision_span same = {0}; same.token_start = 10; same.embedding.token_count = 20; same.embedding.data = (float *)1;
        memset(same.embedding.identity, 0xAA, 32);
        TEST_ASSERT(ds4_vision_lineage_store(&l, &same, 1) == 1);
        TEST_ASSERT(ds4_vision_lineage_prefix_cap_matching(&l, 40, &same, 1) == 40);
        TEST_ASSERT(ds4_vision_lineage_prefix_cap_matching(&l, 40, NULL, 0) == 10);
        ds4_vision_span other = same; memset(other.embedding.identity, 0xBB, 32);
        TEST_ASSERT(ds4_vision_lineage_prefix_cap_matching(&l, 40, &other, 1) == 10);
        ds4_vision_span moved = same; moved.token_start = 11;
        TEST_ASSERT(ds4_vision_lineage_prefix_cap_matching(&l, 40, &moved, 1) == 10);
        ds4_vision_span shorter = same; shorter.embedding.token_count = 19;
        TEST_ASSERT(ds4_vision_lineage_prefix_cap_matching(&l, 40, &shorter, 1) == 10);
        TEST_ASSERT(ds4_vision_lineage_prefix_cap_matching(&l, 5, NULL, 0) == 5);       /* the walk ends before it */
        TEST_ASSERT(ds4_vision_lineage_prefix_cap_matching(&l, 10, NULL, 0) == 10);
        ds4_vision_span ref = same; ref.embedding.data = NULL;                         /* a reference span reproduces it too */
        TEST_ASSERT(ds4_vision_lineage_prefix_cap_matching(&l, 40, &ref, 1) == 40);
        TEST_ASSERT(ds4_vision_identities_prefix_cap(l.v, l.n, 40) == 10);
        TEST_ASSERT(ds4_vision_identities_prefix_cap(l.v, l.n, 3) == 3);
        TEST_ASSERT(ds4_vision_identities_prefix_cap(NULL, 0, 40) == 40);
        ds4_vision_lineage_clear(&l);
    }
    printf("vision-cache: ok\n");
}

/* Track B inc 5: the multimodal compaction protocol's pure half.  A transcript
 * of 100 tokens holds four image blocks: A @10 (10 rows), B @30, C @50 with NO
 * embedding (a reference span: restored by identity, data gone), D @70. */
static void test_vision_compact_group(void) {
    float dummy = 0.0f;
    ds4_vision_span img[4];
    memset(img, 0, sizeof(img));
    const uint32_t starts[4] = {10u, 30u, 50u, 70u};
    for (int i = 0; i < 4; i++) {
        img[i].token_start = starts[i];
        img[i].embedding.token_count = 10u;
        img[i].embedding.data = i == 2 ? NULL : &dummy;
        memset(img[i].embedding.identity, 0x40 + i, sizeof(img[i].embedding.identity));
    }
    /* (a) the tail start: pushed past the data-less block C, pulled back to the
     *     start of a straddled data-bearing block, clamped to bottom, untouched
     *     when no block is involved. */
    TEST_ASSERT(ds4_vision_spans_compact_tail(img, 4, 0u, 100u) == 60u);      /* C reaches past 0 */
    TEST_ASSERT(ds4_vision_spans_compact_tail(img, 4, 35u, 100u) == 60u);     /* B straddled, but C wins first */
    TEST_ASSERT(ds4_vision_spans_compact_tail(img, 4, 55u, 100u) == 60u);     /* inside C: past its end */
    TEST_ASSERT(ds4_vision_spans_compact_tail(img, 4, 60u, 100u) == 60u);     /* at C's end: unchanged */
    TEST_ASSERT(ds4_vision_spans_compact_tail(img, 4, 65u, 100u) == 65u);     /* between C and D */
    TEST_ASSERT(ds4_vision_spans_compact_tail(img, 4, 75u, 100u) == 70u);     /* D straddled: its start */
    TEST_ASSERT(ds4_vision_spans_compact_tail(img, 4, 70u, 100u) == 70u);     /* on D's first row */
    TEST_ASSERT(ds4_vision_spans_compact_tail(img, 4, 80u, 100u) == 80u);     /* just after D */
    TEST_ASSERT(ds4_vision_spans_compact_tail(img, 4, 100u, 100u) == 100u);   /* empty tail stays empty */
    TEST_ASSERT(ds4_vision_spans_compact_tail(img, 4, 120u, 100u) == 100u);   /* clamped */
    TEST_ASSERT(ds4_vision_spans_compact_tail(NULL, 0, 7u, 100u) == 7u);
    {   /* a data-less block reaching past bottom empties the tail */
        ds4_vision_span late = img[2];
        late.token_start = 95u;
        TEST_ASSERT(ds4_vision_spans_compact_tail(&late, 1, 90u, 100u) == 100u);
    }
    {   /* all data-bearing: only the straddle rule applies */
        ds4_vision_span d[4];
        memcpy(d, img, sizeof(d));
        d[2].embedding.data = &dummy;
        TEST_ASSERT(ds4_vision_spans_compact_tail(d, 4, 0u, 100u) == 0u);
        TEST_ASSERT(ds4_vision_spans_compact_tail(d, 4, 55u, 100u) == 50u);
        TEST_ASSERT(ds4_vision_spans_compact_tail(d, 4, 35u, 100u) == 30u);
    }
    /* (b) retained-image selection: whole blocks inside [tail, bottom) survive
     *     with token_start rebased; the rest are dropped; ownership is shared. */
    {
        ds4_vision_span out[4];
        bool kept[4];
        memset(out, 0, sizeof(out));
        TEST_ASSERT(ds4_vision_spans_retain(img, 4, 60u, 100u, 5u, out, kept) == 1);
        TEST_ASSERT(!kept[0] && !kept[1] && !kept[2] && kept[3]);
        TEST_ASSERT(out[0].token_start == 15u && out[0].embedding.token_count == 10u);
        TEST_ASSERT(out[0].embedding.data == &dummy && out[0].embedding.identity[0] == 0x43);
        TEST_ASSERT(img[3].token_start == 70u);   /* the source list is untouched */

        TEST_ASSERT(ds4_vision_spans_retain(img, 4, 0u, 100u, 5u, out, kept) == 4);
        TEST_ASSERT(kept[0] && kept[1] && kept[2] && kept[3]);
        TEST_ASSERT(out[0].token_start == 15u && out[1].token_start == 35u &&
                    out[2].token_start == 55u && out[3].token_start == 75u);
        TEST_ASSERT(out[2].embedding.data == NULL && out[1].embedding.data == &dummy);

        TEST_ASSERT(ds4_vision_spans_retain(img, 4, 30u, 100u, 0u, out, kept) == 3);   /* A dropped */
        TEST_ASSERT(!kept[0] && kept[1] && out[0].token_start == 0u && out[2].token_start == 40u);
        TEST_ASSERT(ds4_vision_spans_retain(img, 4, 30u, 75u, 0u, out, kept) == 2);    /* D crosses bottom */
        TEST_ASSERT(kept[1] && kept[2] && !kept[3]);
        TEST_ASSERT(ds4_vision_spans_retain(img, 4, 35u, 100u, 0u, out, kept) == 2);   /* B crosses the cut */
        TEST_ASSERT(!kept[1] && kept[2] && kept[3] && out[0].token_start == 15u);
        TEST_ASSERT(ds4_vision_spans_retain(img, 4, 100u, 100u, 0u, out, kept) == 0);
        TEST_ASSERT(ds4_vision_spans_retain(NULL, 0, 0u, 100u, 0u, out, kept) == 0);
        TEST_ASSERT(ds4_vision_spans_retain(img, 4, 60u, 100u, 5u, NULL, NULL) == 1);  /* count-only form */
    }
    /* (c) the two together: the cut a compaction actually uses never splits a
     *     block and never retains a data-less one it would have to prefill. */
    for (uint32_t cut = 0; cut <= 100u; cut++) {
        const uint32_t tail = ds4_vision_spans_compact_tail(img, 4, cut, 100u);
        bool kept[4];
        ds4_vision_spans_retain(img, 4, tail, 100u, 0u, NULL, kept);
        for (int i = 0; i < 4; i++) {
            const uint32_t end = img[i].token_start + img[i].embedding.token_count;
            const bool straddles = img[i].token_start < tail && end > tail;
            TEST_ASSERT(!straddles);
            if (!img[i].embedding.data) TEST_ASSERT(!kept[i]);
            if (kept[i]) TEST_ASSERT(img[i].token_start >= tail);
        }
    }
    /* (d) the sync guard's scan: first image row at or after `from`. */
    {
        ds4_tokens t = {0};
        for (int i = 0; i < 6; i++) ds4_tokens_push(&t, 100 + i);
        TEST_ASSERT(ds4_vision_prompt_first_image_row(&t, 0) == -1);
        ds4_tokens_push(&t, (int)(DS4_VISION_ID_BASE + 0));   /* START at 6 */
        ds4_tokens_push(&t, (int)(DS4_VISION_ID_BASE + 2));
        ds4_tokens_push(&t, 7);
        TEST_ASSERT(ds4_vision_prompt_first_image_row(&t, 0) == 6);
        TEST_ASSERT(ds4_vision_prompt_first_image_row(&t, 6) == 6);
        TEST_ASSERT(ds4_vision_prompt_first_image_row(&t, 7) == 7);
        TEST_ASSERT(ds4_vision_prompt_first_image_row(&t, 8) == -1);
        TEST_ASSERT(ds4_vision_prompt_first_image_row(&t, -3) == 6);
        TEST_ASSERT(ds4_vision_prompt_first_image_row(NULL, 0) == -1);
        ds4_tokens_free(&t);
    }
    printf("vision-compact: ok\n");
}

static void test_server_unit_group(void) {
    ds4_server_unit_tests_run();
}

typedef void (*test_fn)(void);

typedef struct {
    const char *flag;
    const char *name;
    const char *desc;
    test_fn fn;
} ds4_test_entry;

static const ds4_test_entry test_entries[] = {
#ifndef DS4_NO_GPU
    {"--long-context", "long-context", "long-context story fact-recall regression", test_long_story_fact_recall},
    {"--tool-call-quality", "tool-call-quality", "model emits valid DSML tool calls", test_tool_call_quality},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison on the standard Metal path", test_official_logprob_vectors},
    {"--local-golden-vectors", "local-golden-vectors", "local top-k/logit drift regression for long Metal prefill", test_local_golden_vectors},
    {"--metal-short-prefill", "metal-short-prefill", "Metal ratio-4 short prefill regression", test_metal_short_prefill_ratio4},
    {"--metal-kernels", "metal-kernels", "isolated Metal kernel numeric regressions", test_metal_kernel_group},
    {"--metal-tensor-equivalence", "metal-tensor-equivalence", "fast/quality Metal prompt-logit and greedy equivalence", test_metal_mpp_equivalence},
#endif
    {"--vision-prompt", "vision-prompt", "Track B inc 3: image-block token stream + session identity units", test_vision_prompt_group},
    {"--vision-cache", "vision-cache", "Track B inc 8: identity markers in cache keys, payload lineage, reference spans", test_vision_cache_group},
    {"--vision-compact", "vision-compact", "Track B inc 5: multimodal compaction tail, retained-image selection, sync guard scan", test_vision_compact_group},
    {"--server", "server", "server parser/rendering/cache unit tests", test_server_unit_group},
};

static void test_print_help(const char *prog) {
    printf("Usage: %s [--all | TEST...]\n\n", prog);
    puts("Tests:");
    puts("  --all");
    puts("      Run every test. This is the default, ordered from slower to faster.");
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        printf("  %-20s %s\n", test_entries[i].flag, test_entries[i].desc);
    }
    puts("  --list");
    puts("      Print test names only.");
#ifndef DS4_NO_GPU
    puts("  --metal-mpp-equivalence");
    puts("      Compatibility alias for --metal-tensor-equivalence.");
#endif
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  DS4_TEST_MODEL=FILE        Model path. Default: ds4flash.gguf");
    puts("  DS4_TEST_LONG_PROMPT=FILE  Rendered long-context story fact prompt.");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Simple official-vector fixture.");
    puts("  DS4_TEST_LOCAL_GOLDEN_FILE=FILE  Local top-k golden-vector fixture.");
    puts("  DS4_TEST_LOCAL_GOLDEN_RECORD=FILE  Re-record the golden fixture to FILE instead of checking.");
    puts("  DS4_TEST_NO_STEP_EXCLUSIONS=1  Check every official-vector step (adjudicate exclusions).");
    puts("  DS4_TEST_MPP_EQ_CASE=NAME  Run only Tensor equivalence cases whose id contains NAME.");
}

static const ds4_test_entry *test_find_entry(const char *arg) {
#ifndef DS4_NO_GPU
    if (!strcmp(arg, "--metal-mpp-equivalence")) {
        arg = "--metal-tensor-equivalence";
    }
#endif
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!strcmp(arg, test_entries[i].flag)) return &test_entries[i];
    }
    return NULL;
}

static void test_run_entry(const ds4_test_entry *entry) {
    int before = test_failures;
    fprintf(stderr, "%s:\n", entry->name);
    entry->fn();
    fprintf(stderr, "%s: ", entry->name);
    ds4_log(stderr,
            test_failures == before ? DS4_LOG_OK : DS4_LOG_ERROR,
            "%s",
            test_failures == before ? "OK" : "ERR");
    fputc('\n', stderr);
}

int main(int argc, char **argv) {
    bool run_all = argc == 1;
    bool selected[sizeof(test_entries) / sizeof(test_entries[0])] = {0};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) {
            run_all = true;
        } else if (!strcmp(argv[i], "--list")) {
            for (size_t j = 0; j < sizeof(test_entries) / sizeof(test_entries[0]); j++) {
                puts(test_entries[j].flag);
            }
            return 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            test_print_help(argv[0]);
            return 0;
        } else {
            const ds4_test_entry *entry = test_find_entry(argv[i]);
            if (!entry) {
                fprintf(stderr, "ds4-test: unknown test switch: %s\n", argv[i]);
                test_print_help(argv[0]);
                return 2;
            }
            selected[(size_t)(entry - test_entries)] = true;
        }
    }

    if (run_all) {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            test_run_entry(&test_entries[i]);
        }
    } else {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            if (selected[i]) test_run_entry(&test_entries[i]);
        }
    }

#ifndef DS4_NO_GPU
    test_close_engines();
#endif

    if (test_failures) {
        fprintf(stderr, "ds4 tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4 tests: ok");
    return 0;
}
