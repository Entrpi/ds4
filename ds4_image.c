#include "ds4_image.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JPEG_IMPLEMENTATION
#define PNG_IMPLEMENTATION
#include "third_party/iris/jpeg.h"
#include "third_party/iris/png.h"

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} ds4_sha256;

static const uint32_t ds4_sha256_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static uint32_t ds4_rotr32(uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32u - bits));
}

static void ds4_sha256_block(ds4_sha256 *sha, const uint8_t block[64]) {
    uint32_t w[64];
    for (unsigned i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (unsigned i = 16; i < 64; i++) {
        uint32_t s0 = ds4_rotr32(w[i - 15], 7) ^ ds4_rotr32(w[i - 15], 18) ^
                      (w[i - 15] >> 3);
        uint32_t s1 = ds4_rotr32(w[i - 2], 17) ^ ds4_rotr32(w[i - 2], 19) ^
                      (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = sha->state[0], b = sha->state[1];
    uint32_t c = sha->state[2], d = sha->state[3];
    uint32_t e = sha->state[4], f = sha->state[5];
    uint32_t g = sha->state[6], h = sha->state[7];
    for (unsigned i = 0; i < 64; i++) {
        uint32_t s1 = ds4_rotr32(e, 6) ^ ds4_rotr32(e, 11) ^ ds4_rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + s1 + ch + ds4_sha256_k[i] + w[i];
        uint32_t s0 = ds4_rotr32(a, 2) ^ ds4_rotr32(a, 13) ^ ds4_rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    sha->state[0] += a; sha->state[1] += b;
    sha->state[2] += c; sha->state[3] += d;
    sha->state[4] += e; sha->state[5] += f;
    sha->state[6] += g; sha->state[7] += h;
}

static void ds4_sha256_init(ds4_sha256 *sha) {
    static const uint32_t initial[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    memcpy(sha->state, initial, sizeof(initial));
    sha->bytes = 0;
    sha->used = 0;
}

static void ds4_sha256_update(ds4_sha256 *sha, const void *data, size_t len) {
    const uint8_t *p = data;
    sha->bytes += len;
    while (len) {
        size_t take = 64 - sha->used;
        if (take > len) take = len;
        memcpy(sha->block + sha->used, p, take);
        sha->used += take;
        p += take;
        len -= take;
        if (sha->used == 64) {
            ds4_sha256_block(sha, sha->block);
            sha->used = 0;
        }
    }
}

static void ds4_sha256_final(ds4_sha256 *sha, uint8_t out[32]) {
    uint64_t bits = sha->bytes * 8u;
    sha->block[sha->used++] = 0x80;
    if (sha->used > 56) {
        memset(sha->block + sha->used, 0, 64 - sha->used);
        ds4_sha256_block(sha, sha->block);
        sha->used = 0;
    }
    memset(sha->block + sha->used, 0, 56 - sha->used);
    for (unsigned i = 0; i < 8; i++) {
        sha->block[63 - i] = (uint8_t)(bits >> (i * 8));
    }
    ds4_sha256_block(sha, sha->block);
    for (unsigned i = 0; i < 8; i++) {
        out[i * 4] = (uint8_t)(sha->state[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(sha->state[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(sha->state[i] >> 8);
        out[i * 4 + 3] = (uint8_t)sha->state[i];
    }
}

static void ds4_image_error(char *error, size_t cap, const char *message) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", message);
}

static uint16_t ds4_exif_u16(const uint8_t *p, int little) {
    return little ? (uint16_t)(p[0] | (p[1] << 8))
                  : (uint16_t)((p[0] << 8) | p[1]);
}

static uint32_t ds4_exif_u32(const uint8_t *p, int little) {
    if (little) {
        return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
               ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static int ds4_jpeg_orientation(const uint8_t *data, size_t len) {
    size_t pos = 2;
    while (pos + 4 <= len && data[pos] == 0xff) {
        uint8_t marker = data[pos + 1];
        pos += 2;
        if (marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd9)) continue;
        if (pos + 2 > len) break;
        size_t segment = ((size_t)data[pos] << 8) | data[pos + 1];
        if (segment < 2 || segment > len - pos) break;
        if (marker == 0xe1 && segment >= 14 &&
            memcmp(data + pos + 2, "Exif\0\0", 6) == 0) {
            const uint8_t *tiff = data + pos + 8;
            size_t tiff_len = segment - 8;
            if (tiff_len < 8) break;
            int little = tiff[0] == 'I' && tiff[1] == 'I';
            int big = tiff[0] == 'M' && tiff[1] == 'M';
            if ((!little && !big) || ds4_exif_u16(tiff + 2, little) != 42) break;
            uint32_t ifd = ds4_exif_u32(tiff + 4, little);
            if (ifd > tiff_len || tiff_len - ifd < 2) break;
            uint16_t count = ds4_exif_u16(tiff + ifd, little);
            size_t entries = (size_t)count * 12;
            if (entries > tiff_len - ifd - 2) break;
            for (uint16_t i = 0; i < count; i++) {
                const uint8_t *entry = tiff + ifd + 2 + (size_t)i * 12;
                if (ds4_exif_u16(entry, little) == 0x0112 &&
                    ds4_exif_u16(entry + 2, little) == 3 &&
                    ds4_exif_u32(entry + 4, little) == 1) {
                    int orientation = ds4_exif_u16(entry + 8, little);
                    return orientation >= 1 && orientation <= 8 ? orientation : 1;
                }
            }
        }
        if (marker == 0xda) break;
        pos += segment;
    }
    return 1;
}

static int ds4_oriented_rgb(
        ds4_image *out,
        const uint8_t *pixels,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        int orientation) {
    uint32_t out_width = orientation >= 5 ? height : width;
    uint32_t out_height = orientation >= 5 ? width : height;
    size_t bytes = (size_t)out_width * out_height * 3;
    uint8_t *rgb = malloc(bytes);
    if (!rgb) return 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t ox = x, oy = y;
            switch (orientation) {
                case 2: ox = width - 1 - x; break;
                case 3: ox = width - 1 - x; oy = height - 1 - y; break;
                case 4: oy = height - 1 - y; break;
                case 5: ox = y; oy = x; break;
                case 6: ox = height - 1 - y; oy = x; break;
                case 7: ox = height - 1 - y; oy = width - 1 - x; break;
                case 8: ox = y; oy = width - 1 - x; break;
                default: break;
            }
            const uint8_t *src = pixels + ((size_t)y * width + x) * channels;
            uint8_t *dst = rgb + ((size_t)oy * out_width + ox) * 3;
            if (channels < 3) {
                dst[0] = dst[1] = dst[2] = src[0];
            } else {
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            }
        }
    }

    out->width = out_width;
    out->height = out_height;
    out->rgb = rgb;
    ds4_sha256 sha;
    ds4_sha256_init(&sha);
    ds4_sha256_update(&sha, &out_width, sizeof(out_width));
    ds4_sha256_update(&sha, &out_height, sizeof(out_height));
    ds4_sha256_update(&sha, rgb, bytes);
    ds4_sha256_final(&sha, out->fingerprint);
    return 1;
}

int ds4_image_decode_memory(
        ds4_image *out,
        const uint8_t *encoded,
        size_t encoded_len,
        char *error,
        size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!encoded || encoded_len == 0 || encoded_len > DS4_IMAGE_MAX_ENCODED_BYTES) {
        ds4_image_error(error, error_cap, "image is empty or exceeds the 64 MiB encoded limit");
        return 0;
    }

    if (encoded_len >= 8 &&
        memcmp(encoded, "\x89PNG\r\n\x1a\n", 8) == 0) {
        png_image *decoded = png_load_mem(encoded, encoded_len);
        if (!decoded) {
            ds4_image_error(error, error_cap, "invalid or unsupported PNG image");
            return 0;
        }
        int ok = ds4_oriented_rgb(out, decoded->data,
                                  (uint32_t)decoded->width,
                                  (uint32_t)decoded->height,
                                  (uint32_t)decoded->channels, 1);
        png_free(decoded);
        if (!ok) ds4_image_error(error, error_cap, "unable to allocate decoded PNG pixels");
        return ok;
    }
    if (encoded_len >= 2 && encoded[0] == 0xff && encoded[1] == 0xd8) {
        jpeg_image *decoded = jpeg_load_mem(encoded, encoded_len);
        if (!decoded) {
            ds4_image_error(error, error_cap, "invalid or unsupported JPEG image");
            return 0;
        }
        int orientation = ds4_jpeg_orientation(encoded, encoded_len);
        int ok = ds4_oriented_rgb(out, decoded->data,
                                  (uint32_t)decoded->width,
                                  (uint32_t)decoded->height,
                                  (uint32_t)decoded->channels, orientation);
        jpeg_free(decoded);
        if (!ok) ds4_image_error(error, error_cap, "unable to allocate decoded JPEG pixels");
        return ok;
    }

    ds4_image_error(error, error_cap, "image must be JPEG or PNG");
    return 0;
}

int ds4_image_decode_file(
        ds4_image *out,
        const char *path,
        char *error,
        size_t error_cap) {
    if (!out || !path || !path[0]) {
        ds4_image_error(error, error_cap, "image path is empty");
        return 0;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        if (error && error_cap) snprintf(error, error_cap, "%s: %s", path, strerror(errno));
        return 0;
    }
    if (fseek(fp, 0, SEEK_END) != 0) goto io_error;
    long end = ftell(fp);
    if (end <= 0 || (unsigned long)end > DS4_IMAGE_MAX_ENCODED_BYTES) {
        fclose(fp);
        ds4_image_error(error, error_cap, "image is empty or exceeds the 64 MiB encoded limit");
        return 0;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) goto io_error;
    size_t len = (size_t)end;
    uint8_t *data = malloc(len);
    if (!data) {
        fclose(fp);
        ds4_image_error(error, error_cap, "unable to allocate encoded image buffer");
        return 0;
    }
    if (fread(data, 1, len, fp) != len) {
        free(data);
        goto io_error;
    }
    fclose(fp);
    int ok = ds4_image_decode_memory(out, data, len, error, error_cap);
    free(data);
    return ok;

io_error:
    if (error && error_cap) snprintf(error, error_cap, "%s: %s", path, strerror(errno));
    fclose(fp);
    return 0;
}

void ds4_image_free(ds4_image *image) {
    if (!image) return;
    free(image->rgb);
    memset(image, 0, sizeof(*image));
}

static uint32_t ds4_align_u32(uint32_t value, uint32_t factor) {
    return (value + factor - 1) / factor * factor;
}

static int ds4_glm53_smart_resize(
        uint32_t height,
        uint32_t width,
        uint32_t min_tokens,
        uint32_t max_tokens,
        uint32_t *target_height,
        uint32_t *target_width) {
    const uint32_t temporal = 2, factor = 28;
    uint64_t pixels_per_token = (uint64_t)temporal * factor * factor;
    uint64_t min_pixels = (uint64_t)min_tokens * pixels_per_token;
    uint64_t max_pixels = (uint64_t)max_tokens * pixels_per_token;
    uint32_t aligned_height = ds4_align_u32(height, factor);
    uint32_t aligned_width = ds4_align_u32(width, factor);
    uint64_t budget = (uint64_t)temporal * aligned_height * aligned_width;

    if (budget < min_pixels) {
        double scale = sqrt((double)min_pixels /
                            ((double)temporal * height * width));
        aligned_height = ds4_align_u32((uint32_t)ceil(height * scale), factor);
        aligned_width = ds4_align_u32((uint32_t)ceil(width * scale), factor);
        budget = (uint64_t)temporal * aligned_height * aligned_width;
    }
    if (budget > max_pixels) {
        if (max_pixels < (uint64_t)temporal * factor * factor) return 0;
        uint32_t low = 1, high = height;
        aligned_height = aligned_width = factor;
        while (low <= high) {
            uint32_t content_height = low + (high - low) / 2;
            uint32_t content_width = (uint32_t)floor(
                    (double)width * content_height / height);
            if (content_width < 1) content_width = 1;
            uint32_t candidate_height = ds4_align_u32(content_height, factor);
            uint32_t candidate_width = ds4_align_u32(content_width, factor);
            uint64_t candidate = (uint64_t)temporal * candidate_height * candidate_width;
            if (candidate <= max_pixels) {
                aligned_height = candidate_height;
                aligned_width = candidate_width;
                low = content_height + 1;
            } else {
                high = content_height - 1;
            }
        }
    }
    *target_height = aligned_height;
    *target_width = aligned_width;
    return 1;
}

static double ds4_cubic(double x, double a) {
    x = fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

static void ds4_resize_rgb_bicubic(
        const uint8_t *src,
        uint32_t src_width,
        uint32_t src_height,
        float *dst,
        uint32_t dst_width,
        uint32_t dst_height,
        uint32_t dst_stride) {
    double scale_x = (double)src_width / dst_width;
    double scale_y = (double)src_height / dst_height;
    double filter_x = scale_x >= 1.0 ? 1.0 / scale_x : 1.0;
    double filter_y = scale_y >= 1.0 ? 1.0 / scale_y : 1.0;
    double support_x = scale_x >= 1.0 ? 2.0 * scale_x : 2.0;
    double support_y = scale_y >= 1.0 ? 2.0 * scale_y : 2.0;

    for (uint32_t dy = 0; dy < dst_height; dy++) {
        double center_y = scale_y * ((double)dy + 0.5);
        int y0 = (int)(center_y - support_y + 0.5);
        int y1 = (int)(center_y + support_y + 0.5);
        if (y0 < 0) y0 = 0;
        if (y1 > (int)src_height) y1 = (int)src_height;
        for (uint32_t dx = 0; dx < dst_width; dx++) {
            double center_x = scale_x * ((double)dx + 0.5);
            int x0 = (int)(center_x - support_x + 0.5);
            int x1 = (int)(center_x + support_x + 0.5);
            if (x0 < 0) x0 = 0;
            if (x1 > (int)src_width) x1 = (int)src_width;
            double sum[3] = {0, 0, 0};
            double weight_sum = 0;
            for (int iy = y0; iy < y1; iy++) {
                double wy = ds4_cubic(((double)iy + 0.5 - center_y) * filter_y,
                                      -0.5);
                for (int ix = x0; ix < x1; ix++) {
                    double wx = ds4_cubic(((double)ix + 0.5 - center_x) * filter_x,
                                          -0.5);
                    double weight = wx * wy;
                    const uint8_t *pixel = src +
                        ((size_t)iy * src_width + (uint32_t)ix) * 3;
                    sum[0] += pixel[0] * weight;
                    sum[1] += pixel[1] * weight;
                    sum[2] += pixel[2] * weight;
                    weight_sum += weight;
                }
            }
            float *pixel = dst + ((size_t)dy * dst_stride + dx) * 3;
            for (unsigned c = 0; c < 3; c++) {
                double value = round(sum[c] / weight_sum);
                if (value < 0.0) value = 0.0;
                if (value > 255.0) value = 255.0;
                pixel[c] = (float)value;
            }
        }
    }
}

int ds4_image_preprocess_glm53(
        ds4_image_patches *out,
        const ds4_image *image,
        uint32_t min_image_tokens,
        uint32_t max_image_tokens,
        char *error,
        size_t error_cap) {
    static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static const float stddev[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!image || !image->rgb || image->width == 0 || image->height == 0 ||
        min_image_tokens == 0 || max_image_tokens < min_image_tokens ||
        max_image_tokens > 8000) {
        ds4_image_error(error, error_cap, "invalid GLM 5.3 image preprocessing parameters");
        return 0;
    }

    uint32_t target_height, target_width;
    if (!ds4_glm53_smart_resize(image->height, image->width,
                                min_image_tokens, max_image_tokens,
                                &target_height, &target_width)) {
        ds4_image_error(error, error_cap, "image token budget is too small");
        return 0;
    }
    double scale = fmin((double)target_height / image->height,
                        (double)target_width / image->width);
    uint64_t min_pixels = (uint64_t)2 * 28 * 28 * min_image_tokens;
    if ((uint64_t)2 * image->height * image->width >= min_pixels && scale > 1.0) {
        scale = 1.0;
    }
    uint32_t content_height = (uint32_t)floor(image->height * scale);
    uint32_t content_width = (uint32_t)floor(image->width * scale);
    if (content_height < 1) content_height = 1;
    if (content_width < 1) content_width = 1;
    if (content_height > target_height) content_height = target_height;
    if (content_width > target_width) content_width = target_width;

    size_t canvas_values = (size_t)target_height * target_width * 3;
    float *canvas = calloc(canvas_values, sizeof(float));
    if (!canvas) {
        ds4_image_error(error, error_cap, "unable to allocate resized image");
        return 0;
    }
    if (content_width == image->width && content_height == image->height) {
        for (uint32_t y = 0; y < content_height; y++) {
            for (uint32_t x = 0; x < content_width; x++) {
                const uint8_t *src = image->rgb + ((size_t)y * image->width + x) * 3;
                float *dst = canvas + ((size_t)y * target_width + x) * 3;
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            }
        }
    } else {
        ds4_resize_rgb_bicubic(image->rgb, image->width, image->height,
                               canvas, content_width, content_height, target_width);
    }

    for (uint32_t y = 0; y < target_height; y++) {
        for (uint32_t x = 0; x < target_width; x++) {
            float *pixel = canvas + ((size_t)y * target_width + x) * 3;
            for (unsigned c = 0; c < 3; c++) {
                float value = (x < content_width && y < content_height) ? pixel[c] : 0.0f;
                pixel[c] = (value / 255.0f - mean[c]) / stddev[c];
            }
        }
    }

    uint32_t grid_height = target_height / 14;
    uint32_t grid_width = target_width / 14;
    uint32_t patch_count = grid_height * grid_width;
    size_t patch_values = (size_t)patch_count * 3 * 2 * 14 * 14;
    float *patches = malloc(patch_values * sizeof(float));
    if (!patches) {
        free(canvas);
        ds4_image_error(error, error_cap, "unable to allocate vision patches");
        return 0;
    }

    size_t index = 0;
    for (uint32_t block_y = 0; block_y < grid_height / 2; block_y++) {
        for (uint32_t block_x = 0; block_x < grid_width / 2; block_x++) {
            for (uint32_t merge_y = 0; merge_y < 2; merge_y++) {
                for (uint32_t merge_x = 0; merge_x < 2; merge_x++) {
                    uint32_t patch_y = block_y * 2 + merge_y;
                    uint32_t patch_x = block_x * 2 + merge_x;
                    for (uint32_t channel = 0; channel < 3; channel++) {
                        for (uint32_t temporal = 0; temporal < 2; temporal++) {
                            (void)temporal;
                            for (uint32_t y = 0; y < 14; y++) {
                                for (uint32_t x = 0; x < 14; x++) {
                                    const float *pixel = canvas +
                                        ((size_t)(patch_y * 14 + y) * target_width +
                                         patch_x * 14 + x) * 3;
                                    patches[index++] = pixel[channel];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    free(canvas);
    if (index != patch_values) {
        free(patches);
        ds4_image_error(error, error_cap, "internal vision patch layout mismatch");
        return 0;
    }

    out->content_width = content_width;
    out->content_height = content_height;
    out->padded_width = target_width;
    out->padded_height = target_height;
    out->grid_width = grid_width;
    out->grid_height = grid_height;
    out->patch_count = patch_count;
    out->image_token_count = patch_count / 4;
    out->patches = patches;
    return 1;
}

void ds4_image_patches_free(ds4_image_patches *patches) {
    if (!patches) return;
    free(patches->patches);
    memset(patches, 0, sizeof(*patches));
}
