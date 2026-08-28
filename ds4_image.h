#ifndef DS4_IMAGE_H
#define DS4_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#define DS4_IMAGE_MAX_ENCODED_BYTES (64u * 1024u * 1024u)
#define DS4_IMAGE_MAX_DIMENSION 16384u
#define DS4_IMAGE_MAX_PIXELS (64u * 1024u * 1024u)

typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t *rgb;
    uint8_t fingerprint[32];
} ds4_image;

typedef struct {
    uint32_t content_width;
    uint32_t content_height;
    uint32_t padded_width;
    uint32_t padded_height;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t patch_count;
    uint32_t image_token_count;
    float *patches;
} ds4_image_patches;

int ds4_image_decode_memory(
        ds4_image     *out,
        const uint8_t *encoded,
        size_t         encoded_len,
        char          *error,
        size_t         error_cap);

int ds4_image_decode_file(
        ds4_image *out,
        const char *path,
        char *error,
        size_t error_cap);

void ds4_image_free(ds4_image *image);

int ds4_image_preprocess_glm53(
        ds4_image_patches *out,
        const ds4_image   *image,
        uint32_t           min_image_tokens,
        uint32_t           max_image_tokens,
        char              *error,
        size_t             error_cap);

void ds4_image_patches_free(ds4_image_patches *patches);

#endif
