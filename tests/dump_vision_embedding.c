/* tests/dump_vision_embedding.c -- Track B inc 2 oracle harness.
 *
 * Opens BASE.gguf with the encoder sidecar SIDECAR.gguf, encodes IMAGE
 * through ds4_engine_vision_encode_file() and writes the natural-grid
 * [token_count x 4096] F32 block to OUT.bin behind a small header.  Built
 * UNCHANGED against upstream antirez/ds4 @110afdd and against this fork:
 * both round-trip BF16 at the same points, so the two dumps must be
 * byte-identical (cmp), and any difference is a port bug, not FP noise.
 *
 *   dump_vision_embedding BASE.gguf SIDECAR.gguf IMAGE OUT.bin [IMAGE OUT.bin ...]
 * (several images per engine open: the 80 GiB base loads once.)
 */
#include "ds4.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 5 || (argc - 3) % 2 != 0) {
        fprintf(stderr, "usage: %s BASE.gguf SIDECAR.gguf IMAGE OUT.bin [IMAGE OUT.bin ...]\n", argv[0]);
        return 2;
    }
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.vision_path = argv[2];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 8;
    opt.power_percent = 100;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) {
        fprintf(stderr, "dump_vision_embedding: engine open failed\n");
        return 1;
    }
    int failed = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        const char *image_path = argv[i], *out_path = argv[i + 1];
        ds4_vision_embedding emb;
        char err[512] = {0};
        if (!ds4_engine_vision_encode_file(e, image_path, &emb, err, sizeof(err))) {
            fprintf(stderr, "dump_vision_embedding: %s: encode failed: %s\n", image_path, err);
            failed++;
            continue;
        }
        FILE *f = fopen(out_path, "wb");
        if (!f) { perror(out_path); ds4_vision_embedding_free(&emb); failed++; continue; }
        const uint32_t hdr[8] = { emb.token_count, emb.layout, emb.grid_width, emb.grid_height,
                                  emb.width, emb.height, emb.content_width, emb.content_height };
        fwrite(hdr, sizeof(hdr), 1, f);
        fwrite(emb.fingerprint, sizeof(emb.fingerprint), 1, f);
        fwrite(emb.data, sizeof(float), (size_t)emb.token_count * 4096u, f);
        fclose(f);
        printf("dump_vision_embedding: %s -> %u tokens (llm grid %ux%u, content %ux%u of %ux%u), fingerprint %02x%02x%02x%02x..., %s\n",
               image_path, emb.token_count, emb.grid_width, emb.grid_height,
               emb.content_width, emb.content_height, emb.width, emb.height,
               emb.fingerprint[0], emb.fingerprint[1], emb.fingerprint[2], emb.fingerprint[3], out_path);
        ds4_vision_embedding_free(&emb);
    }
    ds4_engine_close(e);
    return failed ? 1 : 0;
}
