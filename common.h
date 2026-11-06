/*
 * common.h - shared helpers for all three versions (serial, OpenMP, CUDA).
 *
 *   - read/write grayscale PGM images (the simplest image format: a tiny text
 *     header followed by raw bytes, one byte per pixel)
 *   - a millisecond timer
 *   - the per-pixel filter math used by the serial and OpenMP versions
 *
 * It compiles as C (gcc/clang) and as C++ (nvcc), so the CUDA version can
 * reuse the image I/O and the edge-strength formula.
 */
#ifndef COMMON_H
#define COMMON_H

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Functions marked HD run on both the CPU and the GPU when compiled by nvcc. */
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

typedef struct {
    int width;
    int height;
    uint8_t *data; /* width * height bytes, row by row */
} Image;

static inline void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static inline void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) die("out of memory");
    return p;
}

/* ---------------------------------------------------------------- PGM I/O */

static inline void pgm_skip(FILE *f) { /* skip whitespace and # comments */
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n') {}
        } else if (!isspace(c)) {
            ungetc(c, f);
            return;
        }
    }
}

static inline int pgm_int(FILE *f) {
    int v;
    pgm_skip(f);
    if (fscanf(f, "%d", &v) != 1) die("malformed PGM header");
    return v;
}

static inline Image read_pgm(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }

    char magic[3] = {0};
    if (fread(magic, 1, 2, f) != 2 || strcmp(magic, "P5") != 0)
        die("input must be a binary grayscale PGM (P5) - use: python3 tools.py prepare");

    Image img;
    img.width = pgm_int(f);
    img.height = pgm_int(f);
    int maxval = pgm_int(f);
    if (img.width <= 0 || img.height <= 0 || maxval != 255)
        die("unsupported PGM (need 8-bit grayscale)");
    fgetc(f); /* exactly one whitespace byte separates header and pixels */

    size_t n = (size_t)img.width * img.height;
    img.data = (uint8_t *)xmalloc(n);
    if (fread(img.data, 1, n, f) != n) die("PGM file is truncated");
    fclose(f);
    return img;
}

static inline void write_pgm(const char *path, const uint8_t *data, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    fwrite(data, 1, (size_t)w * h, f);
    fclose(f);
}

/* ------------------------------------------------------------ timing / CSV */

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* Every program prints exactly one CSV line to stdout; bench.sh adds the header:
 * version,threads,width,height,blur_ms,sobel_ms,transfer_ms,total_ms
 * (for CUDA, "threads" = threads per block and transfer_ms = CPU<->GPU copies) */
static inline void print_result(const char *version, int threads, int w, int h,
                                double blur_ms, double sobel_ms, double transfer_ms) {
    printf("%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n", version, threads, w, h,
           blur_ms, sobel_ms, transfer_ms, blur_ms + sobel_ms + transfer_ms);
    fflush(stdout);
}

/* ------------------------------------------------------------ filter math */

static inline HD int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Edge strength from the Sobel gradients: sqrt(gx^2 + gy^2), capped at 255.
 * Shared with the GPU so every version produces bit-identical output. */
static inline HD uint8_t edge_value(int gx, int gy) {
    float mag = sqrtf((float)(gx * gx + gy * gy));
    return (uint8_t)(mag > 255.0f ? 255.0f : mag);
}

/* 5x5 Gaussian blur weights = [1 4 6 4 1] x [1 4 6 4 1]; they add up to 256. */
static const int GAUSS[5] = {1, 4, 6, 4, 1};

/* Blurred value of pixel (x, y). Pixels outside the image use the nearest
 * edge pixel ("clamp to edge"). */
static inline uint8_t blur_pixel(const uint8_t *in, int w, int h, int x, int y) {
    int sum = 0;
    for (int dy = -2; dy <= 2; dy++) {
        const uint8_t *row = in + (size_t)clampi(y + dy, 0, h - 1) * w;
        for (int dx = -2; dx <= 2; dx++)
            sum += GAUSS[dy + 2] * GAUSS[dx + 2] * row[clampi(x + dx, 0, w - 1)];
    }
    return (uint8_t)((sum + 128) >> 8); /* divide by 256, rounded */
}

/* Sobel edge strength of pixel (x, y) from its 3x3 neighbourhood:
 *     p00 p01 p02
 *     p10  .  p12
 *     p20 p21 p22                                                         */
static inline uint8_t sobel_pixel(const uint8_t *in, int w, int h, int x, int y) {
    const uint8_t *up = in + (size_t)clampi(y - 1, 0, h - 1) * w;
    const uint8_t *mid = in + (size_t)y * w;
    const uint8_t *down = in + (size_t)clampi(y + 1, 0, h - 1) * w;
    int xl = clampi(x - 1, 0, w - 1), xr = clampi(x + 1, 0, w - 1);

    int p00 = up[xl], p01 = up[x], p02 = up[xr];
    int p10 = mid[xl], p12 = mid[xr];
    int p20 = down[xl], p21 = down[x], p22 = down[xr];

    int gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20); /* left vs right */
    int gy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02); /* top vs bottom */
    return edge_value(gx, gy);
}

#endif /* COMMON_H */
