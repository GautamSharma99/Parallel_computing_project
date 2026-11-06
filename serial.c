/*
 * serial.c - the baseline: Gaussian blur, then Sobel edge detection,
 * one pixel at a time on a single CPU core.
 *
 * usage: ./serial input.pgm output.pgm [repeats]
 */
#include "common.h"

static void gaussian_blur(const uint8_t *in, uint8_t *out, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            out[(size_t)y * w + x] = blur_pixel(in, w, h, x, y);
}

static void sobel(const uint8_t *in, uint8_t *out, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            out[(size_t)y * w + x] = sobel_pixel(in, w, h, x, y);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s input.pgm output.pgm [repeats]\n", argv[0]);
        return 1;
    }
    int reps = argc > 3 ? atoi(argv[3]) : 5;
    if (reps < 1) reps = 1;

    Image img = read_pgm(argv[1]);
    int w = img.width, h = img.height;
    uint8_t *blurred = (uint8_t *)xmalloc((size_t)w * h);
    uint8_t *edges = (uint8_t *)xmalloc((size_t)w * h);

    /* Run several times and keep the fastest run (the first one also pays
     * for the OS handing us fresh memory pages). */
    double best_blur = 0, best_sobel = 0, best_total = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        gaussian_blur(img.data, blurred, w, h);
        double t1 = now_ms();
        sobel(blurred, edges, w, h);
        double t2 = now_ms();
        if (t2 - t0 < best_total) {
            best_total = t2 - t0;
            best_blur = t1 - t0;
            best_sobel = t2 - t1;
        }
    }

    write_pgm(argv[2], edges, w, h);
    print_result("serial", 1, w, h, best_blur, best_sobel, 0.0);

    free(img.data);
    free(blurred);
    free(edges);
    return 0;
}
