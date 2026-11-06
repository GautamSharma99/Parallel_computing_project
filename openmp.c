/*
 * openmp.c - the same filters, with the image rows shared out among CPU threads.
 *
 * The only real change from serial.c is one line above each outer loop:
 *
 *     #pragma omp parallel for schedule(runtime)
 *
 * Variable scope: the loop counters y and x are private to each thread (every
 * thread has its own copy); the image pointers and sizes are shared (read-only,
 * so no race). Each thread writes different output pixels, so no locks are needed.
 *
 * schedule(runtime) lets us switch between static and dynamic scheduling with
 * the OMP_SCHEDULE environment variable, without recompiling.
 *
 * usage: OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./openmp input.pgm output.pgm [repeats]
 */
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static void gaussian_blur(const uint8_t *in, uint8_t *out, int w, int h) {
    #pragma omp parallel for schedule(runtime)
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            out[(size_t)y * w + x] = blur_pixel(in, w, h, x, y);
}

static void sobel(const uint8_t *in, uint8_t *out, int w, int h) {
    #pragma omp parallel for schedule(runtime)
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            out[(size_t)y * w + x] = sobel_pixel(in, w, h, x, y);
}

/* Label for the CSV, e.g. "openmp_static", taken from the active schedule. */
static const char *version_label(void) {
#ifdef _OPENMP
    omp_sched_t kind;
    int chunk;
    omp_get_schedule(&kind, &chunk);
    switch ((int)((unsigned)kind & 0x7fffffffu)) { /* drop the "monotonic" flag bit */
        case omp_sched_static:  return "openmp_static";
        case omp_sched_dynamic: return "openmp_dynamic";
        case omp_sched_guided:  return "openmp_guided";
        default:                return "openmp_auto";
    }
#else
    return "openmp_disabled"; /* compiled without -fopenmp */
#endif
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s input.pgm output.pgm [repeats]\n", argv[0]);
        return 1;
    }
    int reps = argc > 3 ? atoi(argv[3]) : 5;
    if (reps < 1) reps = 1;

#ifdef _OPENMP
    int threads = omp_get_max_threads();
#else
    int threads = 1;
#endif

    Image img = read_pgm(argv[1]);
    int w = img.width, h = img.height;
    uint8_t *blurred = (uint8_t *)xmalloc((size_t)w * h);
    uint8_t *edges = (uint8_t *)xmalloc((size_t)w * h);

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
    print_result(version_label(), threads, w, h, best_blur, best_sobel, 0.0);

    free(img.data);
    free(blurred);
    free(edges);
    return 0;
}
