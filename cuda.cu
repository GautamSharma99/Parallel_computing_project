/*
 * cuda.cu - the same two filters on an NVIDIA GPU, in two versions:
 *
 *   naive : one GPU thread per pixel; every thread reads its neighbours
 *           straight from (slow) global memory, so each pixel is read
 *           up to 25 times by different threads.
 *   tiled : each block first copies its tile of the image - plus a border
 *           ("halo") of neighbour pixels - into fast on-chip shared memory,
 *           waits for the whole block (__syncthreads), then computes from there.
 *
 * Threads are arranged as a 2D grid of 2D blocks (default 16 x 16 = 256
 * threads per block), so thread (x, y) handles pixel (x, y).
 *
 * usage: ./cuda_filter input.pgm output.pgm [naive|tiled] [block_size] [repeats]
 */
#include "common.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                      \
                    cudaGetErrorString(err_), __FILE__, __LINE__);           \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__constant__ int c_gauss[5] = {1, 4, 6, 4, 1};

/* ======================================================= version 1: naive */

__global__ void blur_naive(const uint8_t *in, uint8_t *out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return; /* the last blocks can hang off the image */

    int sum = 0;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
            sum += c_gauss[dy + 2] * c_gauss[dx + 2] *
                   in[clampi(y + dy, 0, h - 1) * w + clampi(x + dx, 0, w - 1)];
    out[y * w + x] = (uint8_t)((sum + 128) >> 8);
}

__global__ void sobel_naive(const uint8_t *in, uint8_t *out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int xl = clampi(x - 1, 0, w - 1), xr = clampi(x + 1, 0, w - 1);
    int yu = clampi(y - 1, 0, h - 1), yd = clampi(y + 1, 0, h - 1);
    int p00 = in[yu * w + xl], p01 = in[yu * w + x], p02 = in[yu * w + xr];
    int p10 = in[y * w + xl],                         p12 = in[y * w + xr];
    int p20 = in[yd * w + xl], p21 = in[yd * w + x], p22 = in[yd * w + xr];

    int gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
    int gy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
    out[y * w + x] = edge_value(gx, gy);
}

/* ================================================= version 2: shared memory */

/* All threads of the block cooperatively copy a (blockDim + 2R)^2 tile into
 * shared memory. The tile is bigger than the block, so each thread may copy
 * more than one pixel (hence the strided loops). */
__device__ void load_tile(const uint8_t *in, uint8_t *tile, int w, int h, int R) {
    int tw = blockDim.x + 2 * R, th = blockDim.y + 2 * R;
    int x0 = blockIdx.x * blockDim.x - R; /* image coords of the tile's top-left */
    int y0 = blockIdx.y * blockDim.y - R;
    for (int ty = threadIdx.y; ty < th; ty += blockDim.y)
        for (int tx = threadIdx.x; tx < tw; tx += blockDim.x)
            tile[ty * tw + tx] = in[clampi(y0 + ty, 0, h - 1) * w + clampi(x0 + tx, 0, w - 1)];
    __syncthreads(); /* nobody computes until the whole tile is loaded */
}

__global__ void blur_tiled(const uint8_t *in, uint8_t *out, int w, int h) {
    extern __shared__ uint8_t tile[]; /* size given at kernel launch */
    const int R = 2;
    load_tile(in, tile, w, h, R);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return; /* only after __syncthreads() */

    int tw = blockDim.x + 2 * R;
    const uint8_t *t = &tile[threadIdx.y * tw + threadIdx.x]; /* top-left of my 5x5 window */
    int sum = 0;
    for (int dy = 0; dy < 5; dy++)
        for (int dx = 0; dx < 5; dx++)
            sum += c_gauss[dy] * c_gauss[dx] * t[dy * tw + dx];
    out[y * w + x] = (uint8_t)((sum + 128) >> 8);
}

__global__ void sobel_tiled(const uint8_t *in, uint8_t *out, int w, int h) {
    extern __shared__ uint8_t tile[];
    const int R = 1;
    load_tile(in, tile, w, h, R);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int tw = blockDim.x + 2 * R;
    const uint8_t *t = &tile[threadIdx.y * tw + threadIdx.x]; /* top-left of my 3x3 window */
    int p00 = t[0],          p01 = t[1],          p02 = t[2];
    int p10 = t[tw],                              p12 = t[tw + 2];
    int p20 = t[2 * tw],     p21 = t[2 * tw + 1], p22 = t[2 * tw + 2];

    int gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
    int gy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
    out[y * w + x] = edge_value(gx, gy);
}

/* =================================================================== host */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s input.pgm output.pgm [naive|tiled] [block_size] [repeats]\n", argv[0]);
        return 1;
    }
    const char *mode = argc > 3 ? argv[3] : "tiled";
    int bs = argc > 4 ? atoi(argv[4]) : 16;
    int reps = argc > 5 ? atoi(argv[5]) : 5;
    bool tiled = strcmp(mode, "tiled") == 0;
    if (!tiled && strcmp(mode, "naive") != 0) die("mode must be 'naive' or 'tiled'");
    if (bs < 1 || bs > 32) die("block size must be 1..32 (at most 32x32 = 1024 threads per block)");
    if (reps < 1) reps = 1;

    Image img = read_pgm(argv[1]);
    int w = img.width, h = img.height;
    size_t n = (size_t)w * h;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "GPU: %s (compute capability %d.%d, %d SMs)\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    uint8_t *d_in, *d_blurred, *d_edges;
    CUDA_CHECK(cudaMalloc(&d_in, n));
    CUDA_CHECK(cudaMalloc(&d_blurred, n));
    CUDA_CHECK(cudaMalloc(&d_edges, n));
    uint8_t *edges = (uint8_t *)xmalloc(n);

    dim3 block(bs, bs);
    dim3 grid((w + bs - 1) / bs, (h + bs - 1) / bs); /* enough blocks to cover the image */
    size_t smem_blur = (size_t)(bs + 4) * (bs + 4);  /* tile + 2-pixel halo */
    size_t smem_sobel = (size_t)(bs + 2) * (bs + 2); /* tile + 1-pixel halo */

    /* GPU-side timers: ev[0] copy-in ev[1] blur ev[2] sobel ev[3] copy-out ev[4] */
    cudaEvent_t ev[5];
    for (int i = 0; i < 5; i++) CUDA_CHECK(cudaEventCreate(&ev[i]));

    float best[4] = {0, 0, 0, 0};
    double best_total = 1e30;
    for (int r = 0; r < reps; r++) { /* the first run also pays one-time GPU setup */
        CUDA_CHECK(cudaEventRecord(ev[0]));
        CUDA_CHECK(cudaMemcpy(d_in, img.data, n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev[1]));
        if (tiled) blur_tiled<<<grid, block, smem_blur>>>(d_in, d_blurred, w, h);
        else       blur_naive<<<grid, block>>>(d_in, d_blurred, w, h);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev[2]));
        if (tiled) sobel_tiled<<<grid, block, smem_sobel>>>(d_blurred, d_edges, w, h);
        else       sobel_naive<<<grid, block>>>(d_blurred, d_edges, w, h);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev[3]));
        CUDA_CHECK(cudaMemcpy(edges, d_edges, n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ev[4]));
        CUDA_CHECK(cudaEventSynchronize(ev[4]));

        float t[4];
        for (int i = 0; i < 4; i++) CUDA_CHECK(cudaEventElapsedTime(&t[i], ev[i], ev[i + 1]));
        double total = t[0] + t[1] + t[2] + t[3];
        if (total < best_total) {
            best_total = total;
            for (int i = 0; i < 4; i++) best[i] = t[i];
        }
    }

    write_pgm(argv[2], edges, w, h);
    char label[32];
    snprintf(label, sizeof label, "cuda_%s", mode);
    /* blur kernel, sobel kernel, and the two copies counted as "transfer" */
    print_result(label, bs * bs, w, h, best[1], best[2], best[0] + best[3]);

    for (int i = 0; i < 5; i++) cudaEventDestroy(ev[i]);
    cudaFree(d_in);
    cudaFree(d_blurred);
    cudaFree(d_edges);
    free(img.data);
    free(edges);
    return 0;
}
