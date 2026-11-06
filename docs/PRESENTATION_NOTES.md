# Presentation notes

## Slide plan (15 slides, 10–12 min)

1. Title
2. Edge detection has to be fast: 33.2M pixels per 8K frame; serial = 126 ms ≈ 8 fps
3. Blur first, then find the edges: pipeline + before/after image
4. Three versions of the same filter: data parallelism, syllabus mapping
5. OpenMP: one extra line; shared vs private variables, no locks needed
6. CUDA: one thread per pixel; blocks, tile + halo in shared memory, `__syncthreads()`
7. How we measured: hardware, image sizes, timing, correctness check

**→ live demo here** (`bash demo.sh`, then Colab section 10)

8. The GPU wins, OpenMP close behind
9. OpenMP speedup flattens after 4 threads
10. Dynamic scheduling beats static
11. Amdahl's law sets the ceiling: f ≈ 0.15 → max ≈ 6.6×
12. GPU time goes to copying data: 79% copies
13. Shared memory helped blur, not Sobel
14. What we learned
15. Future work + questions

## Demo checklist

- ~10 min before: open the notebook in Colab, *Run all* (needs a T4 GPU runtime)
- Terminal font large (Cmd +), charger plugged in, heavy apps closed
- Live times vary ~10% from the slides: normal run-to-run variation
- Backup: a screen recording of both demo parts in case Wi-Fi or Colab fails

## Likely viva questions

- **Why blur before Sobel?** Sobel amplifies noise; blurring first removes it.
- **Is there a race condition?** No: each thread writes a different output pixel and only reads the input.
- **Why two separate passes / kernels?** Sobel needs the *finished* blurred neighbours, including ones
  computed by other blocks; the kernel boundary acts as a global barrier.
- **Why is OpenMP with 1 thread slower than serial?** Parallel-runtime overhead, and the compiler
  optimises the outlined parallel function less aggressively.
- **Why 16×16 blocks?** 256 threads = 8 warps per block: enough to hide memory latency, within the
  1024-thread limit. Try `BLOCKS="8 16 32" bash bench.sh gpu`.
- **What is a warp?** 32 threads that execute the same instruction together (SIMT).
- **What is the Karp–Flatt metric?** The serial fraction f solved from Amdahl's law using a measured
  speedup: f = (1/S − 1/p) / (1 − 1/p).
- **How could the GPU version be faster?** Pinned memory and CUDA streams to overlap copies with
  compute; keep data on the GPU across many operations; a separable blur (10 multiplies instead of 25).
- **How did you verify correctness?** Byte-for-byte comparison of every output with the serial output.
- **How did you time it?** CPU: `clock_gettime`, best of 5 runs. GPU: `cudaEvent` timers, kernels
  and copies timed separately.
