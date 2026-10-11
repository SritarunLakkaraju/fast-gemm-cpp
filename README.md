# Fast matrix multiply (GEMM) in C++

Single-precision matrix multiply taken step by step from the textbook triple loop to a BLIS-style
kernel, with each step benchmarked in GFLOPS and compared against OpenBLAS.

| Step | What changes | GFLOPS at n = 1000 |
|---|---|---:|
| `naive` | Triple loop (i, j, k) | 2.3 |
| `reordered` | Loop interchange (i, k, j): unit-stride inner loop | 11.4 |
| `tiled` | Cache tiling, still scalar | 10.7 |
| `simd_untiled` | NEON intrinsics, no tiling | 44.0 |
| `simd` | NEON + tiling | 63.8 |
| `packed` | Packing + 8x12 register-blocked micro-kernel | 109.7 |
| OpenBLAS, 1 thread | baseline | 112.0 |
| `packed_omp` | The packed kernel on 11 threads (OpenMP) | 654.8 |
| OpenBLAS, 11 threads | baseline | 369.1 |

Measured on an Apple M3 Pro (5 performance + 6 efficiency cores), Apple clang 21, `-O3 -mcpu=native`, no `-ffast-math`. Raw output is in `results/`.

## What the measurements showed

- The packed kernel is within about 5% of OpenBLAS on one core for large matrices, and clearly
  behind it for small ones (58 vs 105 GFLOPS at n = 64): packing has fixed costs that a small
  multiply cannot amortise.
- Tiling alone did not speed up the scalar code on this machine, and the tiled SIMD kernel is
  *slower* at power-of-two sizes (36 GFLOPS at n = 2048 against 61 at n = 2000). The pattern fits
  cache-set conflicts between tile rows, though that was not confirmed with performance counters.
  Packing removes the effect.
- With all 11 cores this kernel reaches 693 GFLOPS at n = 2000. The Homebrew OpenBLAS build peaks
  at 544 GFLOPS with 5 threads and does not improve when the efficiency cores are added. The
  parallel loop here uses a dynamic schedule so slow cores simply take fewer blocks.
- Every kernel is checked against OpenBLAS's result in the same run that times it.

## How the fast kernel works

```
for jc in N step NC            slab of B
  for pc in K step KC          pack B        -> stays in L2/L3
    for ic in M step MC        pack A        -> stays in L2     (the OpenMP loop)
      for jr in nc step 12     one strip of packed B            -> stays in L1
        for ir in mc step 8    micro-kernel: an 8 x 12 block of C held in 24 NEON registers
```

The micro-kernel loads 5 vectors and performs 24 fused multiply-adds per step of k without
touching C; the block is written back once at the end. Operands are first copied ("packed") into
contiguous buffers in exactly the order the kernel reads them.

## Requirements

- macOS on Apple Silicon (the SIMD code is ARM NEON) with Apple clang
- Homebrew `libomp` and `openblas` (the baseline)

```sh
brew install libomp openblas
```

## Build and run

```sh
make          # build bench_gemm
make run      # full benchmark and thread-scaling sweep -> results/
./bench_gemm /dev/null 11 256,1000     # quick run: threads, then a list of sizes
```

## Files

| File | Contents |
|---|---|
| `gemm.h` | All kernels, header-only: `naive`, `reordered`, `tiled`, `simd`, `packed`, `packed_omp`, `sgemm` |
| `bench_gemm.cpp` | Benchmark, correctness check and thread-scaling sweep |
| `deps/` | Timer and aligned-buffer helpers, shared compiler flags |
| `results/` | Benchmark output from the test machine |

## Limitations

- An AVX2 + FMA micro-kernel for x86 is included and compiles, but it has never been run.
- Only `C = A * B`: no transposes, no alpha/beta, single precision only.
- Block sizes were tuned for one processor.

## Part of a series

Five projects that build on each other:

1. [fast-gemm-cpp](https://github.com/SritarunLakkaraju/fast-gemm-cpp) - matrix multiply, the kernel everything else uses
2. [conv2d-kernels-cpp](https://github.com/SritarunLakkaraju/conv2d-kernels-cpp) - convolution on top of that GEMM
3. [tiny-inference-engine-cpp](https://github.com/SritarunLakkaraju/tiny-inference-engine-cpp) - a small inference engine on top of both
4. [int8-quantized-kernels-cpp](https://github.com/SritarunLakkaraju/int8-quantized-kernels-cpp) - the same ideas in 8-bit integers
5. [transformer-kernels-cpp](https://github.com/SritarunLakkaraju/transformer-kernels-cpp) - softmax, LayerNorm and attention

Each repository is self-contained: the headers it needs from the earlier projects are copied into `deps/`.
