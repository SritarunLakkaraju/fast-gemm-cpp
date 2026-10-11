// bench_gemm.cpp - measures every optimisation step in GFLOPS and compares
// against OpenBLAS (single- and multi-threaded).
//
// usage: bench_gemm [out.csv] [threads] [n1,n2,...]
//        bench_gemm scaling [out.csv]      thread-scaling sweep at n = 2000
#include <cblas.h>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "deps/util.h"
#include "gemm.h"

using GemmFn = std::function<void(int, const float*, const float*, float*)>;

struct Kernel {
    std::string name;
    GemmFn fn;
    int max_n;       // skip sizes above this (the slow kernels take minutes at 2048)
    bool threaded;
};

static int scaling(const char* csv_path) {
    const int n = 2000, max_threads = omp_get_max_threads();
    util::avec<float> A((size_t)n * n), B((size_t)n * n), C((size_t)n * n);
    util::Rng rng(7);
    util::fill_uniform(A.data(), A.size(), rng);
    util::fill_uniform(B.data(), B.size(), rng);
    FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
    if (csv) std::fprintf(csv, "threads,packed_simd_omp,openblas\n");
    std::printf("Thread scaling at n = %d (GFLOPS)\n%8s%18s%12s\n", n, "threads", "packed_simd_omp", "openblas");
    for (int t = 1; t <= max_threads; ++t) {
        omp_set_num_threads(t);
        openblas_set_num_threads(t);
        const double flops = 2.0 * n * n * n * 1e-9;
        const double mine = flops / util::time_best([&] { gemm::packed_omp(n, n, n, A.data(), n, B.data(), n, C.data(), n); });
        const double ob = flops / util::time_best([&] {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);
        });
        std::printf("%8d%18.1f%12.1f\n", t, mine, ob);
        if (csv) std::fprintf(csv, "%d,%.2f,%.2f\n", t, mine, ob);
    }
    if (csv) std::fclose(csv);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "scaling") == 0) return scaling(argc > 2 ? argv[2] : nullptr);
    const char* csv_path = argc > 1 ? argv[1] : nullptr;
    const int max_threads = omp_get_max_threads();
    const int threads = argc > 2 ? std::atoi(argv[2]) : max_threads;

    auto own = [](void (*f)(int, int, int, const float*, int, const float*, int, float*, int)) {
        return [f](int n, const float* A, const float* B, float* C) { f(n, n, n, A, n, B, n, C, n); };
    };
    auto blas = [](int n, const float* A, const float* B, float* C) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);
    };

    const std::vector<Kernel> kernels = {
        {"naive_ijk", own(gemm::naive), 1024, false},
        {"reordered_ikj", own(gemm::reordered), 2048, false},
        {"tiled", own(gemm::tiled), 2048, false},
        {"simd_untiled", own(gemm::simd_untiled), 2048, false},
        {"tiled_simd", own(gemm::simd), 2048, false},
        {"packed_simd", own(gemm::packed), 2048, false},
        {"openblas_1t", blas, 2048, false},
        {"packed_simd_omp", own(gemm::packed_omp), 2048, true},
        {"openblas_mt", blas, 2048, true},
    };
    std::vector<int> sizes = {64, 128, 256, 512, 1000, 1024, 2000, 2048};
    if (argc > 3) {  // optional comma-separated size list
        sizes.clear();
        for (char* tok = std::strtok(argv[3], ","); tok; tok = std::strtok(nullptr, ",")) sizes.push_back(std::atoi(tok));
    }

    FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
    if (csv) std::fprintf(csv, "kernel,n,threads,seconds,gflops,max_abs_err\n");

    std::printf("GEMM benchmark, square n x n, float32, %d threads for the threaded kernels\n", threads);
    std::printf("%-18s", "kernel");
    for (int n : sizes) std::printf("%9d", n);
    std::printf("   (GFLOPS)\n");

    // Reference results per size, computed once with OpenBLAS.
    std::vector<util::avec<float>> As, Bs, Refs;
    util::Rng rng(42);
    for (int n : sizes) {
        As.emplace_back((size_t)n * n);
        Bs.emplace_back((size_t)n * n);
        Refs.emplace_back((size_t)n * n);
        util::fill_uniform(As.back().data(), (size_t)n * n, rng);
        util::fill_uniform(Bs.back().data(), (size_t)n * n, rng);
        openblas_set_num_threads(max_threads);
        blas(n, As.back().data(), Bs.back().data(), Refs.back().data());
    }

    for (const Kernel& k : kernels) {
        const int nt = k.threaded ? threads : 1;
        omp_set_num_threads(nt);
        openblas_set_num_threads(nt);
        std::printf("%-18s", k.name.c_str());
        for (size_t s = 0; s < sizes.size(); ++s) {
            const int n = sizes[s];
            if (n > k.max_n) {
                std::printf("%9s", "-");
                continue;
            }
            util::avec<float> C((size_t)n * n);
            const double t = util::time_best([&] { k.fn(n, As[s].data(), Bs[s].data(), C.data()); });
            const double gflops = 2.0 * n * n * n / t * 1e-9;
            const double err = util::max_abs_diff(C.data(), Refs[s].data(), (size_t)n * n);
            std::printf("%9.2f", gflops);
            std::fflush(stdout);
            if (csv) std::fprintf(csv, "%s,%d,%d,%.6e,%.3f,%.3e\n", k.name.c_str(), n, nt, t, gflops, err);
            // Summation order differs between kernels, so allow rounding noise that grows with n.
            if (err > 1e-5 * n) std::printf("  [MISMATCH n=%d err=%.2e]", n, err);
        }
        std::printf("\n");
    }
    if (csv) std::fclose(csv);
    return 0;
}
