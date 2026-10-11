include deps/flags.mk

all: bench_gemm

bench_gemm: bench_gemm.cpp gemm.h deps/util.h
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $(BLAS_CFLAGS) $< -o $@ $(OMP_LDFLAGS) $(BLAS_LDFLAGS)

run: bench_gemm
	mkdir -p results
	./bench_gemm results/gemm.csv | tee results/gemm.txt
	./bench_gemm scaling results/scaling.csv | tee results/scaling.txt

clean:
	rm -f bench_gemm

.PHONY: all run clean
