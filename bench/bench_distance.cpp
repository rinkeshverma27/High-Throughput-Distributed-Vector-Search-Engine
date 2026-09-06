#include "vectordb/engine/distance.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "=== Distance Kernels Benchmark (AVX2 vs Scalar) ===" << std::endl;

    const size_t dim = 768; // BERT embedding dimension
    const size_t iterations = 1000000; // 1M distance operations

    std::vector<float> a(dim, 0.5f);
    std::vector<float> b(dim, 0.75f);

    // Warmup
    volatile float dummy = 0.0f;
    for (int i = 0; i < 1000; ++i) {
        dummy += vectordb::DistanceKernel::l2_distance_scalar(a.data(), b.data(), dim);
    }

    // Scalar Benchmark
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        dummy += vectordb::DistanceKernel::l2_distance_scalar(a.data(), b.data(), dim);
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    auto elapsed_scalar = std::chrono::duration_cast<std::chrono::microseconds>(end_scalar - start_scalar).count();

    // AVX2 Benchmark
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        dummy += vectordb::DistanceKernel::l2_distance_avx2(a.data(), b.data(), dim);
    }
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    auto elapsed_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end_avx2 - start_avx2).count();

    double ns_per_op_scalar = (double)elapsed_scalar * 1000.0 / iterations;
    double ns_per_op_avx2 = (double)elapsed_avx2 * 1000.0 / iterations;

    std::cout << "Dimension: " << dim << " (768-d)\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Scalar L2: " << elapsed_scalar / 1000.0 << " ms (" << ns_per_op_scalar << " ns/op)\n";
    std::cout << "AVX2 L2:   " << elapsed_avx2 / 1000.0 << " ms (" << ns_per_op_avx2 << " ns/op)\n";
    std::cout << "Speedup:   " << (ns_per_op_scalar / ns_per_op_avx2) << "x\n";

    return 0;
}
