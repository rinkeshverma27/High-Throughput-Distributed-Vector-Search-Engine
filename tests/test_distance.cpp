#include "vectordb/engine/distance.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "[Test] Running Distance Kernels Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> a(dim, 1.0f);
    std::vector<float> b(dim, 2.0f);

    float l2_scalar = vectordb::DistanceKernel::l2_distance_scalar(a.data(), b.data(), dim);
    float l2_avx2 = vectordb::DistanceKernel::l2_distance_avx2(a.data(), b.data(), dim);

    std::cout << "L2 Scalar: " << l2_scalar << ", L2 AVX2: " << l2_avx2 << std::endl;
    assert(std::abs(l2_scalar - 128.0f) < 1e-4);
    assert(std::abs(l2_avx2 - 128.0f) < 1e-4);

    float dot_scalar = vectordb::DistanceKernel::dot_product_scalar(a.data(), b.data(), dim);
    float dot_avx2 = vectordb::DistanceKernel::dot_product_avx2(a.data(), b.data(), dim);

    std::cout << "Dot Scalar: " << dot_scalar << ", Dot AVX2: " << dot_avx2 << std::endl;
    assert(std::abs(dot_scalar - 256.0f) < 1e-4);
    assert(std::abs(dot_avx2 - 256.0f) < 1e-4);

    float cos_dist = vectordb::DistanceKernel::cosine_distance(a.data(), b.data(), dim);
    std::cout << "Cosine Distance (identical angle vectors): " << cos_dist << std::endl;
    assert(std::abs(cos_dist - 0.0f) < 1e-4);

    std::cout << ">>> Distance Kernels Test PASSED! <<<" << std::endl;
    return 0;
}
