#pragma once

#include "vectordb/common/types.hpp"
#include <cstddef>

namespace vectordb {

class DistanceKernel {
public:
    // Computes squared Euclidean distance (L2^2)
    static float l2_distance(const float* a, const float* b, size_t dim);

    // Computes dot product (inner product)
    static float dot_product(const float* a, const float* b, size_t dim);

    // Computes cosine distance = 1.0 - (a . b) / (|a| * |b|)
    static float cosine_distance(const float* a, const float* b, size_t dim);

    // Dynamic dispatch by metric type
    static float compute(const float* a, const float* b, size_t dim, MetricType metric);

    // Vectorized SIMD implementations
    static float l2_distance_avx2(const float* a, const float* b, size_t dim);
    static float dot_product_avx2(const float* a, const float* b, size_t dim);

    // Scalar fallback implementations
    static float l2_distance_scalar(const float* a, const float* b, size_t dim);
    static float dot_product_scalar(const float* a, const float* b, size_t dim);

    // CPU feature detection
    static bool has_avx2();
};

} // namespace vectordb
