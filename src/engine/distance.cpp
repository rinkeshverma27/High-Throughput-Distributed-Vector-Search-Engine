#include "vectordb/engine/distance.hpp"
#include <cmath>
#include <immintrin.h>

namespace vectordb {

bool DistanceKernel::has_avx2() {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

float DistanceKernel::l2_distance_scalar(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float DistanceKernel::dot_product_scalar(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2,fma")))
float DistanceKernel::l2_distance_avx2(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();

    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum256 = _mm256_fmadd_ps(diff, diff, sum256);
    }

    // Horizontal sum of 8 floats in sum256
    alignas(32) float buffer[8];
    _mm256_storeu_ps(buffer, sum256);
    float total = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                  buffer[4] + buffer[5] + buffer[6] + buffer[7];

    // Cleanup tail
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }

    return total;
}

__attribute__((target("avx2,fma")))
float DistanceKernel::dot_product_avx2(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();

    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum256 = _mm256_fmadd_ps(va, vb, sum256);
    }

    alignas(32) float buffer[8];
    _mm256_storeu_ps(buffer, sum256);
    float total = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                  buffer[4] + buffer[5] + buffer[6] + buffer[7];

    for (; i < dim; ++i) {
        total += a[i] * b[i];
    }

    return total;
}
#else
float DistanceKernel::l2_distance_avx2(const float* a, const float* b, size_t dim) {
    return l2_distance_scalar(a, b, dim);
}

float DistanceKernel::dot_product_avx2(const float* a, const float* b, size_t dim) {
    return dot_product_scalar(a, b, dim);
}
#endif

float DistanceKernel::l2_distance(const float* a, const float* b, size_t dim) {
    static const bool use_avx2 = has_avx2();
    if (use_avx2) {
        return l2_distance_avx2(a, b, dim);
    }
    return l2_distance_scalar(a, b, dim);
}

float DistanceKernel::dot_product(const float* a, const float* b, size_t dim) {
    static const bool use_avx2 = has_avx2();
    if (use_avx2) {
        return dot_product_avx2(a, b, dim);
    }
    return dot_product_scalar(a, b, dim);
}

float DistanceKernel::cosine_distance(const float* a, const float* b, size_t dim) {
    float dot = dot_product(a, b, dim);
    float norm_a = dot_product(a, a, dim);
    float norm_b = dot_product(b, b, dim);
    if (norm_a <= 0.0f || norm_b <= 0.0f) return 1.0f;
    float sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    return 1.0f - sim;
}

float DistanceKernel::compute(const float* a, const float* b, size_t dim, MetricType metric) {
    switch (metric) {
        case MetricType::L2:
            return l2_distance(a, b, dim);
        case MetricType::DOT_PRODUCT:
            return -dot_product(a, b, dim); // Negate for min-heap compatibility
        case MetricType::COSINE:
            return cosine_distance(a, b, dim);
        default:
            return l2_distance(a, b, dim);
    }
}

} // namespace vectordb
