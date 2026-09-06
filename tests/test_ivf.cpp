#include "vectordb/engine/ivf_index.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

int main() {
    std::cout << "[Test] Running IVF Index & K-Means Clustering Test..." << std::endl;

    vectordb::EngineConfig config;
    config.dimension = 32;
    config.metric = vectordb::MetricType::L2;
    config.ivf_centroids = 8;
    config.ivf_nprobe = 4;

    vectordb::IVFIndex ivf(config);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<vectordb::Vector> train_data;
    for (size_t i = 0; i < 200; ++i) {
        vectordb::Vector v(config.dimension);
        for (size_t d = 0; d < config.dimension; ++d) v[d] = dist(rng);
        train_data.push_back(v);
    }

    ivf.train(train_data, 10);
    assert(ivf.num_centroids() == 8);
    std::cout << "K-Means successfully trained " << ivf.num_centroids() << " centroids" << std::endl;

    for (size_t i = 0; i < train_data.size(); ++i) {
        ivf.add(i + 1, train_data[i]);
    }
    assert(ivf.total_vectors() == 200);

    // Search for vector #5
    auto results = ivf.search(train_data[4], 5, 4);
    assert(!results.empty());
    std::cout << "Top IVF match ID: " << results[0].id << " (Expected: 5), Distance: " << results[0].distance << std::endl;
    assert(results[0].id == 5);
    assert(results[0].distance < 1e-4);

    std::cout << ">>> IVF Index Test PASSED! <<<" << std::endl;
    return 0;
}
