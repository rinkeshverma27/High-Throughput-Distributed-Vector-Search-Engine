#include "vectordb/engine/hnsw_index.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

int main() {
    std::cout << "[Test] Running HNSW Index Insertion and Search Test..." << std::endl;

    vectordb::EngineConfig config;
    config.dimension = 64;
    config.metric = vectordb::MetricType::L2;
    config.hnsw_m = 16;
    config.hnsw_ef_construction = 100;
    config.hnsw_ef_search = 50;

    vectordb::HNSWIndex index(config);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const size_t num_vectors = 1000;
    std::vector<vectordb::Vector> raw_vectors;
    raw_vectors.reserve(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i) {
        vectordb::Vector v(config.dimension);
        for (size_t d = 0; d < config.dimension; ++d) {
            v[d] = dist(rng);
        }
        index.insert(i + 1, v);
        raw_vectors.push_back(v);
    }

    assert(index.size() == num_vectors);
    std::cout << "Inserted " << index.size() << " vectors. Graph max level: " << index.max_level() << std::endl;

    // Search for the exact 10th vector (self-query should return ID 10 as top-1 with 0 distance)
    auto results = index.search(raw_vectors[9], 5);
    assert(!results.empty());
    std::cout << "Top result ID: " << results[0].id << " (Expected: 10), Distance: " << results[0].distance << std::endl;
    assert(results[0].id == 10);
    assert(results[0].distance < 1e-5);

    std::cout << ">>> HNSW Index Test PASSED! <<<" << std::endl;
    return 0;
}
