#pragma once

#include "vectordb/common/types.hpp"
#include "vectordb/common/config.hpp"
#include "vectordb/engine/distance.hpp"
#include <vector>
#include <shared_mutex>
#include <random>

namespace vectordb {

class IVFIndex {
public:
    explicit IVFIndex(const EngineConfig& config);

    // Trains centroids using Lloyd's K-Means algorithm
    void train(const std::vector<Vector>& training_data, size_t max_iterations = 15);

    // Inserts a vector into the closest centroid's inverted list
    void add(VectorID id, const Vector& vector);

    // Searches the top-k nearest neighbors across nprobe centroids
    std::vector<SearchResult> search(const Vector& query, size_t k, size_t nprobe) const;

    size_t num_centroids() const { return centroids_.size(); }
    size_t total_vectors() const;

private:
    size_t find_closest_centroid(const float* vec) const;

    EngineConfig config_;
    std::vector<Vector> centroids_;
    std::vector<std::vector<std::pair<VectorID, Vector>>> inverted_lists_;
    mutable std::shared_mutex mutex_;
    bool is_trained_{false};
};

} // namespace vectordb
