#include <mutex>
#include "vectordb/engine/ivf_index.hpp"
#include <limits>
#include <algorithm>
#include <queue>

namespace vectordb {

IVFIndex::IVFIndex(const EngineConfig& config)
    : config_(config) {
    if (config_.ivf_centroids == 0) config_.ivf_centroids = 16;
    if (config_.ivf_nprobe == 0) config_.ivf_nprobe = 4;
}

size_t IVFIndex::find_closest_centroid(const float* vec) const {
    size_t best_idx = 0;
    float best_dist = std::numeric_limits<float>::max();

    for (size_t i = 0; i < centroids_.size(); ++i) {
        float d = DistanceKernel::compute(centroids_[i].data(), vec, config_.dimension, config_.metric);
        if (d < best_dist) {
            best_dist = d;
            best_idx = i;
        }
    }
    return best_idx;
}

void IVFIndex::train(const std::vector<Vector>& training_data, size_t max_iterations) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (training_data.empty()) return;

    size_t k = std::min(config_.ivf_centroids, training_data.size());
    centroids_.resize(k);
    inverted_lists_.resize(k);

    // 1. Initialize centroids randomly
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, training_data.size() - 1);
    for (size_t i = 0; i < k; ++i) {
        centroids_[i] = training_data[dist(rng)];
    }

    // 2. Lloyd's K-Means Iterations
    std::vector<Vector> new_centroids(k, Vector(config_.dimension, 0.0f));
    std::vector<size_t> counts(k, 0);

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        for (size_t i = 0; i < k; ++i) {
            std::fill(new_centroids[i].begin(), new_centroids[i].end(), 0.0f);
            counts[i] = 0;
        }

        for (const auto& vec : training_data) {
            size_t c_idx = find_closest_centroid(vec.data());
            counts[c_idx]++;
            for (size_t d = 0; d < config_.dimension; ++d) {
                new_centroids[c_idx][d] += vec[d];
            }
        }

        for (size_t i = 0; i < k; ++i) {
            if (counts[i] > 0) {
                for (size_t d = 0; d < config_.dimension; ++d) {
                    centroids_[i][d] = new_centroids[i][d] / counts[i];
                }
            }
        }
    }

    is_trained_ = true;
}

void IVFIndex::add(VectorID id, const Vector& vector) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!is_trained_ || centroids_.empty()) {
        if (centroids_.empty()) {
            centroids_.push_back(vector);
            inverted_lists_.resize(1);
            is_trained_ = true;
        }
    }

    size_t centroid_idx = find_closest_centroid(vector.data());
    inverted_lists_[centroid_idx].emplace_back(id, vector);
}

std::vector<SearchResult> IVFIndex::search(const Vector& query, size_t k, size_t nprobe) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!is_trained_ || centroids_.empty()) return {};

    nprobe = std::min(nprobe, centroids_.size());

    // Find top nprobe closest centroids
    std::vector<std::pair<float, size_t>> centroid_distances;
    centroid_distances.reserve(centroids_.size());
    for (size_t i = 0; i < centroids_.size(); ++i) {
        float d = DistanceKernel::compute(centroids_[i].data(), query.data(), config_.dimension, config_.metric);
        centroid_distances.emplace_back(d, i);
    }
    std::sort(centroid_distances.begin(), centroid_distances.end());

    // Min-heap or sorted candidates from visited inverted lists
    std::vector<SearchResult> candidates;
    for (size_t i = 0; i < nprobe; ++i) {
        size_t c_idx = centroid_distances[i].second;
        for (const auto& [vec_id, vec_data] : inverted_lists_[c_idx]) {
            float d = DistanceKernel::compute(vec_data.data(), query.data(), config_.dimension, config_.metric);
            candidates.push_back({vec_id, d});
        }
    }

    std::sort(candidates.begin(), candidates.end());
    if (candidates.size() > k) {
        candidates.resize(k);
    }
    return candidates;
}

size_t IVFIndex::total_vectors() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& list : inverted_lists_) {
        count += list.size();
    }
    return count;
}

} // namespace vectordb
