#pragma once

#include "vectordb/common/types.hpp"
#include "vectordb/common/config.hpp"
#include "vectordb/engine/distance.hpp"

#include <vector>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <queue>

namespace vectordb {

class HNSWIndex {
public:
    explicit HNSWIndex(const EngineConfig& config);
    ~HNSWIndex() = default;

    // Core vector operations
    void insert(VectorID id, const Vector& vector);
    void insert_batch(const std::vector<std::pair<VectorID, Vector>>& batch);
    std::vector<SearchResult> search(const Vector& query, size_t k, size_t ef_search = 0) const;

    size_t size() const;
    size_t dimension() const { return config_.dimension; }
    int max_level() const { return max_level_; }

private:
    struct HNSWNode {
        VectorID id;
        Vector data;
        int level;
        std::vector<std::vector<NodeID>> neighbors; // neighbors[layer]
    };

    int assign_random_level();
    float compute_distance(NodeID internal_id, const float* query_data) const;
    float compute_distance_nodes(NodeID a, NodeID b) const;

    std::vector<std::pair<float, NodeID>> search_layer(
        const float* query_data,
        const std::vector<NodeID>& entry_points,
        size_t ef,
        int layer
    ) const;

    void select_neighbors_simple(
        NodeID candidate_id,
        std::vector<std::pair<float, NodeID>>& candidates,
        size_t max_m,
        int layer
    );

    EngineConfig config_;
    double mult_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};

    mutable std::shared_mutex global_mutex_;
    std::vector<HNSWNode> nodes_;
    std::unordered_map<VectorID, NodeID> id_map_;
    int max_level_{-1};
    NodeID enter_node_{0};
    bool has_enter_node_{false};
};

} // namespace vectordb
