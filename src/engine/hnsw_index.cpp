#include <mutex>
#include "vectordb/engine/hnsw_index.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_set>

namespace vectordb {

HNSWIndex::HNSWIndex(const EngineConfig& config)
    : config_(config),
      mult_(1.0 / std::log(static_cast<double>(config.hnsw_m))),
      rng_(42) {
    if (config_.hnsw_m == 0) config_.hnsw_m = 16;
    if (config_.hnsw_ef_construction == 0) config_.hnsw_ef_construction = 200;
    if (config_.hnsw_ef_search == 0) config_.hnsw_ef_search = 100;
}

int HNSWIndex::assign_random_level() {
    double r = uniform_dist_(rng_);
    if (r == 0.0) r = 0.0000001;
    return static_cast<int>(-std::log(r) * mult_);
}

size_t HNSWIndex::size() const {
    std::shared_lock<std::shared_mutex> lock(global_mutex_);
    return nodes_.size();
}

float HNSWIndex::compute_distance(NodeID internal_id, const float* query_data) const {
    return DistanceKernel::compute(
        nodes_[internal_id].data.data(),
        query_data,
        config_.dimension,
        config_.metric
    );
}

float HNSWIndex::compute_distance_nodes(NodeID a, NodeID b) const {
    return DistanceKernel::compute(
        nodes_[a].data.data(),
        nodes_[b].data.data(),
        config_.dimension,
        config_.metric
    );
}

std::vector<std::pair<float, NodeID>> HNSWIndex::search_layer(
    const float* query_data,
    const std::vector<NodeID>& entry_points,
    size_t ef,
    int layer
) const {
    std::unordered_set<NodeID> visited;
    // min-heap for dynamic candidate set
    std::priority_queue<
        std::pair<float, NodeID>,
        std::vector<std::pair<float, NodeID>>,
        std::greater<>
    > candidates;

    // max-heap for found nearest elements (worse distance on top)
    std::priority_queue<std::pair<float, NodeID>> w_results;

    for (NodeID ep : entry_points) {
        float d = compute_distance(ep, query_data);
        visited.insert(ep);
        candidates.emplace(d, ep);
        w_results.emplace(d, ep);
    }

    while (!candidates.empty()) {
        auto [c_dist, c_node] = candidates.top();
        candidates.pop();

        if (c_dist > w_results.top().first) {
            break;
        }

        const auto& neighbors = nodes_[c_node].neighbors[layer];
        for (NodeID neighbor : neighbors) {
            if (visited.insert(neighbor).second) {
                float n_dist = compute_distance(neighbor, query_data);
                if (w_results.size() < ef || n_dist < w_results.top().first) {
                    candidates.emplace(n_dist, neighbor);
                    w_results.emplace(n_dist, neighbor);
                    if (w_results.size() > ef) {
                        w_results.pop();
                    }
                }
            }
        }
    }

    std::vector<std::pair<float, NodeID>> sorted_results;
    sorted_results.reserve(w_results.size());
    while (!w_results.empty()) {
        sorted_results.push_back(w_results.top());
        w_results.pop();
    }
    std::reverse(sorted_results.begin(), sorted_results.end());
    return sorted_results;
}

void HNSWIndex::select_neighbors_simple(
    NodeID candidate_id,
    std::vector<std::pair<float, NodeID>>& candidates,
    size_t max_m,
    int layer
) {
    std::sort(candidates.begin(), candidates.end());
    auto& neighbors = nodes_[candidate_id].neighbors[layer];
    neighbors.clear();
    for (size_t i = 0; i < candidates.size() && i < max_m; ++i) {
        if (candidates[i].second != candidate_id) {
            neighbors.push_back(candidates[i].second);
        }
    }
}

void HNSWIndex::insert(VectorID id, const Vector& vector) {
    std::unique_lock<std::shared_mutex> lock(global_mutex_);

    if (id_map_.find(id) != id_map_.end()) {
        return; // already inserted
    }

    int insert_level = assign_random_level();
    NodeID new_id = static_cast<NodeID>(nodes_.size());

    HNSWNode new_node;
    new_node.id = id;
    new_node.data = vector;
    new_node.level = insert_level;
    new_node.neighbors.resize(insert_level + 1);

    nodes_.push_back(std::move(new_node));
    id_map_[id] = new_id;

    if (!has_enter_node_) {
        enter_node_ = new_id;
        max_level_ = insert_level;
        has_enter_node_ = true;
        return;
    }

    NodeID curr_obj = enter_node_;
    float curr_dist = compute_distance(curr_obj, vector.data());

    // 1. Traverse top layers greedily down to insert_level + 1
    for (int l = max_level_; l > insert_level; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (NodeID neighbor : nodes_[curr_obj].neighbors[l]) {
                float d = compute_distance(neighbor, vector.data());
                if (d < curr_dist) {
                    curr_dist = d;
                    curr_obj = neighbor;
                    changed = true;
                }
            }
        }
    }

    // 2. Connect from min(max_level, insert_level) down to layer 0
    std::vector<NodeID> enter_points = {curr_obj};
    for (int l = std::min(max_level_, insert_level); l >= 0; --l) {
        size_t max_m = (l == 0) ? (2 * config_.hnsw_m) : config_.hnsw_m;
        auto candidates = search_layer(vector.data(), enter_points, config_.hnsw_ef_construction, l);

        select_neighbors_simple(new_id, candidates, max_m, l);

        // Add bidirectional connections and shrink overflow
        for (NodeID neighbor : nodes_[new_id].neighbors[l]) {
            auto& n_neighbors = nodes_[neighbor].neighbors[l];
            n_neighbors.push_back(new_id);
            if (n_neighbors.size() > max_m) {
                std::vector<std::pair<float, NodeID>> n_candidates;
                for (NodeID nn : n_neighbors) {
                    n_candidates.emplace_back(compute_distance_nodes(neighbor, nn), nn);
                }
                select_neighbors_simple(neighbor, n_candidates, max_m, l);
            }
        }

        enter_points.clear();
        for (const auto& c : candidates) {
            enter_points.push_back(c.second);
        }
    }

    if (insert_level > max_level_) {
        max_level_ = insert_level;
        enter_node_ = new_id;
    }
}

void HNSWIndex::insert_batch(const std::vector<std::pair<VectorID, Vector>>& batch) {
    for (const auto& item : batch) {
        insert(item.first, item.second);
    }
}

std::vector<SearchResult> HNSWIndex::search(
    const Vector& query,
    size_t k,
    size_t ef_search
) const {
    std::shared_lock<std::shared_mutex> lock(global_mutex_);
    if (!has_enter_node_ || nodes_.empty()) {
        return {};
    }

    if (ef_search == 0) {
        ef_search = std::max(k, config_.hnsw_ef_search);
    }

    NodeID curr_obj = enter_node_;
    float curr_dist = compute_distance(curr_obj, query.data());

    // Greedy search through top layers down to layer 1
    for (int l = max_level_; l > 0; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (NodeID neighbor : nodes_[curr_obj].neighbors[l]) {
                float d = compute_distance(neighbor, query.data());
                if (d < curr_dist) {
                    curr_dist = d;
                    curr_obj = neighbor;
                    changed = true;
                }
            }
        }
    }

    // Search layer 0 with ef_search
    auto candidates = search_layer(query.data(), {curr_obj}, ef_search, 0);

    std::vector<SearchResult> results;
    results.reserve(std::min(k, candidates.size()));
    for (size_t i = 0; i < candidates.size() && i < k; ++i) {
        results.push_back({
            nodes_[candidates[i].second].id,
            candidates[i].first
        });
    }

    return results;
}

} // namespace vectordb
