#include "vectordb/cluster/consistent_hash.hpp"
#include <sstream>
#include <algorithm>
#include <unordered_set>

namespace vectordb {

// Fast FNV-1a 64-bit hash
uint64_t ConsistentHashRing::hash_key(const std::string& key) {
    uint64_t hash = 14695981039346656037ULL;
    for (char c : key) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t ConsistentHashRing::hash_vector_id(VectorID id) {
    uint64_t x = id;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

ConsistentHashRing::ConsistentHashRing(size_t vnodes_per_node)
    : vnodes_(vnodes_per_node) {}

void ConsistentHashRing::add_node(const std::string& node_address) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < vnodes_; ++i) {
        std::string vnode_key = node_address + "#" + std::to_string(i);
        uint64_t token = hash_key(vnode_key);
        ring_[token] = node_address;
    }
}

void ConsistentHashRing::remove_node(const std::string& node_address) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < vnodes_; ++i) {
        std::string vnode_key = node_address + "#" + std::to_string(i);
        uint64_t token = hash_key(vnode_key);
        ring_.erase(token);
    }
}

std::string ConsistentHashRing::get_primary_node(VectorID id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ring_.empty()) return "";

    uint64_t token = hash_vector_id(id);
    auto it = ring_.lower_bound(token);
    if (it == ring_.end()) {
        it = ring_.begin();
    }
    return it->second;
}

std::vector<std::string> ConsistentHashRing::get_replica_nodes(
    VectorID id,
    size_t replication_factor
) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ring_.empty()) return {};

    uint64_t token = hash_vector_id(id);
    auto it = ring_.lower_bound(token);
    if (it == ring_.end()) it = ring_.begin();

    std::vector<std::string> result;
    std::unordered_set<std::string> seen;

    auto start_it = it;
    do {
        if (seen.insert(it->second).second) {
            result.push_back(it->second);
            if (result.size() == replication_factor) break;
        }
        ++it;
        if (it == ring_.end()) it = ring_.begin();
    } while (it != start_it);

    return result;
}

size_t ConsistentHashRing::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ring_.size();
}

} // namespace vectordb
