#pragma once

#include "vectordb/common/types.hpp"
#include <map>
#include <string>
#include <vector>
#include <mutex>

namespace vectordb {

class ConsistentHashRing {
public:
    explicit ConsistentHashRing(size_t vnodes_per_node = 150);

    void add_node(const std::string& node_address);
    void remove_node(const std::string& node_address);
    std::string get_primary_node(VectorID id) const;
    std::vector<std::string> get_replica_nodes(VectorID id, size_t replication_factor) const;

    size_t size() const;

private:
    static uint64_t hash_key(const std::string& key);
    static uint64_t hash_vector_id(VectorID id);

    size_t vnodes_;
    mutable std::mutex mutex_;
    std::map<uint64_t, std::string> ring_; // token -> node address
};

} // namespace vectordb
