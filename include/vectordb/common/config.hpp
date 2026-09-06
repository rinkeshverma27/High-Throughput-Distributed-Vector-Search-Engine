#pragma once

#include "vectordb/common/types.hpp"
#include <string>
#include <cstdint>

namespace vectordb {

struct EngineConfig {
    size_t dimension{768};
    MetricType metric{MetricType::L2};
    size_t hnsw_m{16};
    size_t hnsw_ef_construction{200};
    size_t hnsw_ef_search{100};
    size_t ivf_centroids{256};
    size_t ivf_nprobe{16};
};

struct StorageConfig {
    std::string data_dir{"./data"};
    std::string wal_path{"./data/wal.log"};
    size_t segment_max_bytes{512 * 1024 * 1024}; // 512MB
    bool enable_mmap{true};
    bool enable_compaction{true};
};

struct ClusterConfig {
    uint32_t node_id{1};
    uint16_t port{9000};
    uint16_t raft_port{9001};
    size_t replica_factor{3};
    size_t num_shards{8};
    size_t vnodes_per_shard{150};
};

struct ServerConfig {
    EngineConfig engine;
    StorageConfig storage;
    ClusterConfig cluster;
};

} // namespace vectordb
