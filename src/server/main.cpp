#include "vectordb/engine/hnsw_index.hpp"
#include "vectordb/storage/wal.hpp"
#include "vectordb/cluster/consistent_hash.hpp"
#include "vectordb/server/tcp_server.hpp"
#include "vectordb/common/config.hpp"

#include <iostream>
#include <csignal>
#include <chrono>

std::atomic<bool> g_stop{false};

void signal_handler(int) {
    g_stop = true;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "=======================================================\n";
    std::cout << " High-Throughput Distributed Vector Search Engine (C++20)\n";
    std::cout << "=======================================================\n";

    vectordb::ServerConfig config;
    config.engine.dimension = 128;
    config.engine.metric = vectordb::MetricType::L2;
    config.cluster.port = 9000;

    std::cout << "[Init] AVX2 Acceleration Available: " 
              << (vectordb::DistanceKernel::has_avx2() ? "YES (Hardware SIMD Active)" : "NO (Scalar Fallback)") 
              << std::endl;

    auto index = std::make_shared<vectordb::HNSWIndex>(config.engine);
    auto wal = std::make_shared<vectordb::WriteAheadLog>("./data/wal.log");

    // Replay WAL on startup
    auto replayed = wal->replay();
    if (!replayed.empty()) {
        std::cout << "[WAL] Recovered " << replayed.size() << " records from WAL" << std::endl;
        for (const auto& [id, vec] : replayed) {
            index->insert(id, vec);
        }
    }

    vectordb::ConsistentHashRing hash_ring(150);
    hash_ring.add_node("127.0.0.1:9000");
    hash_ring.add_node("127.0.0.1:9001");
    hash_ring.add_node("127.0.0.1:9002");
    std::cout << "[Cluster] Consistent Hash Ring initialized with " << hash_ring.size() << " virtual nodes" << std::endl;

    vectordb::TCPServer server(config, index, wal);
    server.start();

    std::cout << "[Ready] System initialized successfully. Press Ctrl+C to terminate." << std::endl;

    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "\n[Shutdown] Stopping server and flushing WAL..." << std::endl;
    server.stop();
    wal->flush();
    std::cout << "[Shutdown] Completed cleanly." << std::endl;
    return 0;
}
