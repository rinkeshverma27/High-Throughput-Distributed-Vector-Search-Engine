#pragma once

#include "vectordb/engine/hnsw_index.hpp"
#include "vectordb/storage/wal.hpp"
#include "vectordb/common/thread_pool.hpp"
#include "vectordb/common/config.hpp"
#include "vectordb/server/protocol.hpp"

#include <atomic>
#include <memory>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <thread>

namespace vectordb {

class TCPServer {
public:
    TCPServer(const ServerConfig& config, std::shared_ptr<HNSWIndex> index, std::shared_ptr<WriteAheadLog> wal)
        : config_(config), index_(index), wal_(wal), running_(false), server_fd_(-1), pool_(8) {}

    ~TCPServer() { stop(); }

    void start() {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            std::cerr << "Failed to create socket" << std::endl;
            return;
        }

        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(config_.cluster.port);

        if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
            std::cerr << "Failed to bind to port " << config_.cluster.port << std::endl;
            return;
        }

        if (listen(server_fd_, 128) < 0) {
            std::cerr << "Failed to listen" << std::endl;
            return;
        }

        running_ = true;
        std::cout << "[VectorDB Server] Listening on TCP port " << config_.cluster.port << " (C++20)" << std::endl;

        accept_thread_ = std::thread([this]() {
            while (running_) {
                sockaddr_in client_addr{};
                socklen_t client_len = sizeof(client_addr);
                int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
                if (client_fd < 0) {
                    if (!running_) break;
                    continue;
                }
                pool_.enqueue([this, client_fd]() {
                    handle_client(client_fd);
                });
            }
        });
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        if (server_fd_ >= 0) {
            shutdown(server_fd_, SHUT_RDWR);
            close(server_fd_);
            server_fd_ = -1;
        }
        if (accept_thread_.joinable()) {
            accept_thread_.join();
        }
    }

    bool is_running() const { return running_; }

private:
    static bool send_all(int fd, const void* data, size_t size) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data);
        size_t total = 0;
        while (total < size) {
            ssize_t sent = write(fd, ptr + total, size - total);
            if (sent <= 0) return false;
            total += sent;
        }
        return true;
    }

    void handle_client(int client_fd) {
        while (running_) {
            MessageHeader header;
            ssize_t n = read(client_fd, &header, sizeof(MessageHeader));
            if (n <= 0) break;

            std::vector<uint8_t> payload(header.payload_size);
            size_t bytes_read = 0;
            while (bytes_read < header.payload_size) {
                ssize_t r = read(client_fd, payload.data() + bytes_read, header.payload_size - bytes_read);
                if (r <= 0) break;
                bytes_read += r;
            }
            if (bytes_read < header.payload_size) break;

            if (header.command == static_cast<uint8_t>(CommandType::PING)) {
                uint8_t pong = static_cast<uint8_t>(CommandType::PONG);
                send_all(client_fd, &pong, 1);
            }
            else if (header.command == static_cast<uint8_t>(CommandType::STATS)) {
                uint64_t count = index_->size();
                send_all(client_fd, &count, sizeof(count));
            }
            else if (header.command == static_cast<uint8_t>(CommandType::INSERT)) {
                if (payload.size() >= sizeof(VectorID) + sizeof(uint32_t)) {
                    VectorID id;
                    uint32_t dim;
                    std::memcpy(&id, payload.data(), sizeof(VectorID));
                    std::memcpy(&dim, payload.data() + sizeof(VectorID), sizeof(uint32_t));

                    Vector vec(dim);
                    std::memcpy(vec.data(), payload.data() + sizeof(VectorID) + sizeof(uint32_t), dim * sizeof(float));

                    wal_->append_insert(id, vec);
                    index_->insert(id, vec);

                    uint8_t ok = static_cast<uint8_t>(ResponseStatus::OK);
                    send_all(client_fd, &ok, 1);
                }
            }
            else if (header.command == static_cast<uint8_t>(CommandType::SEARCH)) {
                if (payload.size() >= sizeof(uint32_t) * 2) {
                    uint32_t k;
                    uint32_t dim;
                    std::memcpy(&k, payload.data(), sizeof(uint32_t));
                    std::memcpy(&dim, payload.data() + sizeof(uint32_t), sizeof(uint32_t));

                    Vector query(dim);
                    std::memcpy(query.data(), payload.data() + sizeof(uint32_t) * 2, dim * sizeof(float));

                    auto results = index_->search(query, k);
                    uint32_t num_res = static_cast<uint32_t>(results.size());

                    send_all(client_fd, &num_res, sizeof(uint32_t));
                    for (const auto& r : results) {
                        send_all(client_fd, &r.id, sizeof(VectorID));
                        send_all(client_fd, &r.distance, sizeof(float));
                    }
                }
            }
        }
        close(client_fd);
    }

    ServerConfig config_;
    std::shared_ptr<HNSWIndex> index_;
    std::shared_ptr<WriteAheadLog> wal_;
    std::atomic<bool> running_;
    int server_fd_;
    ThreadPool pool_;
    std::thread accept_thread_;
};

} // namespace vectordb
