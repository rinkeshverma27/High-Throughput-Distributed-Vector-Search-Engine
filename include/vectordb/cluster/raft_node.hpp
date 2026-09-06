#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>

namespace vectordb {

enum class RaftRole {
    FOLLOWER,
    CANDIDATE,
    LEADER
};

struct LogEntry {
    uint64_t term;
    uint64_t index;
    std::string command;
};

class RaftNode {
public:
    explicit RaftNode(uint32_t node_id, const std::vector<std::string>& peers)
        : node_id_(node_id), peers_(peers), current_term_(0), voted_for_(0), role_(RaftRole::FOLLOWER) {}

    RaftRole get_role() const { return role_; }
    uint64_t get_term() const { return current_term_; }
    bool is_leader() const { return role_ == RaftRole::LEADER; }

    bool append_entry(const std::string& command) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (role_ != RaftRole::LEADER) return false;
        log_.push_back({current_term_, log_.size() + 1, command});
        return true;
    }

private:
    uint32_t node_id_;
    std::vector<std::string> peers_;
    uint64_t current_term_;
    uint32_t voted_for_;
    std::atomic<RaftRole> role_;
    std::vector<LogEntry> log_;
    mutable std::mutex mutex_;
};

} // namespace vectordb
