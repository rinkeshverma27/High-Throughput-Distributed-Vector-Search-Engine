#pragma once

#include "vectordb/common/types.hpp"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>

namespace vectordb {

class MetadataFilter {
public:
    void index_metadata(VectorID id, const MetadataMap& metadata) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        for (const auto& [k, v] : metadata) {
            index_[k + "=" + v].insert(id);
        }
    }

    bool matches(VectorID id, const std::string& key, const std::string& value) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = index_.find(key + "=" + value);
        if (it == index_.end()) return false;
        return it->second.count(id) > 0;
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, std::unordered_set<VectorID>> index_;
};

} // namespace vectordb
