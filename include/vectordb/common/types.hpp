#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace vectordb {

using VectorID = uint64_t;
using Vector = std::vector<float>;
using NodeID = uint32_t;

enum class MetricType : uint8_t {
    L2 = 0,
    COSINE = 1,
    DOT_PRODUCT = 2
};

struct SearchResult {
    VectorID id;
    float distance;

    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }

    bool operator>(const SearchResult& other) const {
        return distance > other.distance;
    }
};

using MetadataMap = std::unordered_map<std::string, std::string>;

struct Record {
    VectorID id;
    Vector vector;
    MetadataMap metadata;
};

} // namespace vectordb
