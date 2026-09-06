#pragma once

#include "vectordb/common/types.hpp"
#include <string>
#include <fstream>
#include <mutex>
#include <vector>

namespace vectordb {

enum class WALRecordType : uint8_t {
    INSERT = 1,
    DELETE = 2,
    CHECKPOINT = 3
};

#pragma pack(push, 1)
struct WALHeader {
    uint32_t magic{0x56444257}; // 'VDBW' (VectorDB WAL)
    uint8_t  type;
    uint32_t payload_size;
    uint32_t crc32;
};
#pragma pack(pop)

class WriteAheadLog {
public:
    explicit WriteAheadLog(const std::string& path);
    ~WriteAheadLog();

    bool append_insert(VectorID id, const Vector& vector);
    bool append_delete(VectorID id);
    void flush();

    // Replay on startup to recover uncommitted records
    std::vector<std::pair<VectorID, Vector>> replay();

    static uint32_t compute_crc32(const uint8_t* data, size_t len);

private:
    std::string path_;
    std::ofstream file_;
    std::mutex mutex_;
};

} // namespace vectordb
