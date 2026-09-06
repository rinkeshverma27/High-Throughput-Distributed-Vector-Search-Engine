#include "vectordb/storage/wal.hpp"
#include <filesystem>
#include <iostream>
#include <cstring>

namespace vectordb {

static const uint32_t crc32_table[16] = {
    0x00000000, 0x1db71064, 0x3b6e20c8, 0x26d930ac,
    0x76dc4190, 0x6b6b51f4, 0x4db26158, 0x5005713c,
    0xedb88320, 0xf00f9344, 0xd6d6a3e8, 0xcb61b38c,
    0x9b64c2b0, 0x86d3d2d4, 0xa00ae278, 0xbdbdf21c
};

uint32_t WriteAheadLog::compute_crc32(const uint8_t* data, size_t len) {
    uint32_t crc = ~0u;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        crc = (crc >> 4) ^ crc32_table[crc & 0x0f];
        crc = (crc >> 4) ^ crc32_table[crc & 0x0f];
    }
    return ~crc;
}

WriteAheadLog::WriteAheadLog(const std::string& path) : path_(path) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    file_.open(path, std::ios::binary | std::ios::app);
}

WriteAheadLog::~WriteAheadLog() {
    flush();
    if (file_.is_open()) file_.close();
}

void WriteAheadLog::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) file_.flush();
}

bool WriteAheadLog::append_insert(VectorID id, const Vector& vector) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!file_.is_open()) return false;

    uint32_t payload_size = static_cast<uint32_t>(sizeof(VectorID) + vector.size() * sizeof(float));
    std::vector<uint8_t> payload(payload_size);

    std::memcpy(payload.data(), &id, sizeof(VectorID));
    std::memcpy(payload.data() + sizeof(VectorID), vector.data(), vector.size() * sizeof(float));

    WALHeader header;
    header.type = static_cast<uint8_t>(WALRecordType::INSERT);
    header.payload_size = payload_size;
    header.crc32 = compute_crc32(payload.data(), payload.size());

    file_.write(reinterpret_cast<const char*>(&header), sizeof(WALHeader));
    file_.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    file_.flush();

    return file_.good();
}

bool WriteAheadLog::append_delete(VectorID id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!file_.is_open()) return false;

    uint32_t payload_size = static_cast<uint32_t>(sizeof(VectorID));
    uint8_t payload[sizeof(VectorID)];
    std::memcpy(payload, &id, sizeof(VectorID));

    WALHeader header;
    header.type = static_cast<uint8_t>(WALRecordType::DELETE);
    header.payload_size = payload_size;
    header.crc32 = compute_crc32(payload, sizeof(VectorID));

    file_.write(reinterpret_cast<const char*>(&header), sizeof(WALHeader));
    file_.write(reinterpret_cast<const char*>(payload), sizeof(VectorID));
    file_.flush();

    return file_.good();
}

std::vector<std::pair<VectorID, Vector>> WriteAheadLog::replay() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<VectorID, Vector>> records;

    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) return records;

    while (in.peek() != EOF) {
        WALHeader header;
        in.read(reinterpret_cast<char*>(&header), sizeof(WALHeader));
        if (!in.good() || header.magic != 0x56444257) break;

        std::vector<uint8_t> payload(header.payload_size);
        in.read(reinterpret_cast<char*>(payload.data()), header.payload_size);
        if (!in.good()) break;

        if (compute_crc32(payload.data(), payload.size()) != header.crc32) {
            std::cerr << "WAL CRC mismatch, truncated or corrupted log" << std::endl;
            break;
        }

        if (header.type == static_cast<uint8_t>(WALRecordType::INSERT)) {
            VectorID id;
            std::memcpy(&id, payload.data(), sizeof(VectorID));
            size_t floats = (header.payload_size - sizeof(VectorID)) / sizeof(float);
            Vector vec(floats);
            std::memcpy(vec.data(), payload.data() + sizeof(VectorID), floats * sizeof(float));
            records.emplace_back(id, std::move(vec));
        }
    }

    return records;
}

} // namespace vectordb
