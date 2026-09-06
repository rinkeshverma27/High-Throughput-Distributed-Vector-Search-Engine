#pragma once

#include <cstdint>

namespace vectordb {

enum class CommandType : uint8_t {
    INSERT = 0x01,
    SEARCH = 0x02,
    PING   = 0x03,
    PONG   = 0x04,
    STATS  = 0x05
};

enum class ResponseStatus : uint8_t {
    OK = 0x00,
    ERROR = 0x01
};

#pragma pack(push, 1)
struct MessageHeader {
    uint8_t command;
    uint32_t payload_size;
};
#pragma pack(pop)

} // namespace vectordb
