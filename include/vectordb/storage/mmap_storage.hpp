#pragma once

#include <cstddef>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>

namespace vectordb {

class MmapStorage {
public:
    MmapStorage(const std::string& path, size_t initial_size)
        : path_(path), size_(initial_size), fd_(-1), data_(nullptr) {
        fd_ = open(path.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) throw std::runtime_error("Failed to open file for mmap: " + path);

        if (ftruncate(fd_, size_) != 0) {
            close(fd_);
            throw std::runtime_error("Failed to set file size for mmap");
        }

        data_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("mmap failed");
        }
        madvise(data_, size_, MADV_RANDOM);
    }

    ~MmapStorage() {
        if (data_ && data_ != MAP_FAILED) {
            msync(data_, size_, MS_SYNC);
            munmap(data_, size_);
        }
        if (fd_ >= 0) close(fd_);
    }

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }

private:
    std::string path_;
    size_t size_;
    int fd_;
    void* data_;
};

} // namespace vectordb
