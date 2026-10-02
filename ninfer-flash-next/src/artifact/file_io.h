#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace ninfer::artifact {

// Read-only shared mapping of a byte range. Pages come from the page cache on demand, so a
// mapping larger than free RAM stays evictable; the OS is advised that access is random.
class MappedRegion {
public:
    MappedRegion() = default;
    MappedRegion(void* mapping, std::size_t length, const std::byte* data, std::uint64_t bytes)
        : mapping_(mapping), length_(length), data_(data), bytes_(bytes) {}
    ~MappedRegion();
    MappedRegion(MappedRegion&& other) noexcept;
    MappedRegion& operator=(MappedRegion&& other) noexcept;
    MappedRegion(const MappedRegion&)            = delete;
    MappedRegion& operator=(const MappedRegion&) = delete;

    [[nodiscard]] const std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

private:
    void* mapping_         = nullptr;
    std::size_t length_    = 0;
    const std::byte* data_ = nullptr;
    std::uint64_t bytes_   = 0;
};

// Direct reads require aligned offsets and buffers. A short final direct block is allowed;
// read_exact always requires the complete requested byte range.
class InputFile {
public:
    explicit InputFile(std::filesystem::path path);
    ~InputFile();
    InputFile(const InputFile&)            = delete;
    InputFile& operator=(const InputFile&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset,
                                          std::span<std::byte> destination) const;
    [[nodiscard]] MappedRegion map(std::uint64_t offset, std::uint64_t bytes) const;

private:
    std::filesystem::path path_;
    int fd_                = -1;
    mutable int direct_fd_ = -1;
    std::uint64_t bytes_   = 0;
};

} // namespace ninfer::artifact
