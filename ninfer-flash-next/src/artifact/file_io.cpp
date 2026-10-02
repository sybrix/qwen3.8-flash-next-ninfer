#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::artifact {
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}

MappedRegion::~MappedRegion() {
    if (mapping_ != nullptr) { ::munmap(mapping_, length_); }
}

MappedRegion::MappedRegion(MappedRegion&& other) noexcept
    : mapping_(std::exchange(other.mapping_, nullptr)), length_(std::exchange(other.length_, 0)),
      data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

MappedRegion& MappedRegion::operator=(MappedRegion&& other) noexcept {
    if (this != &other) {
        if (mapping_ != nullptr) { ::munmap(mapping_, length_); }
        mapping_ = std::exchange(other.mapping_, nullptr);
        length_  = std::exchange(other.length_, 0);
        data_    = std::exchange(other.data_, nullptr);
        bytes_   = std::exchange(other.bytes_, 0);
    }
    return *this;
}

MappedRegion InputFile::map(std::uint64_t offset, std::uint64_t bytes) const {
    if (!bytes || offset > bytes_ || bytes > bytes_ - offset ||
        bytes > std::numeric_limits<std::size_t>::max() / 2) {
        throw ArtifactError(path_.string() + ": mapping exceeds file length");
    }
    const auto page  = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
    const auto begin = offset / page * page;
    const auto length = static_cast<std::size_t>(offset - begin + bytes);
    void* mapping     = ::mmap(nullptr, length, PROT_READ, MAP_SHARED, fd_, file_offset(begin));
    if (mapping == MAP_FAILED) { fail(path_, "mmap"); }
    (void)::madvise(mapping, length, MADV_RANDOM);
    return MappedRegion(mapping, length,
                        static_cast<const std::byte*>(mapping) + (offset - begin), bytes);
}

} // namespace ninfer::artifact
