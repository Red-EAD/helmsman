#include "util/file/files_rw.hpp"

namespace minihypervec
{
  namespace util
  {
    namespace
    {

      static int32_t write_all(int fd, const char *buf, size_t len)
      {
        size_t off = 0;
        while (off < len)
        {
          ssize_t n = ::write(fd, buf + off, len - off);
          if (n < 0)
          {
            if (errno == EINTR)
              continue;
            return -errno;
          }
          off += static_cast<size_t>(n);
        }
        return 0;
      }

      static int32_t fsync_dir_of_path(const std::string &path)
      {
        std::filesystem::path p(path);
        std::filesystem::path dir =
            p.has_parent_path() ? p.parent_path() : std::filesystem::path(".");
        int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
        if (dfd < 0)
          return -errno;

        if (::fsync(dfd) != 0)
        {
          int e = errno;
          ::close(dfd);
          return -e;
        }
        ::close(dfd);
        return 0;
      }
    } // namespace

    int32_t persist_string_atomic_fsync(const std::string &path,
                                        const std::string &data)
    {
      std::string tmp = path + ".tmp";
      // 0644 可按需要调整权限
      int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd < 0)
        return -errno;

      int32_t rc = write_all(fd, data.data(), data.size());
      if (rc != 0)
      {
        ::close(fd);
        (void)::unlink(tmp.c_str());
        return rc;
      }
      if (::fsync(fd) != 0)
      {
        int e = errno;
        ::close(fd);
        (void)::unlink(tmp.c_str());
        return -e;
      }
      if (::close(fd) != 0)
      {
        int e = errno;
        (void)::unlink(tmp.c_str());
        return -e;
      }
      if (::rename(tmp.c_str(), path.c_str()) != 0)
      {
        int e = errno;
        (void)::unlink(tmp.c_str());
        return -e;
      }
      return fsync_dir_of_path(path);
    }

    int32_t read_file_to_string(const std::string &path, std::string &out)
    {
      int fd = ::open(path.c_str(), O_RDONLY);
      if (fd < 0)
        return -errno;

      std::string buf;
      buf.reserve(4096);
      constexpr size_t kChunk = 1 << 16;
      std::string chunk;
      chunk.resize(kChunk);

      while (true)
      {
        ssize_t n = ::read(fd, chunk.data(), kChunk);
        if (n < 0)
        {
          if (errno == EINTR)
            continue;
          int e = errno;
          ::close(fd);
          return -e;
        }
        if (n == 0)
          break;
        buf.append(chunk.data(), static_cast<size_t>(n));
      }

      if (::close(fd) != 0)
        return -errno;

      out.swap(buf);
      return 0;
    }
  } // namespace util
} // namespace minihypervec