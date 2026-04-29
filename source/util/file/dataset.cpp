#include "util/file/dataset.hpp"

namespace minihypervec
{
  namespace util
  {
    int32_t loadFileToMemory(const std::string &file_path, void *data_addr,
                             uint64_t total_size)
    {
      std::ifstream inFile(file_path, std::ios::binary | std::ios::ate);
      if (!inFile)
      {
        std::cerr << "Error: Unable to open file for reading: " << file_path
                  << std::endl;
        return -1;
      }
      std::streamsize fileSize = inFile.tellg();
      if (fileSize != total_size)
      {
        std::cerr << "Error: File is empty or size not equal to total_size: "
                  << file_path << std::endl;
        std::cout << "file size: " << fileSize << " total size: " << total_size
                  << std::endl;
        return -1;
      }
      inFile.seekg(0, std::ios::beg);
      if (data_addr == nullptr)
      {
        std::cerr << "Error: Memory allocation failed" << std::endl;
        return -1;
      }
      if (!inFile.read(reinterpret_cast<char *>(data_addr), fileSize))
      {
        std::cerr << "Error: Failed to read data from file: " << file_path
                  << std::endl;
        return -1;
      }
      inFile.close();
      return 0;
    }

    int32_t saveMemoryToFile(const std::string &file_path, void *data_addr,
                             uint64_t total_size)
    {
      std::ofstream out_file(file_path, std::ios::binary | std::ios::out);
      if (!out_file.is_open())
      {
        return -1;
      }
      out_file.write(static_cast<char *>(data_addr), total_size);
      if (out_file.fail())
      {
        out_file.close();
        return -2;
      }
      out_file.close();
      return 0;
    }

    namespace
    {
      template <class T>
      static void load_le(const void *src, T &out)
      {
        std::memcpy(&out, src, sizeof(T));
      }
    }

    bool Dataset<int8_t>::mul_overflow(uint64_t a, uint64_t b, uint64_t *out)
    {
      if (a == 0 || b == 0)
      {
        *out = 0;
        return false;
      }
      if (a > std::numeric_limits<uint64_t>::max() / b)
        return true;
      *out = a * b;
      return false;
    }
    bool Dataset<int8_t>::add_overflow(uint64_t a, uint64_t b, uint64_t *out)
    {
      if (a > std::numeric_limits<uint64_t>::max() - b)
        return true;
      *out = a + b;
      return false;
    }

    bool Dataset<int8_t>::mapped_ok() const noexcept
    {
      return map_ != (void *)-1 && data_base_ != nullptr &&
             file_size_ >= 8;
    }

    void Dataset<int8_t>::move_from(Dataset &&o) noexcept
    {
      file_path = std::move(o.file_path);
      dim = o.dim;
      total_cnt = o.total_cnt;

      fd_ = o.fd_;
      o.fd_ = -1;
      map_ = o.map_;
      o.map_ = (void *)-1;
      file_size_ = o.file_size_;
      o.file_size_ = 0;
      data_base_ = o.data_base_;
      o.data_base_ = nullptr;
      data_region_bytes_ = o.data_region_bytes_;
      o.data_region_bytes_ = 0;
    }

    void Dataset<int8_t>::cleanup() noexcept
    {
      if (map_ != (void *)-1)
      {
        ::munmap(map_,
                 static_cast<size_t>(std::min<uint64_t>(file_size_, SIZE_MAX)));
        map_ = (void *)-1;
      }
      if (fd_ >= 0)
      {
        ::close(fd_);
        fd_ = -1;
      }
      data_base_ = nullptr;
      file_size_ = 0;
      data_region_bytes_ = 0;
      dim = 0;
      total_cnt = 0;
    }

    void Dataset<int8_t>::open_file_and_map(const std::string &path)
    {
      file_path = path;

      fd_ = ::open(path.c_str(), O_RDONLY);
      if (fd_ < 0)
      {
        throw std::system_error(errno, std::generic_category(),
                                "open failed: " + path);
      }

      struct stat st{};
      if (::fstat(fd_, &st) != 0)
      {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::system_error(e, std::generic_category(), "fstat failed");
      }
      if (st.st_size < 8)
      {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("file too small (< 8 header bytes)");
      }
      file_size_ = static_cast<uint64_t>(st.st_size);

      if (file_size_ > SIZE_MAX)
      {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("file too large to map on this platform");
      }

      map_ = ::mmap(nullptr, static_cast<size_t>(file_size_), PROT_READ,
                    MAP_PRIVATE, fd_, 0);
      madvise(map_, static_cast<size_t>(file_size_), MADV_WILLNEED);
      if (map_ == MAP_FAILED)
      {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::system_error(e, std::generic_category(), "mmap failed");
      }

      const std::byte *base = static_cast<const std::byte *>(map_);

      uint64_t num_points_qi = 0;
      uint32_t dim_qi = 0;
      bool qi_ok = false;
      if (file_size_ >= 12)
      {
        load_le(base + 0, num_points_qi);
        load_le(base + 8, dim_qi);

        uint64_t data_bytes = 0, expect_total = 0;
        if (!mul_overflow(num_points_qi, static_cast<uint64_t>(dim_qi),
                          &data_bytes) &&
            !add_overflow(12, data_bytes, &expect_total) &&
            expect_total == file_size_)
        {
          qi_ok = true;
          std::cout << "Detected header: <QI>" << std::endl;
        }
      }

      uint32_t num_points_ii = 0;
      uint32_t dim_ii = 0;
      bool ii_ok = false;
      if (!qi_ok && file_size_ >= 8)
      {
        load_le(base + 0, num_points_ii);
        load_le(base + 4, dim_ii);
        uint64_t data_bytes = 0, expect_total = 0;
        if (!mul_overflow(static_cast<uint64_t>(num_points_ii),
                          static_cast<uint64_t>(dim_ii), &data_bytes) &&
            !add_overflow(8, data_bytes, &expect_total) &&
            expect_total == file_size_)
        {
          ii_ok = true;
          std::cout << "Detected header: <II>" << std::endl;
        }
      }

      uint64_t header_size = 0;
      uint64_t num_points = 0;
      uint32_t dim_local = 0;

      if (qi_ok)
      {
        header_size = 12;
        num_points = num_points_qi;
        dim_local = dim_qi;
      }
      else if (ii_ok)
      {
        header_size = 8;
        num_points = num_points_ii;
        dim_local = dim_ii;
      }
      else
      {
        bool accepted = false;
        if (file_size_ >= 12)
        {
          load_le(base + 0, num_points_qi);
          load_le(base + 8, dim_qi);
          uint64_t data_bytes = 0, need = 0;
          if (!mul_overflow(num_points_qi, static_cast<uint64_t>(dim_qi),
                            &data_bytes) &&
              !add_overflow(12, data_bytes, &need) && file_size_ >= need &&
              dim_qi > 0)
          {
            header_size = 12;
            num_points = num_points_qi;
            dim_local = dim_qi;
            accepted = true;
          }
        }
        if (!accepted && file_size_ >= 8)
        {
          load_le(base + 0, num_points_ii);
          load_le(base + 4, dim_ii);
          uint64_t data_bytes = 0, need = 0;
          if (!mul_overflow(static_cast<uint64_t>(num_points_ii),
                            static_cast<uint64_t>(dim_ii), &data_bytes) &&
              !add_overflow(8, data_bytes, &need) && file_size_ >= need &&
              dim_ii > 0)
          {
            header_size = 8;
            num_points = num_points_ii;
            dim_local = dim_ii;
            accepted = true;
          }
        }
        if (!accepted)
        {
          cleanup();
          throw std::runtime_error(
              "Unrecognized header: neither <QI> nor <II> fits this file.");
        }
      }

      uint64_t data_bytes = 0, need_total = 0;
      if (mul_overflow(num_points, static_cast<uint64_t>(dim_local), &data_bytes) ||
          add_overflow(header_size, data_bytes, &need_total) ||
          need_total > file_size_)
      {
        cleanup();
        throw std::runtime_error("file truncated or header/dtype mismatch");
      }

      dim = dim_local;
      total_cnt = num_points;
      data_base_ = reinterpret_cast<const int8_t *>(base + header_size);
      data_region_bytes_ = data_bytes;

      (void)posix_madvise(
          const_cast<int8_t *>(data_base_),
          static_cast<size_t>(std::min<uint64_t>(data_region_bytes_, SIZE_MAX)),
          POSIX_MADV_RANDOM);
    }

    Dataset<int8_t>::Dataset(const std::string &path) { open_file_and_map(path); }
    Dataset<int8_t>::~Dataset() { cleanup(); }
    Dataset<int8_t>::Dataset(Dataset &&other) noexcept
    {
      move_from(std::move(other));
    }
    Dataset<int8_t> &Dataset<int8_t>::operator=(Dataset &&other) noexcept
    {
      if (this != &other)
      {
        cleanup();
        move_from(std::move(other));
      }
      return *this;
    }

    int32_t Dataset<int8_t>::getVecs(const std::vector<uint64_t> &ids,
                                     std::vector<int8_t> &out_vecs)
    {
      if (!mapped_ok() || dim == 0)
        return -1;

      uint64_t out_elems64 = 0;
      if (mul_overflow(static_cast<uint64_t>(ids.size()),
                       static_cast<uint64_t>(dim), &out_elems64))
        return -3;
      if (out_elems64 > SIZE_MAX)
        return -3;
      out_vecs.resize(static_cast<size_t>(out_elems64));

      const uint64_t row_bytes = static_cast<uint64_t>(dim);
      const int8_t *base = data_base_;

      for (size_t j = 0; j < ids.size(); ++j)
      {
        uint64_t id = ids[j];
        if (id >= total_cnt)
          return -2;

        uint64_t offset_bytes = 0;
        if (mul_overflow(id, row_bytes, &offset_bytes))
          return -3;

        if (offset_bytes > data_region_bytes_)
          return -3;
        if (row_bytes > data_region_bytes_ - offset_bytes)
          return -3;

        const int8_t *src = base + offset_bytes;
        int8_t *dst = out_vecs.data() + static_cast<size_t>(j) * dim;

        (void)posix_madvise(
            const_cast<int8_t *>(src),
            static_cast<size_t>(std::min<uint64_t>(row_bytes, SIZE_MAX)),
            POSIX_MADV_WILLNEED);

        std::memcpy(dst, src, static_cast<size_t>(row_bytes));
      }
      return 0;
    }

    bool Dataset<float>::mul_overflow(uint64_t a, uint64_t b, uint64_t *out)
    {
      if (a == 0 || b == 0)
      {
        *out = 0;
        return false;
      }
      if (a > std::numeric_limits<uint64_t>::max() / b)
        return true;
      *out = a * b;
      return false;
    }
    bool Dataset<float>::add_overflow(uint64_t a, uint64_t b, uint64_t *out)
    {
      if (a > std::numeric_limits<uint64_t>::max() - b)
        return true;
      *out = a + b;
      return false;
    }

    bool Dataset<float>::mapped_ok() const noexcept
    {
      return map_ != (void *)-1 && data_base_ != nullptr &&
             file_size_ >= 8;
    }

    void Dataset<float>::move_from(Dataset &&o) noexcept
    {
      file_path = std::move(o.file_path);
      dim = o.dim;
      total_cnt = o.total_cnt;

      fd_ = o.fd_;
      o.fd_ = -1;
      map_ = o.map_;
      o.map_ = (void *)-1;
      file_size_ = o.file_size_;
      o.file_size_ = 0;
      data_base_ = o.data_base_;
      o.data_base_ = nullptr;
      data_region_bytes_ = o.data_region_bytes_;
      o.data_region_bytes_ = 0;
    }

    void Dataset<float>::cleanup() noexcept
    {
      if (map_ != (void *)-1)
      {
        ::munmap(map_,
                 static_cast<size_t>(std::min<uint64_t>(file_size_, SIZE_MAX)));
        map_ = (void *)-1;
      }
      if (fd_ >= 0)
      {
        ::close(fd_);
        fd_ = -1;
      }
      data_base_ = nullptr;
      file_size_ = 0;
      data_region_bytes_ = 0;
      dim = 0;
      total_cnt = 0;
    }

    void Dataset<float>::open_file_and_map(const std::string &path)
    {
      file_path = path;

      fd_ = ::open(path.c_str(), O_RDONLY);
      if (fd_ < 0)
      {
        throw std::system_error(errno, std::generic_category(),
                                "open failed: " + path);
      }

      struct stat st{};
      if (::fstat(fd_, &st) != 0)
      {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::system_error(e, std::generic_category(), "fstat failed");
      }
      if (st.st_size < 8)
      {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("file too small (< 8 header bytes)");
      }
      file_size_ = static_cast<uint64_t>(st.st_size);

      if (file_size_ > SIZE_MAX)
      {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("file too large to map on this platform");
      }

      map_ = ::mmap(nullptr, static_cast<size_t>(file_size_), PROT_READ,
                    MAP_PRIVATE, fd_, 0);
      madvise(map_, static_cast<size_t>(file_size_), MADV_WILLNEED);
      if (map_ == MAP_FAILED)
      {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::system_error(e, std::generic_category(), "mmap failed");
      }

      const std::byte *base = static_cast<const std::byte *>(map_);
      constexpr uint64_t kElemSize = sizeof(float);

      uint64_t num_points_qi = 0;
      uint32_t dim_qi = 0;
      bool qi_ok = false;
      if (file_size_ >= 12)
      {
        load_le(base + 0, num_points_qi);
        load_le(base + 8, dim_qi);

        uint64_t data_elems = 0, data_bytes = 0, expect_total = 0;
        if (!mul_overflow(num_points_qi, static_cast<uint64_t>(dim_qi),
                          &data_elems) &&
            !mul_overflow(data_elems, kElemSize, &data_bytes) &&
            !add_overflow(12, data_bytes, &expect_total) &&
            expect_total == file_size_)
        {
          qi_ok = true;
          std::cout << "Dataset<float>: <QI> header detected\n";
        }
      }

      uint32_t num_points_ii = 0;
      uint32_t dim_ii = 0;
      bool ii_ok = false;
      if (!qi_ok && file_size_ >= 8)
      {
        load_le(base + 0, num_points_ii);
        load_le(base + 4, dim_ii);

        uint64_t data_elems = 0, data_bytes = 0, expect_total = 0;
        if (!mul_overflow(static_cast<uint64_t>(num_points_ii),
                          static_cast<uint64_t>(dim_ii), &data_elems) &&
            !mul_overflow(data_elems, kElemSize, &data_bytes) &&
            !add_overflow(8, data_bytes, &expect_total) &&
            expect_total == file_size_)
        {
          ii_ok = true;
          std::cout << "Dataset<float>: <II> header detected\n";
        }
      }

      uint64_t header_size = 0;
      uint64_t num_points = 0;
      uint32_t dim_local = 0;

      if (qi_ok)
      {
        header_size = 12;
        num_points = num_points_qi;
        dim_local = dim_qi;
      }
      else if (ii_ok)
      {
        header_size = 8;
        num_points = num_points_ii;
        dim_local = dim_ii;
      }
      else
      {
        bool accepted = false;
        if (file_size_ >= 12)
        {
          load_le(base + 0, num_points_qi);
          load_le(base + 8, dim_qi);
          uint64_t data_elems = 0, data_bytes = 0, need = 0;
          if (!mul_overflow(num_points_qi, static_cast<uint64_t>(dim_qi),
                            &data_elems) &&
              !mul_overflow(data_elems, kElemSize, &data_bytes) &&
              !add_overflow(12, data_bytes, &need) && file_size_ >= need &&
              dim_qi > 0)
          {
            header_size = 12;
            num_points = num_points_qi;
            dim_local = dim_qi;
            accepted = true;
          }
        }
        if (!accepted && file_size_ >= 8)
        {
          load_le(base + 0, num_points_ii);
          load_le(base + 4, dim_ii);
          uint64_t data_elems = 0, data_bytes = 0, need = 0;
          if (!mul_overflow(static_cast<uint64_t>(num_points_ii),
                            static_cast<uint64_t>(dim_ii), &data_elems) &&
              !mul_overflow(data_elems, kElemSize, &data_bytes) &&
              !add_overflow(8, data_bytes, &need) && file_size_ >= need &&
              dim_ii > 0)
          {
            header_size = 8;
            num_points = num_points_ii;
            dim_local = dim_ii;
            accepted = true;
          }
        }
        if (!accepted)
        {
          cleanup();
          throw std::runtime_error(
              "Unrecognized header: neither <QI> nor <II> fits this file.");
        }
      }

      uint64_t data_elems = 0, data_bytes = 0, need_total = 0;
      if (mul_overflow(num_points, static_cast<uint64_t>(dim_local), &data_elems) ||
          mul_overflow(data_elems, kElemSize, &data_bytes) ||
          add_overflow(header_size, data_bytes, &need_total) ||
          need_total > file_size_)
      {
        cleanup();
        throw std::runtime_error("file truncated or header/dtype mismatch");
      }

      dim = dim_local;
      total_cnt = num_points;
      data_base_ = reinterpret_cast<const float *>(base + header_size);
      data_region_bytes_ = data_bytes;

      (void)posix_madvise(
          const_cast<float *>(data_base_),
          static_cast<size_t>(std::min<uint64_t>(data_region_bytes_, SIZE_MAX)),
          POSIX_MADV_RANDOM);
    }

    Dataset<float>::Dataset(const std::string &path) { open_file_and_map(path); }
    Dataset<float>::~Dataset() { cleanup(); }
    Dataset<float>::Dataset(Dataset &&other) noexcept
    {
      move_from(std::move(other));
    }
    Dataset<float> &Dataset<float>::operator=(Dataset &&other) noexcept
    {
      if (this != &other)
      {
        cleanup();
        move_from(std::move(other));
      }
      return *this;
    }

    int32_t Dataset<float>::getVecs(const std::vector<uint64_t> &ids,
                                    std::vector<float> &out_vecs)
    {
      if (!mapped_ok() || dim == 0)
        return -1;

      uint64_t out_elems64 = 0;
      if (mul_overflow(static_cast<uint64_t>(ids.size()),
                       static_cast<uint64_t>(dim), &out_elems64))
        return -3;
      if (out_elems64 > SIZE_MAX)
        return -3;
      out_vecs.resize(static_cast<size_t>(out_elems64));

      const uint64_t row_bytes = static_cast<uint64_t>(dim) * sizeof(float);
      const float *base = data_base_;

      for (size_t j = 0; j < ids.size(); ++j)
      {
        uint64_t id = ids[j];
        if (id >= total_cnt)
          return -2;

        uint64_t offset_bytes = 0;
        if (mul_overflow(id, row_bytes, &offset_bytes))
          return -3;

        if (offset_bytes > data_region_bytes_)
          return -3;
        if (row_bytes > data_region_bytes_ - offset_bytes)
          return -3;

        const float *src = reinterpret_cast<const float *>(
            reinterpret_cast<const std::byte *>(base) + offset_bytes);
        float *dst = out_vecs.data() + static_cast<size_t>(j) * dim;

        (void)posix_madvise(
            const_cast<float *>(src),
            static_cast<size_t>(std::min<uint64_t>(row_bytes, SIZE_MAX)),
            POSIX_MADV_WILLNEED);

        std::memcpy(dst, src, static_cast<size_t>(row_bytes));
      }
      return 0;
    }

    bool Dataset<uint64_t>::mul_overflow(uint64_t a, uint64_t b, uint64_t *out)
    {
      if (a == 0 || b == 0)
      {
        *out = 0;
        return false;
      }
      if (a > std::numeric_limits<uint64_t>::max() / b)
        return true;
      *out = a * b;
      return false;
    }
    bool Dataset<uint64_t>::add_overflow(uint64_t a, uint64_t b, uint64_t *out)
    {
      if (a > std::numeric_limits<uint64_t>::max() - b)
        return true;
      *out = a + b;
      return false;
    }

    bool Dataset<uint64_t>::mapped_ok() const noexcept
    {
      return map_ != (void *)-1 && data_base_ != nullptr &&
             file_size_ >= 8;
    }

    void Dataset<uint64_t>::move_from(Dataset &&o) noexcept
    {
      file_path = std::move(o.file_path);
      dim = o.dim;
      total_cnt = o.total_cnt;

      fd_ = o.fd_;
      o.fd_ = -1;
      map_ = o.map_;
      o.map_ = (void *)-1;
      file_size_ = o.file_size_;
      o.file_size_ = 0;
      data_base_ = o.data_base_;
      o.data_base_ = nullptr;
      data_region_bytes_ = o.data_region_bytes_;
      o.data_region_bytes_ = 0;
    }

    void Dataset<uint64_t>::cleanup() noexcept
    {
      if (map_ != (void *)-1)
      {
        ::munmap(map_,
                 static_cast<size_t>(std::min<uint64_t>(file_size_, SIZE_MAX)));
        map_ = (void *)-1;
      }
      if (fd_ >= 0)
      {
        ::close(fd_);
        fd_ = -1;
      }
      data_base_ = nullptr;
      file_size_ = 0;
      data_region_bytes_ = 0;
      dim = 0;
      total_cnt = 0;
    }

    void Dataset<uint64_t>::open_file_and_map(const std::string &path)
    {
      file_path = path;

      fd_ = ::open(path.c_str(), O_RDONLY);
      if (fd_ < 0)
      {
        throw std::system_error(errno, std::generic_category(),
                                "open failed: " + path);
      }

      struct stat st{};
      if (::fstat(fd_, &st) != 0)
      {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::system_error(e, std::generic_category(), "fstat failed");
      }
      if (st.st_size < 8)
      {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("file too small (< 8 header bytes)");
      }
      file_size_ = static_cast<uint64_t>(st.st_size);

      if (file_size_ > SIZE_MAX)
      {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("file too large to map on this platform");
      }

      map_ = ::mmap(nullptr, static_cast<size_t>(file_size_), PROT_READ,
                    MAP_PRIVATE, fd_, 0);
      madvise(map_, static_cast<size_t>(file_size_), MADV_WILLNEED);
      if (map_ == MAP_FAILED)
      {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::system_error(e, std::generic_category(), "mmap failed");
      }

      const std::byte *base = static_cast<const std::byte *>(map_);
      constexpr uint64_t kElemSize = sizeof(uint64_t);

      uint64_t num_points_qi = 0;
      uint32_t dim_qi = 0;
      bool qi_ok = false;
      if (file_size_ >= 12)
      {
        load_le(base + 0, num_points_qi);
        load_le(base + 8, dim_qi);

        uint64_t data_elems = 0, data_bytes = 0, expect_total = 0;
        if (!mul_overflow(num_points_qi, static_cast<uint64_t>(dim_qi),
                          &data_elems) &&
            !mul_overflow(data_elems, kElemSize, &data_bytes) &&
            !add_overflow(12, data_bytes, &expect_total) &&
            expect_total == file_size_)
        {
          qi_ok = true;
        }
      }

      uint32_t num_points_ii = 0;
      uint32_t dim_ii = 0;
      bool ii_ok = false;
      if (!qi_ok && file_size_ >= 8)
      {
        load_le(base + 0, num_points_ii);
        load_le(base + 4, dim_ii);

        uint64_t data_elems = 0, data_bytes = 0, expect_total = 0;
        if (!mul_overflow(static_cast<uint64_t>(num_points_ii),
                          static_cast<uint64_t>(dim_ii), &data_elems) &&
            !mul_overflow(data_elems, kElemSize, &data_bytes) &&
            !add_overflow(8, data_bytes, &expect_total) &&
            expect_total == file_size_)
        {
          ii_ok = true;
        }
      }

      uint64_t header_size = 0;
      uint64_t num_points = 0;
      uint32_t dim_local = 0;

      if (qi_ok)
      {
        header_size = 12;
        num_points = num_points_qi;
        dim_local = dim_qi;
      }
      else if (ii_ok)
      {
        header_size = 8;
        num_points = num_points_ii;
        dim_local = dim_ii;
      }
      else
      {
        bool accepted = false;
        if (file_size_ >= 12)
        {
          load_le(base + 0, num_points_qi);
          load_le(base + 8, dim_qi);
          uint64_t data_elems = 0, data_bytes = 0, need = 0;
          if (!mul_overflow(num_points_qi, static_cast<uint64_t>(dim_qi),
                            &data_elems) &&
              !mul_overflow(data_elems, kElemSize, &data_bytes) &&
              !add_overflow(12, data_bytes, &need) && file_size_ >= need &&
              dim_qi > 0)
          {
            header_size = 12;
            num_points = num_points_qi;
            dim_local = dim_qi;
            accepted = true;
          }
        }
        if (!accepted && file_size_ >= 8)
        {
          load_le(base + 0, num_points_ii);
          load_le(base + 4, dim_ii);
          uint64_t data_elems = 0, data_bytes = 0, need = 0;
          if (!mul_overflow(static_cast<uint64_t>(num_points_ii),
                            static_cast<uint64_t>(dim_ii), &data_elems) &&
              !mul_overflow(data_elems, kElemSize, &data_bytes) &&
              !add_overflow(8, data_bytes, &need) && file_size_ >= need &&
              dim_ii > 0)
          {
            header_size = 8;
            num_points = num_points_ii;
            dim_local = dim_ii;
            accepted = true;
          }
        }
        if (!accepted)
        {
          cleanup();
          throw std::runtime_error(
              "Unrecognized header: neither <QI> nor <II> fits this file.");
        }
      }

      uint64_t data_elems = 0, data_bytes = 0, need_total = 0;
      if (mul_overflow(num_points, static_cast<uint64_t>(dim_local), &data_elems) ||
          mul_overflow(data_elems, kElemSize, &data_bytes) ||
          add_overflow(header_size, data_bytes, &need_total) ||
          need_total > file_size_)
      {
        cleanup();
        throw std::runtime_error("file truncated or header/dtype mismatch");
      }

      dim = dim_local;
      total_cnt = num_points;
      data_base_ = reinterpret_cast<const uint64_t *>(base + header_size);
      data_region_bytes_ = data_bytes;

      (void)posix_madvise(
          const_cast<uint64_t *>(data_base_),
          static_cast<size_t>(std::min<uint64_t>(data_region_bytes_, SIZE_MAX)),
          POSIX_MADV_RANDOM);
    }

    Dataset<uint64_t>::Dataset(const std::string &path) { open_file_and_map(path); }
    Dataset<uint64_t>::~Dataset() { cleanup(); }
    Dataset<uint64_t>::Dataset(Dataset &&other) noexcept
    {
      move_from(std::move(other));
    }
    Dataset<uint64_t> &Dataset<uint64_t>::operator=(Dataset &&other) noexcept
    {
      if (this != &other)
      {
        cleanup();
        move_from(std::move(other));
      }
      return *this;
    }

    int32_t Dataset<uint64_t>::getVecs(const std::vector<uint64_t> &ids,
                                       std::vector<uint64_t> &out_vecs)
    {
      if (!mapped_ok() || dim == 0)
        return -1;

      uint64_t out_elems64 = 0;
      if (mul_overflow(static_cast<uint64_t>(ids.size()),
                       static_cast<uint64_t>(dim), &out_elems64))
        return -3;
      if (out_elems64 > SIZE_MAX)
        return -3;
      out_vecs.resize(static_cast<size_t>(out_elems64));

      const uint64_t row_bytes = static_cast<uint64_t>(dim) * sizeof(uint64_t);
      const std::byte *base = reinterpret_cast<const std::byte *>(data_base_);

      for (size_t j = 0; j < ids.size(); ++j)
      {
        uint64_t id = ids[j];
        if (id >= total_cnt)
          return -2;

        uint64_t offset_bytes = 0;
        if (mul_overflow(id, row_bytes, &offset_bytes))
          return -3;

        if (offset_bytes > data_region_bytes_)
          return -3;
        if (row_bytes > data_region_bytes_ - offset_bytes)
          return -3;

        const void *src = base + offset_bytes;
        uint64_t *dst = out_vecs.data() + static_cast<size_t>(j) * dim;

        (void)posix_madvise(
            const_cast<void *>(src),
            static_cast<size_t>(std::min<uint64_t>(row_bytes, SIZE_MAX)),
            POSIX_MADV_WILLNEED);

        std::memcpy(dst, src, static_cast<size_t>(row_bytes));
      }
      return 0;
    }


    bool Dataset<int32_t>::mul_overflow(uint64_t a, uint64_t b, uint64_t *out)
    {
      if (a == 0 || b == 0)
      {
        *out = 0;
        return false;
      }
      if (a > std::numeric_limits<uint64_t>::max() / b)
        return true;
      *out = a * b;
      return false;
    }
    bool Dataset<int32_t>::add_overflow(uint64_t a, uint64_t b, uint64_t *out)
    {
      if (a > std::numeric_limits<uint64_t>::max() - b)
        return true;
      *out = a + b;
      return false;
    }

    bool Dataset<int32_t>::mapped_ok() const noexcept
    {
      return map_ != (void *)-1 && data_base_ != nullptr &&
             file_size_ >= 8;
    }

    void Dataset<int32_t>::move_from(Dataset &&o) noexcept
    {
      file_path = std::move(o.file_path);
      dim = o.dim;
      total_cnt = o.total_cnt;

      fd_ = o.fd_;
      o.fd_ = -1;
      map_ = o.map_;
      o.map_ = (void *)-1;
      file_size_ = o.file_size_;
      o.file_size_ = 0;
      data_base_ = o.data_base_;
      o.data_base_ = nullptr;
      data_region_bytes_ = o.data_region_bytes_;
      o.data_region_bytes_ = 0;
    }

    void Dataset<int32_t>::cleanup() noexcept
    {
      if (map_ != (void *)-1)
      {
        ::munmap(map_,
                 static_cast<size_t>(std::min<uint64_t>(file_size_, SIZE_MAX)));
        map_ = (void *)-1;
      }
      if (fd_ >= 0)
      {
        ::close(fd_);
        fd_ = -1;
      }
      data_base_ = nullptr;
      file_size_ = 0;
      data_region_bytes_ = 0;
      dim = 0;
      total_cnt = 0;
    }

    void Dataset<int32_t>::open_file_and_map(const std::string &path)
    {
      file_path = path;

      fd_ = ::open(path.c_str(), O_RDONLY);
      if (fd_ < 0)
      {
        throw std::system_error(errno, std::generic_category(),
                                "open failed: " + path);
      }

      struct stat st{};
      if (::fstat(fd_, &st) != 0)
      {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::system_error(e, std::generic_category(), "fstat failed");
      }
      if (st.st_size < 8)
      {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("file too small (< 8 header bytes)");
      }
      file_size_ = static_cast<uint64_t>(st.st_size);

      if (file_size_ > SIZE_MAX)
      {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("file too large to map on this platform");
      }

      map_ = ::mmap(nullptr, static_cast<size_t>(file_size_), PROT_READ,
                    MAP_PRIVATE, fd_, 0);
      madvise(map_, static_cast<size_t>(file_size_), MADV_WILLNEED);
      if (map_ == MAP_FAILED)
      {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::system_error(e, std::generic_category(), "mmap failed");
      }

      const std::byte *base = static_cast<const std::byte *>(map_);
      constexpr uint64_t kElemSize = sizeof(int32_t);

      uint64_t num_points_qi = 0;
      uint32_t dim_qi = 0;
      bool qi_ok = false;
      if (file_size_ >= 12)
      {
        load_le(base + 0, num_points_qi);
        load_le(base + 8, dim_qi);

        uint64_t data_elems = 0, data_bytes = 0, expect_total = 0;
        if (!mul_overflow(num_points_qi, static_cast<uint64_t>(dim_qi),
                          &data_elems) &&
            !mul_overflow(data_elems, kElemSize, &data_bytes) &&
            !add_overflow(12, data_bytes, &expect_total) &&
            expect_total == file_size_)
        {
          qi_ok = true;
        }
      }

      uint32_t num_points_ii = 0;
      uint32_t dim_ii = 0;
      bool ii_ok = false;
      if (!qi_ok && file_size_ >= 8)
      {
        load_le(base + 0, num_points_ii);
        load_le(base + 4, dim_ii);

        uint64_t data_elems = 0, data_bytes = 0, expect_total = 0;
        if (!mul_overflow(static_cast<uint64_t>(num_points_ii),
                          static_cast<uint64_t>(dim_ii), &data_elems) &&
            !mul_overflow(data_elems, kElemSize, &data_bytes) &&
            !add_overflow(8, data_bytes, &expect_total) &&
            expect_total == file_size_)
        {
          ii_ok = true;
        }
      }

      uint64_t header_size = 0;
      uint64_t num_points = 0;
      uint32_t dim_local = 0;

      if (qi_ok)
      {
        header_size = 12;
        num_points = num_points_qi;
        dim_local = dim_qi;
      }
      else if (ii_ok)
      {
        header_size = 8;
        num_points = num_points_ii;
        dim_local = dim_ii;
      }
      else
      {
        bool accepted = false;
        if (file_size_ >= 12)
        {
          load_le(base + 0, num_points_qi);
          load_le(base + 8, dim_qi);
          uint64_t data_elems = 0, data_bytes = 0, need = 0;
          if (!mul_overflow(num_points_qi, static_cast<uint64_t>(dim_qi),
                            &data_elems) &&
              !mul_overflow(data_elems, kElemSize, &data_bytes) &&
              !add_overflow(12, data_bytes, &need) && file_size_ >= need &&
              dim_qi > 0)
          {
            header_size = 12;
            num_points = num_points_qi;
            dim_local = dim_qi;
            accepted = true;
          }
        }
        if (!accepted && file_size_ >= 8)
        {
          load_le(base + 0, num_points_ii);
          load_le(base + 4, dim_ii);
          uint64_t data_elems = 0, data_bytes = 0, need = 0;
          if (!mul_overflow(static_cast<uint64_t>(num_points_ii),
                            static_cast<uint64_t>(dim_ii), &data_elems) &&
              !mul_overflow(data_elems, kElemSize, &data_bytes) &&
              !add_overflow(8, data_bytes, &need) && file_size_ >= need &&
              dim_ii > 0)
          {
            header_size = 8;
            num_points = num_points_ii;
            dim_local = dim_ii;
            accepted = true;
          }
        }
        if (!accepted)
        {
          cleanup();
          throw std::runtime_error(
              "Unrecognized header: neither <QI> nor <II> fits this file.");
        }
      }

      uint64_t data_elems = 0, data_bytes = 0, need_total = 0;
      if (mul_overflow(num_points, static_cast<uint64_t>(dim_local), &data_elems) ||
          mul_overflow(data_elems, kElemSize, &data_bytes) ||
          add_overflow(header_size, data_bytes, &need_total) ||
          need_total > file_size_)
      {
        cleanup();
        throw std::runtime_error("file truncated or header/dtype mismatch");
      }

      dim = dim_local;
      total_cnt = num_points;
      data_base_ = reinterpret_cast<const int32_t *>(base + header_size);
      data_region_bytes_ = data_bytes;

      (void)posix_madvise(
          const_cast<int32_t *>(data_base_),
          static_cast<size_t>(std::min<uint64_t>(data_region_bytes_, SIZE_MAX)),
          POSIX_MADV_RANDOM);
    }

    Dataset<int32_t>::Dataset(const std::string &path) { open_file_and_map(path); }
    Dataset<int32_t>::~Dataset() { cleanup(); }
    Dataset<int32_t>::Dataset(Dataset &&other) noexcept
    {
      move_from(std::move(other));
    }
    Dataset<int32_t> &Dataset<int32_t>::operator=(Dataset &&other) noexcept
    {
      if (this != &other)
      {
        cleanup();
        move_from(std::move(other));
      }
      return *this;
    }

    int32_t Dataset<int32_t>::getVecs(const std::vector<uint64_t> &ids,
                                      std::vector<int32_t> &out_vecs)
    {
      if (!mapped_ok() || dim == 0)
        return -1;

      uint64_t out_elems64 = 0;
      if (mul_overflow(static_cast<uint64_t>(ids.size()),
                       static_cast<uint64_t>(dim), &out_elems64))
        return -3;
      if (out_elems64 > SIZE_MAX)
        return -3;
      out_vecs.resize(static_cast<size_t>(out_elems64));

      const uint64_t row_bytes = static_cast<uint64_t>(dim) * sizeof(int32_t);
      const std::byte *base = reinterpret_cast<const std::byte *>(data_base_);

      for (size_t j = 0; j < ids.size(); ++j)
      {
        uint64_t id = ids[j];
        if (id >= total_cnt)
          return -2;

        uint64_t offset_bytes = 0;
        if (mul_overflow(id, row_bytes, &offset_bytes))
          return -3;

        if (offset_bytes > data_region_bytes_)
          return -3;
        if (row_bytes > data_region_bytes_ - offset_bytes)
          return -3;

        const void *src = base + offset_bytes;
        int32_t *dst = out_vecs.data() + static_cast<size_t>(j) * dim;

        (void)posix_madvise(
            const_cast<void *>(src),
            static_cast<size_t>(std::min<uint64_t>(row_bytes, SIZE_MAX)),
            POSIX_MADV_WILLNEED);

        std::memcpy(dst, src, static_cast<size_t>(row_bytes));
      }
      return 0;
    }

  }
}