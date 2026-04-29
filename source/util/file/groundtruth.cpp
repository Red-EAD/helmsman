#include "util/file/groundtruth.hpp"

namespace minihypervec
{
  namespace util
  {
    static inline void load_u32_le(const void *p, uint32_t &out)
    {
      std::memcpy(&out, p, sizeof(uint32_t));
    }

    bool GtReader::mul_overflow_u64(uint64_t a, uint64_t b, uint64_t *out)
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
    bool GtReader::add_overflow_u64(uint64_t a, uint64_t b, uint64_t *out)
    {
      if (a > std::numeric_limits<uint64_t>::max() - b)
        return true;
      *out = a + b;
      return false;
    }

    GtReader::GtReader(const std::string &path)
    {
      open_and_map(path);
      parse_header_and_layout();
    }
    GtReader::~GtReader() { cleanup(); }

    GtReader::GtReader(GtReader &&o) noexcept { move_from(std::move(o)); }
    GtReader &GtReader::operator=(GtReader &&o) noexcept
    {
      if (this != &o)
      {
        cleanup();
        move_from(std::move(o));
      }
      return *this;
    }

    void GtReader::move_from(GtReader &&o) noexcept
    {
      fd_ = o.fd_;
      o.fd_ = -1;
      map_ = o.map_;
      o.map_ = (void *)-1;
      file_size_ = o.file_size_;
      o.file_size_ = 0;

      nq_ = o.nq_;
      o.nq_ = 0;
      k_ = o.k_;
      o.k_ = 0;
      ids_off_ = o.ids_off_;
      o.ids_off_ = 0;
      dists_off_ = o.dists_off_;
      o.dists_off_ = 0;
      ids_stride_ = o.ids_stride_;
      o.ids_stride_ = 0;
    }

    void GtReader::cleanup() noexcept
    {
      if (map_ != (void *)-1)
      {
        ::munmap(map_, file_size_);
        map_ = (void *)-1;
      }
      if (fd_ >= 0)
      {
        ::close(fd_);
        fd_ = -1;
      }
      file_size_ = 0;
      nq_ = k_ = 0;
      ids_off_ = dists_off_ = 0;
      ids_stride_ = 0;
    }

    void GtReader::open_and_map(const std::string &path)
    {
      fd_ = ::open(path.c_str(), O_RDONLY);
      if (fd_ < 0)
        throw std::system_error(errno, std::generic_category(), "open failed");

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
        throw std::runtime_error("file too small (< 8)");
      }
      file_size_ = static_cast<size_t>(st.st_size);

      map_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
      if (map_ == MAP_FAILED)
      {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::system_error(e, std::generic_category(), "mmap failed");
      }
    }

    void GtReader::parse_header_and_layout()
    {
      const uint8_t *base = static_cast<const uint8_t *>(map_);

      load_u32_le(base + 0, nq_);
      load_u32_le(base + 4, k_);
      if (nq_ == 0 || k_ == 0)
      {
        throw std::runtime_error("invalid header: nq==0 or k==0");
      }

      uint64_t elems = 0, dist_bytes = 0;
      if (mul_overflow_u64(nq_, k_, &elems))
      {
        throw std::runtime_error("overflow on nq*k");
      }
      if (mul_overflow_u64(elems, sizeof(float), &dist_bytes))
      {
        throw std::runtime_error("overflow on dist bytes");
      }

      if (file_size_ < 8)
        throw std::runtime_error("file too small");
      const uint64_t payload = static_cast<uint64_t>(file_size_ - 8);
      if (payload < dist_bytes)
      {
        throw std::runtime_error("file truncated: payload < dist_bytes");
      }

      const uint64_t ids_bytes = payload - dist_bytes;
      if (elems == 0)
        throw std::runtime_error("nq*k == 0 (unexpected)");

      if (ids_bytes % elems != 0)
      {
        throw std::runtime_error("ids_bytes not divisible by (nq*k)");
      }
      const uint64_t stride = ids_bytes / elems;
      if (!(stride == 4 || stride == 8))
      {
        throw std::runtime_error("unsupported id width: expect 4 or 8 bytes");
      }

      ids_stride_ = stride;
      ids_off_ = 8;
      dists_off_ = static_cast<size_t>(8 + ids_bytes);

      uint64_t expect_total = 0;
      if (add_overflow_u64(8, ids_bytes, &expect_total) ||
          add_overflow_u64(expect_total, dist_bytes, &expect_total) ||
          expect_total != file_size_)
      {
        throw std::runtime_error("file size mismatch (header/layout)");
      }

      (void)posix_madvise(const_cast<uint8_t *>(base), file_size_,
                          POSIX_MADV_RANDOM);
    }

    int GtReader::getGroundTruth(uint32_t q, uint32_t k,
                                 std::vector<uint64_t> &out_ids,
                                 std::vector<float> &out_dists) const
    {
      if (q >= nq_)
        return -1;
      if (k == 0)
      {
        out_ids.clear();
        out_dists.clear();
        return 0;
      }

      const uint32_t kk = (k > k_) ? k_ : k;
      out_ids.resize(kk);
      out_dists.resize(kk);

      const uint8_t *base = static_cast<const uint8_t *>(map_);

      const uint64_t row_elems = static_cast<uint64_t>(k_);
      const uint64_t id_row_off =
          static_cast<uint64_t>(ids_off_) +
          static_cast<uint64_t>(q) * row_elems * ids_stride_;
      const uint8_t *idp = base + id_row_off;

      if (ids_stride_ == 4)
      {
        for (uint32_t i = 0; i < kk; ++i)
        {
          uint32_t v32;
          std::memcpy(&v32, idp + static_cast<size_t>(i) * 4ULL, 4);
          out_ids[i] = static_cast<uint64_t>(v32);
        }
      }
      else
      {
        std::memcpy(out_ids.data(), idp, static_cast<size_t>(kk) * 8U);
      }

      const uint64_t dist_row_off =
          static_cast<uint64_t>(dists_off_) +
          static_cast<uint64_t>(q) * row_elems * sizeof(float);
      const uint8_t *dp = base + dist_row_off;
      std::memcpy(out_dists.data(), dp, static_cast<size_t>(kk) * sizeof(float));

      return 0;
    }

    int GtReader::gettGroundTruthBatch(
        const std::vector<uint32_t> &queries, uint32_t k,
        std::vector<std::vector<uint64_t>> &out_ids,
        std::vector<std::vector<float>> &out_dists) const
    {
      out_ids.clear();
      out_dists.clear();
      out_ids.resize(queries.size());
      out_dists.resize(queries.size());
      if (queries.empty() || k == 0)
        return 0;

      const uint32_t kk = (k > k_) ? k_ : k;

      const uint8_t *base = static_cast<const uint8_t *>(map_);
      const uint64_t row_elems = static_cast<uint64_t>(k_);

      for (size_t t = 0; t < queries.size(); ++t)
      {
        const uint32_t q = queries[t];
        if (q >= nq_)
          return -1;

        auto &ids = out_ids[t];
        auto &dists = out_dists[t];
        ids.resize(kk);
        dists.resize(kk);

        const uint64_t id_row_off =
            static_cast<uint64_t>(ids_off_) +
            static_cast<uint64_t>(q) * row_elems * ids_stride_;
        const uint8_t *idp = base + id_row_off;

        if (ids_stride_ == 4)
        {
          for (uint32_t i = 0; i < kk; ++i)
          {
            uint32_t v32;
            std::memcpy(&v32, idp + static_cast<size_t>(i) * 4ULL, 4);
            ids[i] = static_cast<uint64_t>(v32);
          }
        }
        else
        {
          std::memcpy(ids.data(), idp, static_cast<size_t>(kk) * 8U);
        }
        const uint64_t dist_row_off =
            static_cast<uint64_t>(dists_off_) +
            static_cast<uint64_t>(q) * row_elems * sizeof(float);
        const uint8_t *dp = base + dist_row_off;
        std::memcpy(dists.data(), dp, static_cast<size_t>(kk) * sizeof(float));
      }

      return 0;
    }

  } // namespace util
} // namespace minihypervec