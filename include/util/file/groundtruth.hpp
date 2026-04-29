#pragma once

#include "root.hpp"

namespace minihypervec
{
  namespace util
  {
    class GtReader
    {
    public:
      explicit GtReader(const std::string &path);
      ~GtReader();

      GtReader(const GtReader &) = delete;
      GtReader &operator=(const GtReader &) = delete;
      GtReader(GtReader &&) noexcept;
      GtReader &operator=(GtReader &&) noexcept;

      uint32_t num_queries() const noexcept { return nq_; }
      uint32_t k() const noexcept { return k_; }
      bool ids_are_u64() const noexcept { return ids_stride_ == 8; }
      uint8_t id_width_bytes() const noexcept
      {
        return static_cast<uint8_t>(ids_stride_);
      }

      int32_t getGroundTruth(uint32_t query, uint32_t k,
                             std::vector<uint64_t> &out_ids,
                             std::vector<float> &out_dists) const;

      int32_t gettGroundTruthBatch(
          const std::vector<uint32_t> &queries, uint32_t k,
          std::vector<std::vector<uint64_t>> &out_ids,
          std::vector<std::vector<float>> &out_dists) const;

    private:
      int32_t fd_ = -1;
      void *map_ = (void *)-1;
      size_t file_size_ = 0;

      uint32_t nq_ = 0;
      uint32_t k_ = 0;
      size_t ids_off_ = 0;
      size_t dists_off_ = 0;
      uint64_t ids_stride_ = 0;

      static bool mul_overflow_u64(uint64_t a, uint64_t b, uint64_t *out);
      static bool add_overflow_u64(uint64_t a, uint64_t b, uint64_t *out);
      void cleanup() noexcept;
      void move_from(GtReader &&o) noexcept;
      void open_and_map(const std::string &path);
      void parse_header_and_layout();
    };

  } // namespace util
} // namespace minihypervec