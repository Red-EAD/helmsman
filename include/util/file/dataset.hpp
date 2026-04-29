#pragma once

#include "root.hpp"

namespace minihypervec
{
  namespace util
  {
    template <typename T>
    class Dataset;

    template <>
    class Dataset<int8_t>
    {
    public:
      std::string file_path;
      uint32_t dim = 0;
      uint64_t total_cnt = 0;

      explicit Dataset(const std::string &path);
      ~Dataset();

      Dataset(const Dataset &) = delete;
      Dataset &operator=(const Dataset &) = delete;
      Dataset(Dataset &&) noexcept;
      Dataset &operator=(Dataset &&) noexcept;

      int32_t getVecs(const std::vector<uint64_t> &ids,
                      std::vector<int8_t> &out_vecs);
      int8_t *getDataBase() const { return const_cast<int8_t *>(data_base_); }

    private:
      int32_t fd_ = -1;
      void *map_ = (void *)-1;
      uint64_t file_size_ = 0;

      const int8_t *data_base_ = nullptr;
      uint64_t data_region_bytes_ = 0;

      static bool mul_overflow(uint64_t a, uint64_t b, uint64_t *out);
      static bool add_overflow(uint64_t a, uint64_t b, uint64_t *out);

      bool mapped_ok() const noexcept;
      void move_from(Dataset &&o) noexcept;
      void cleanup() noexcept;
      void open_file_and_map(const std::string &path);
    };

    template <>
    class Dataset<float>
    {
    public:
      std::string file_path;
      uint32_t dim = 0;
      uint64_t total_cnt = 0;

      explicit Dataset(const std::string &path);
      ~Dataset();

      Dataset(const Dataset &) = delete;
      Dataset &operator=(const Dataset &) = delete;
      Dataset(Dataset &&) noexcept;
      Dataset &operator=(Dataset &&) noexcept;

      int32_t getVecs(const std::vector<uint64_t> &ids,
                      std::vector<float> &out_vecs);
      float *getDataBase() const { return const_cast<float *>(data_base_); }

    private:
      int32_t fd_ = -1;
      void *map_ = (void *)-1;
      uint64_t file_size_ = 0;

      const float *data_base_ = nullptr;
      uint64_t data_region_bytes_ = 0;

      static bool mul_overflow(uint64_t a, uint64_t b, uint64_t *out);
      static bool add_overflow(uint64_t a, uint64_t b, uint64_t *out);

      bool mapped_ok() const noexcept;
      void move_from(Dataset &&o) noexcept;
      void cleanup() noexcept;
      void open_file_and_map(const std::string &path);
    };

    template <>
    class Dataset<uint64_t>
    {
    public:
      std::string file_path;
      uint32_t dim = 0;
      uint64_t total_cnt = 0;

      explicit Dataset(const std::string &path);
      ~Dataset();

      Dataset(const Dataset &) = delete;
      Dataset &operator=(const Dataset &) = delete;
      Dataset(Dataset &&) noexcept;
      Dataset &operator=(Dataset &&) noexcept;

      int32_t getVecs(const std::vector<uint64_t> &ids,
                      std::vector<uint64_t> &out_vecs);
      uint64_t *getDataBase() const { return const_cast<uint64_t *>(data_base_); }

    private:
      int32_t fd_ = -1;
      void *map_ = (void *)-1;
      uint64_t file_size_ = 0;

      const uint64_t *data_base_ = nullptr;
      uint64_t data_region_bytes_ = 0;

      static bool mul_overflow(uint64_t a, uint64_t b, uint64_t *out);
      static bool add_overflow(uint64_t a, uint64_t b, uint64_t *out);

      bool mapped_ok() const noexcept;
      void move_from(Dataset &&o) noexcept;
      void cleanup() noexcept;
      void open_file_and_map(const std::string &path);
    };

    template <>
    class Dataset<int32_t>
    {
    public:
      std::string file_path;
      uint32_t dim = 0;
      uint64_t total_cnt = 0;

      explicit Dataset(const std::string &path);
      ~Dataset();

      Dataset(const Dataset &) = delete;
      Dataset &operator=(const Dataset &) = delete;
      Dataset(Dataset &&) noexcept;
      Dataset &operator=(Dataset &&) noexcept;

      int32_t getVecs(const std::vector<uint64_t> &ids,
                      std::vector<int32_t> &out_vecs);
      int32_t *getDataBase() const { return const_cast<int32_t *>(data_base_); }

    private:
      int32_t fd_ = -1;
      void *map_ = (void *)-1;
      uint64_t file_size_ = 0;

      const int32_t *data_base_ = nullptr;
      uint64_t data_region_bytes_ = 0;

      static bool mul_overflow(uint64_t a, uint64_t b, uint64_t *out);
      static bool add_overflow(uint64_t a, uint64_t b, uint64_t *out);

      bool mapped_ok() const noexcept;
      void move_from(Dataset &&o) noexcept;
      void cleanup() noexcept;
      void open_file_and_map(const std::string &path);
    };

  } // namespace util
} // namespace minihypervec