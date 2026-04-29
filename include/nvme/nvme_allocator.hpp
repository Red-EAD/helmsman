#pragma once

#include "nvme/nvme_meta.hpp"

namespace minihypervec
{
  namespace nvme
  {
    enum : int32_t
    {
      kOk = 0,
      kNoSpace = 1,
      kInvalidArg = 2,
      kNoDevices = 3,
      kDeviceMissing = 4,
      kStalled = 5
    };
    struct Chunk
    {
      uint32_t nvme_id;
      uint64_t start_page;
      uint64_t page_count;

      bool empty() const { return page_count == 0; }
    };

    struct AllocationPlan
    {
      uint64_t total_pages = 0;
      uint64_t total_bytes = 0;
      std::vector<Chunk> chunks;
      bool ok() const { return total_pages > 0 && !chunks.empty(); }
    };

    struct ChunkParams
    {
      uint64_t chunk_pages = 131072;
      uint64_t chunk_bytes = 64ull * 1024 * 1024;
      bool by_bytes = true;
    };

    struct AllocatorInitConfig
    {
      NVMeMetaHandler *meta_handler = nullptr;
      ChunkParams fixed;
    };

    class NVMeAllocator
    {
    public:
      struct DeviceState
      {
        uint32_t nvme_id = 0;
        uint64_t capacity_chunks = 0;
        std::vector<uint64_t> free_chunks;
      };

    public:
      NVMeAllocator() = default;

      bool configured_{false};
      NVMeMetaHandler *meta_handler_{nullptr};
      uint64_t chunk_pages_{0};
      std::unordered_map<uint32_t, DeviceState> dev_states_;
      std::vector<uint32_t> order_dev_;

    public:
      static NVMeAllocator *getInstance();
      int32_t configure(const AllocatorInitConfig &cfg, bool call_init = true);
      int32_t allocate(uint64_t size_bytes, AllocationPlan &plan);
      uint32_t pageSize() const;
      uint64_t chunkBytes() const;
      bool isConfigured() const { return configured_; }
      enum : int32_t
      {
        kOk = 0,
        kNotConfigured = -200,
        kAlreadyConfigured = -201,
        kInvalidMetaHandler = -202,
        kInvalidArgs = -203,
        kNotInitialized = -204,
      };

    private:
      int32_t init();
      NVMeDeviceMeta *findDeviceById(uint32_t id);
      uint64_t totalFreeChunks() const;
    };
  } // namespace nvme
} // namespace minihypervec