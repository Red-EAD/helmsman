#pragma once

#include "nvme/nvme_manager.hpp"
#include "runtime/resource/resource.hpp"

namespace minihypervec
{
  namespace resource
  {
    struct SearchResourceConfig
    {
      collection::VecType vec_type;
      collection::IndexType index_type;
      uint64_t io_buf_bytes_required;
      uint64_t dis_buf_bytes_required;
    };

    class SearchResourcePoolLockFree
    {
    public:
      uint64_t io_buf_bytes_per_worker = 0;
      uint64_t dis_buf_bytes_per_worker = 0;
      std::vector<char *> worker_io_bufs;
      std::vector<char *> worker_dis_bufs;

      nvme::NVMeManager *g_nvme_manager{nullptr};
      std::vector<std::vector<std::vector<uint64_t>>> worker_dev_que_ids;

    public:
      static SearchResourcePoolLockFree *getInstance();

      int32_t initHostRsrc(uint32_t worker_cnt,
                           uint64_t io_buf_bytes,
                           uint64_t dis_buf_bytes);
      int32_t getSearchResource(uint32_t worker_id,
                                const SearchResourceConfig &config,
                                SearchTempResource *search_resource);

      int32_t releaseHostRsrc();

    private:
      int32_t getSearchResourceHvc(uint32_t worker_id,
                                   const SearchResourceConfig &config,
                                   SearchTempResource *search_resource);
    };
  } // namespace resource
} // namespace minihypervec