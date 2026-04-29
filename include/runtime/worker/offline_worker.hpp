#pragma once

#include "collection/collection_meta.hpp"
#include "index/hyperconst_imp.hpp"
#include "index/params.hpp"
#include "meta_path.hpp"
#include "nvme/nvme_manager.hpp"
#include "runtime/cluster/cluster_map.hpp"

namespace minihypervec
{
  namespace runtime
  {
    class OfflineWorker
    {
    public:
      static OfflineWorker *getInstance();

      OfflineWorker(const OfflineWorker &) = delete;
      OfflineWorker &operator=(const OfflineWorker &) = delete;

    public:
      char *write_buffer{nullptr};
      uint64_t max_write_bytes_once{0};
      nvme::NVMeManager *nvme_manager{nullptr};
      uint32_t parallel_degree{1};
      std::vector<std::vector<uint64_t>> dev_que_id;

    public:
      int32_t init(uint64_t max_buf_size, uint32_t parallel_degree_ = 1);
      int32_t destroy();
      resource::DeployTempResource getDeployResource() const;
      void bindToCPU(int32_t cpu_id = 0) const;
      int32_t deployIndex(const std::string &collection_name);

    private:
      OfflineWorker() = default;
    };

  } // namespace runtime
} // namespace minihypervec