#pragma once

#include "collection/collection_meta.hpp"
#include "meta_path.hpp"
#include "nvme/nvme_allocator.hpp"
#include "nvme/nvme_manager.hpp"
#include "runtime/env/index_holder.hpp"
#include "runtime/resource/search_rsrcpool.hpp"
#include "runtime/worker/offline_worker.hpp"
#include "runtime/worker/serving_worker.hpp"

namespace minihypervec
{
  namespace runtime
  {
    using json = nlohmann::json;
    struct MiniHyperVecHostRsrcParam
    {
      uint64_t io_buf_bytes_per_worker = 0;
      uint64_t dis_buf_bytes_per_worker = 0;

      json to_json() const;
      void from_json(const json &j);
      void printRsrcParam() const;
    };

    struct MiniHyperVecEnvParam
    {
      uint32_t offline_worker_cnt = 0;
      uint32_t offline_parallel_degree = 32;
      uint64_t offline_worker_mem_bytes = 0;
      uint32_t serving_worker_cnt = 0;
      std::vector<int32_t> serving_worker_core_ids = {};
      MiniHyperVecHostRsrcParam serving_host_rsrc_param;
      json to_json() const;
      void from_json(const json &j);
      void printEnvParam() const;
    };

    class MiniHyperVecEnv
    {
    public:
      MiniHyperVecEnvParam g_env_param;

    public:
      static MiniHyperVecEnv *getInstance();
      int32_t loadEnvParam(const std::string &env_param_path);
      int32_t saveEnvParam(const std::string &env_param_path);

    public:
      int32_t initForDeploy();
      int32_t initForSearch(const std::string &collection_name);
      int32_t shutdownForDeploy();
      int32_t shutdownForSearch();

    public:
      int32_t initPathConfig();
      int32_t initNVMe();
      int32_t initIndexHolder(const std::string &collection_name);

      int32_t initSearchResourcePool();
      int32_t initOfflineWorkers();
      int32_t initServingWorkers();

      int32_t shutdownNVMe();
      int32_t shutdownSearchResourcePool();
      int32_t shutdownOfflineWorker();
      int32_t shutdownServingWorker();

      void printMiniHyperVecInfo() const;
    };
  } // namespace runtime
} // namespace minihypervec