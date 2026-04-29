#include "runtime/env/minihypervec_env.hpp"
namespace minihypervec
{
  namespace runtime
  {

    json MiniHyperVecHostRsrcParam::to_json() const
    {
      json j;
      j["io_buf_bytes_per_worker"] = io_buf_bytes_per_worker;
      j["dis_buf_bytes_per_worker"] = dis_buf_bytes_per_worker;
      return j;
    }

    void MiniHyperVecHostRsrcParam::from_json(const json &j)
    {
      io_buf_bytes_per_worker = j.at("io_buf_bytes_per_worker").get<uint64_t>();
      dis_buf_bytes_per_worker = j.at("dis_buf_bytes_per_worker").get<uint64_t>();
    }

    void MiniHyperVecHostRsrcParam::printRsrcParam() const
    {
      std::cout << "MiniHyperVecHostRsrcParam:" << std::endl;
      std::cout << "  IO Buffer Bytes per Worker: " << io_buf_bytes_per_worker
                << "Bytes"
                << " - " << io_buf_bytes_per_worker / 1024ul / 1024ul << "MB"
                << std::endl;
      std::cout << "  DIS Buffer Bytes per Worker: " << dis_buf_bytes_per_worker
                << "Bytes"
                << " - " << dis_buf_bytes_per_worker / 1024ul << "KB" << std::endl;
    }

    void MiniHyperVecEnvParam::printEnvParam() const
    {
      std::cout << "MiniHyperVecEnvParam:" << std::endl;
      std::cout << "  Offline Parallel Degree: " << offline_parallel_degree
                << std::endl;
      std::cout << "  Offline Worker Memory Bytes: " << offline_worker_mem_bytes
                << "Bytes"
                << " - " << offline_worker_mem_bytes / 1024ul / 1024ul << "MB"
                << std::endl;
      std::cout << "  Serving Worker Count: " << serving_worker_cnt << std::endl;
      std::cout << "  Serving Worker Core IDs: ";
      for (const auto &core_id : serving_worker_core_ids)
      {
        std::cout << core_id << " ";
      }
      std::cout << std::endl;
      serving_host_rsrc_param.printRsrcParam();
    }

    json MiniHyperVecEnvParam::to_json() const
    {
      json j;
      j["offline_worker_cnt"] = offline_worker_cnt;
      j["offline_parallel_degree"] = offline_parallel_degree;
      j["offline_worker_mem_bytes"] = offline_worker_mem_bytes;
      j["serving_worker_cnt"] = serving_worker_cnt;
      j["serving_worker_core_ids"] = serving_worker_core_ids;
      j["serving_host_rsrc_param"] = serving_host_rsrc_param.to_json();
      return j;
    }

    void MiniHyperVecEnvParam::from_json(const json &j)
    {
      offline_worker_cnt = j.at("offline_worker_cnt").get<uint32_t>();
      offline_parallel_degree = j.at("offline_parallel_degree").get<uint32_t>();
      offline_worker_mem_bytes = j.at("offline_worker_mem_bytes").get<uint64_t>();
      serving_worker_cnt = j.at("serving_worker_cnt").get<uint32_t>();
      serving_worker_core_ids =
          j.at("serving_worker_core_ids").get<std::vector<int32_t>>();
      serving_host_rsrc_param.from_json(j.at("serving_host_rsrc_param"));
    }

    int32_t MiniHyperVecEnv::loadEnvParam(const std::string &env_param_path)
    {
      std::ifstream ifs(env_param_path);
      if (!ifs.is_open())
      {
        std::cerr << "Failed to open env param file: " << env_param_path
                  << std::endl;
        return -1;
      }
      json j;
      try
      {
        ifs >> j;
        g_env_param.from_json(j);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Failed to parse env param file: " << e.what() << std::endl;
        return -1;
      }
      return 0;
    }

    MiniHyperVecEnv *MiniHyperVecEnv::getInstance()
    {
      static MiniHyperVecEnv instance;
      return &instance;
    }

    int32_t MiniHyperVecEnv::initPathConfig()
    {
      PathConfig *pc = PathConfig::getInstance();
      if (pc->init(g_path_config.data()) != 0)
      {
        std::cerr << "Failed to initialize PathConfig." << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t MiniHyperVecEnv::initNVMe()
    {
      nvme::NVMeManager *g_nvme_manager = nvme::NVMeManager::getInstance();
      if (g_nvme_manager->initNVMeMeta(nvme::getNVMeMetaPath()) != 0)
      {
        std::cerr << "Failed to initialize NVMe metadata." << std::endl;
        return -1;
      }
      if (g_nvme_manager->initNVMeEnv() != 0)
      {
        std::cerr << "Failed to initialize NVMe environment." << std::endl;
        return -1;
      }
      if (g_nvme_manager->initNVMeDev() != 0)
      {
        std::cerr << "Failed to initialize NVMe devices." << std::endl;
        return -1;
      }
      std::cout << "Got nvme dev num: " << (uint32_t)g_nvme_manager->getNVMeDevNum()
                << std::endl;
      nvme::AllocatorInitConfig g_nvme_allocator_cfg;
      g_nvme_allocator_cfg.meta_handler = nvme::NVMeMetaHandler::getInstance();
      if (nvme::NVMeAllocator::getInstance()->configure(
              g_nvme_allocator_cfg, /*call_init=*/true) != 0)
      {
        std::cerr << "Failed to initialize NVMe allocator." << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t MiniHyperVecEnv::saveEnvParam(const std::string &env_param_path)
    {
      std::ofstream ofs(env_param_path);
      if (!ofs.is_open())
      {
        std::cerr << "Failed to open env param file for writing: " << env_param_path
                  << std::endl;
        return -1;
      }
      json j = g_env_param.to_json();
      try
      {
        ofs << j.dump(4);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Failed to write env param to file: " << e.what() << std::endl;
        return -1;
      }
      ofs.flush();
      ofs.close();
      return 0;
    }

    int32_t MiniHyperVecEnv::initIndexHolder(const std::string &collection_name)
    {
      if (collection_name.empty())
      {
        std::cerr << "Failed to initialize index holder: collection_name is empty."
                  << std::endl;
        return -1;
      }

      IndexHolder *index_holder = IndexHolder::getInstance();
      auto index_ptr = std::make_shared<index::HyperConstImp<int32_t>>();
      if (index_ptr->loadIndex(collection_name) != 0)
      {
        std::cerr << "Error: load HV_CONST_Int8 index failed for collection: "
                  << collection_name << std::endl;
        return -1;
      }
      return index_holder->initIndex(collection_name, index_ptr);
    }

    int32_t MiniHyperVecEnv::initSearchResourcePool()
    {
      auto *g_search_rsrc_pool =
          resource::SearchResourcePoolLockFree::getInstance();
      const auto &rsrc_param = g_env_param.serving_host_rsrc_param;
      if (g_search_rsrc_pool->initHostRsrc(
              g_env_param.serving_worker_cnt, rsrc_param.io_buf_bytes_per_worker,
              rsrc_param.dis_buf_bytes_per_worker) != 0)
      {
        std::cerr << "Failed to initialize search resource pool." << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t MiniHyperVecEnv::initOfflineWorkers()
    {
      auto w = OfflineWorker::getInstance();
      if (w->init(g_env_param.offline_worker_mem_bytes, g_env_param.offline_parallel_degree) != 0)
      {
        std::cerr << "Failed to initialize offline worker." << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t MiniHyperVecEnv::initServingWorkers()
    {
      auto *g_serving_worker_pool = ServingWorkerPool::getInstance();
      if (g_serving_worker_pool->init(g_env_param.serving_worker_cnt,
                                      g_env_param.serving_worker_core_ids) != 0)
      {
        std::cerr << "Failed to initialize serving worker pool." << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t MiniHyperVecEnv::shutdownNVMe()
    {
      auto *g_nvme_manager = nvme::NVMeManager::getInstance();
      g_nvme_manager->releaseNVMeEnv();
      return 0;
    }

    int32_t MiniHyperVecEnv::shutdownSearchResourcePool()
    {
      auto *g_search_rsrc_pool =
          resource::SearchResourcePoolLockFree::getInstance();
      if (g_search_rsrc_pool->releaseHostRsrc() != 0)
      {
        std::cerr << "Failed to release search resource pool during shutdown."
                  << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t MiniHyperVecEnv::shutdownOfflineWorker()
    {
      auto w = OfflineWorker::getInstance();
      if (w->destroy() != 0)
      {
        std::cerr << "Failed to destroy offline worker." << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t MiniHyperVecEnv::shutdownServingWorker()
    {
      auto *g_serving_worker_pool = ServingWorkerPool::getInstance();
      if (g_serving_worker_pool->shutdown() != 0)
      {
        std::cerr << "Failed to shutdown serving worker pool." << std::endl;
        return -1;
      }
      return 0;
    }

    void MiniHyperVecEnv::printMiniHyperVecInfo() const { minihypervec::printAuthorInfo(); }

    int32_t MiniHyperVecEnv::initForDeploy()
    {
      if (initPathConfig() != 0)
      {
        std::cerr << "Failed to initialize path configuration." << std::endl;
        return -1;
      }
      if (loadEnvParam(release::constants::getHardwareMetaPath()) != 0)
      {
        std::cerr << "Failed to load MiniHyperVec environment parameters." << std::endl;
        return -1;
      }
      if (initNVMe() != 0)
      {
        std::cerr << "Failed to initialize NVMe subsystem." << std::endl;
        return -1;
      }
      if (initOfflineWorkers() != 0)
      {
        std::cerr << "Failed to initialize Offline Workers." << std::endl;
        return -1;
      }
      printMiniHyperVecInfo();
      return 0;
    }

    int32_t MiniHyperVecEnv::initForSearch(const std::string &collection_name)
    {
      if (initPathConfig() != 0)
      {
        std::cerr << "Failed to initialize path configuration." << std::endl;
        return -1;
      }
      if (loadEnvParam(release::constants::getHardwareMetaPath()) != 0)
      {
        std::cerr << "Failed to load MiniHyperVec environment parameters." << std::endl;
        return -1;
      }
      if (initNVMe() != 0)
      {
        std::cerr << "Failed to initialize NVMe subsystem." << std::endl;
        return -1;
      }
      if (initIndexHolder(collection_name) != 0)
      {
        std::cerr << "Failed to initialize index holder." << std::endl;
        return -1;
      }
      if (initSearchResourcePool() != 0)
      {
        std::cerr << "Failed to initialize Search Resource Pool." << std::endl;
        return -1;
      }
      if (initServingWorkers() != 0)
      {
        std::cerr << "Failed to initialize Serving Workers." << std::endl;
        return -1;
      }
      printMiniHyperVecInfo();
      return 0;
    }

    int32_t MiniHyperVecEnv::shutdownForDeploy()
    {
      if (shutdownOfflineWorker() != 0)
      {
        std::cerr << "Failed to shutdown Offline Workers." << std::endl;
        return -1;
      }
      if (shutdownNVMe() != 0)
      {
        std::cerr << "Failed to shutdown NVMe subsystem." << std::endl;
        return -1;
      }
      if (saveEnvParam(release::constants::getHardwareMetaPath()) != 0)
      {
        std::cerr << "Failed to save MiniHyperVec environment parameters." << std::endl;
        return -1;
      }
      printAuthorInfo();
      return 0;
    }

    int32_t MiniHyperVecEnv::shutdownForSearch()
    {
      if (shutdownServingWorker() != 0)
      {
        std::cerr << "Failed to shutdown Serving Workers." << std::endl;
        return -1;
      }
      if (shutdownSearchResourcePool() != 0)
      {
        std::cerr << "Failed to shutdown Search Resource Pool." << std::endl;
        return -1;
      }
      if (shutdownNVMe() != 0)
      {
        std::cerr << "Failed to shutdown NVMe subsystem." << std::endl;
        return -1;
      }
      if (saveEnvParam(release::constants::getHardwareMetaPath()) != 0)
      {
        std::cerr << "Failed to save MiniHyperVec environment parameters." << std::endl;
        return -1;
      }
      printAuthorInfo();
      return 0;
    }

  } // namespace runtime
} // namespace minihypervec