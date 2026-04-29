#include "runtime/resource/search_rsrcpool.hpp"

namespace minihypervec
{
  namespace resource
  {
    SearchResourcePoolLockFree *SearchResourcePoolLockFree::getInstance()
    {
      static SearchResourcePoolLockFree g_inst;
      return &g_inst;
    }

    int32_t SearchResourcePoolLockFree::initHostRsrc(uint32_t worker_cnt,
                                                     uint64_t io_buf_bytes,
                                                     uint64_t dis_buf_bytes)
    {
      io_buf_bytes_per_worker = io_buf_bytes;
      dis_buf_bytes_per_worker = dis_buf_bytes;

      worker_io_bufs.resize(worker_cnt, nullptr);
      worker_dis_bufs.resize(worker_cnt, nullptr);

      worker_dev_que_ids.resize(worker_cnt);
      g_nvme_manager = nvme::NVMeManager::getInstance();

      for (uint32_t worker_id = 0; worker_id < worker_cnt; worker_id++)
      {
        worker_io_bufs[worker_id] = static_cast<char *>(nvme::NVMeManager::mallocNVMeHostBuf(io_buf_bytes_per_worker));
        if (worker_io_bufs[worker_id] == nullptr)
        {
          std::cerr << "Failed to allocate io_bufs[" << worker_id << "] of size "
                    << static_cast<uint64_t>(io_buf_bytes_per_worker) << std::endl;
          return -1;
        }
        worker_dis_bufs[worker_id] = static_cast<char *>(malloc(dis_buf_bytes_per_worker));
        if (worker_dis_bufs[worker_id] == nullptr)
        {
          std::cerr << "Failed to allocate dis_bufs[" << worker_id << "] of size "
                    << static_cast<uint64_t>(dis_buf_bytes_per_worker) << std::endl;
          return -1;
        }
        uint32_t nvme_cnt = g_nvme_manager->getNVMeDevNum();
        worker_dev_que_ids[worker_id].resize(nvme_cnt);
        for (uint32_t nvme_id = 0; nvme_id < nvme_cnt; ++nvme_id)
        {
          g_nvme_manager->allocQue(
              nvme_id, 1,
              worker_dev_que_ids[worker_id]
                                [nvme_id]);
        }
      }
      return 0;
    }

    int32_t SearchResourcePoolLockFree::releaseHostRsrc()
    {
      for (uint32_t worker_id = 0; worker_id < worker_io_bufs.size(); worker_id++)
      {
        if (worker_io_bufs[worker_id] != nullptr)
        {
          nvme::NVMeManager::freeNVMeHostBuf(worker_io_bufs[worker_id]);
          worker_io_bufs[worker_id] = nullptr;
        }
        if (worker_dis_bufs[worker_id] != nullptr)
        {
          free(worker_dis_bufs[worker_id]);
          worker_dis_bufs[worker_id] = nullptr;
        }
      }
      worker_dev_que_ids.clear();
      return 0;
    }

    int32_t SearchResourcePoolLockFree::getSearchResourceHvc(
        uint32_t worker_id,
        const SearchResourceConfig &config,
        SearchTempResource *search_resource)
    {
      if (worker_id >= worker_io_bufs.size())
      {
        std::cerr << "Invalid worker_id " << worker_id << std::endl;
        return -1;
      }
      if (io_buf_bytes_per_worker < config.io_buf_bytes_required ||
          dis_buf_bytes_per_worker < config.dis_buf_bytes_required)
      {
        std::cerr << "Insufficient io_buf or dis_buf for worker " << worker_id
                  << ", required: " << config.io_buf_bytes_required
                  << ", available: " << io_buf_bytes_per_worker
                  << "; required: " << config.dis_buf_bytes_required
                  << ", available: " << dis_buf_bytes_per_worker << std::endl;
        return -1;
      }
      switch (config.vec_type)
      {
      case collection::VecType::INT8:
      {
        auto *hvc_int8_resource =
          dynamic_cast<MiniHyperVecConstSearchResource<int32_t> *>(search_resource);
        if (hvc_int8_resource == nullptr ||
            hvc_int8_resource->resource_type !=
            SearchResourceType::MINI_HYPERVEC_CONST_CPU_RESOURCE)
        {
          std::cerr << "SearchResourcePoolLockFree::getSearchResourceHvcInt8: "
                       "search_resource cast error"
                    << std::endl;
          return -1;
        }
        hvc_int8_resource->io_read_buf = reinterpret_cast<int8_t *>(worker_io_bufs[worker_id]);
        hvc_int8_resource->dis_addr = reinterpret_cast<int32_t *>(worker_dis_bufs[worker_id]);
        hvc_int8_resource->dev_que_id = worker_dev_que_ids[worker_id];
        break;
      }
      default:
        std::cerr << "SearchResourcePoolLockFree::getSearchResourceHvc: "
                     "unsupported vec_type"
                  << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t SearchResourcePoolLockFree::getSearchResource(
        uint32_t worker_id, const SearchResourceConfig &config,
        SearchTempResource *search_resource)
    {
      switch (config.index_type)
      {
      case collection::IndexType::HV_CONST:
        return getSearchResourceHvc(worker_id, config, search_resource);
        break;

      default:
        std::cerr << "SearchResourcePoolLockFree::getSearchResource: "
                     "unsupported index_type"
                  << std::endl;
        return -1;
        break;
      }
    }

  } // namespace resource
} // namespace minihypervec