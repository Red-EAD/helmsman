#include "runtime/worker/offline_worker.hpp"

namespace minihypervec
{
  namespace runtime
  {
    OfflineWorker *OfflineWorker::getInstance()
    {
      static OfflineWorker instance;
      return &instance;
    }

    int32_t OfflineWorker::init(uint64_t max_buf_size, uint32_t parallel_degree_)
    {
      max_write_bytes_once = max_buf_size;
      parallel_degree = parallel_degree_;
      nvme_manager = nvme::NVMeManager::getInstance();
      write_buffer = static_cast<char *>(
          nvme::NVMeManager::mallocNVMeHostBuf(max_write_bytes_once));

      if (write_buffer == nullptr)
      {
        std::cerr << "OfflineWorker::init: failed to allocate write buffer"
                  << std::endl;
        return -1;
      }

      uint32_t nvme_dev_num = nvme_manager->getNVMeDevNum();
      dev_que_id.resize(nvme_dev_num);
      for (uint32_t dev_id = 0; dev_id < nvme_dev_num; dev_id++)
      {
        int32_t ret =
            nvme_manager->allocQue(dev_id, parallel_degree, dev_que_id[dev_id]);
        if (ret != 0)
        {
          std::cerr << "OfflineWorker::init: failed to alloc que for dev " << dev_id
                    << ", ret=" << ret << std::endl;
          return ret;
        }
        std::cout << "OfflineWorker::init: on dev: " << dev_id
                  << ", parallel_degree=" << parallel_degree
                  << " with I/O queues=" << dev_que_id[dev_id].size() << std::endl;
      }
      return 0;
    }

    resource::DeployTempResource OfflineWorker::getDeployResource() const
    {
      resource::DeployTempResource res;
      res.write_buffer = write_buffer;
      res.max_write_bytes_once = max_write_bytes_once;
      res.dev_que_id = dev_que_id;
      return res;
    }

    void OfflineWorker::bindToCPU(int32_t cpu_id) const
    {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(cpu_id, &cpuset);
      pthread_t current_thread = pthread_self();
      int32_t rc =
          pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
      if (rc != 0)
      {
        std::cerr << "OfflineWorker::bindToCPU: failed to bind to CPU, rc=" << rc
                  << std::endl;
      }
    }

    int32_t OfflineWorker::deployIndex(const std::string &collection_name)
    {
      auto index_ptr = std::make_shared<index::HyperConstImp<int32_t>>();
      return index_ptr->deployIndex(collection_name, getDeployResource());
    }

    int32_t OfflineWorker::destroy()
    {
      if (write_buffer != nullptr)
      {
        nvme::NVMeManager::freeNVMeHostBuf(write_buffer);
        write_buffer = nullptr;
      }
      return 0;
    }

  } // namespace runtime
} // namespace minihypervec