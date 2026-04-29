#include "runtime/worker/serving_worker.hpp"

namespace minihypervec
{
  namespace runtime
  {
    int32_t ServingWorker::init(uint32_t worker_id, int32_t core_id)
    {
      m_worker_id = worker_id;
      m_running_core_id = core_id;
      g_search_rsrc_pool = resource::SearchResourcePoolLockFree::getInstance();
      index_holder = runtime::IndexHolder::getInstance();
      return 0;
    }

    void ServingWorker::bindToCPU() const
    {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(m_running_core_id, &cpuset);
      pthread_t current_thread = pthread_self();
      int32_t rc =
          pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
      if (rc != 0)
      {
        std::cerr << "ServingWorker::bindToCPU: failed to bind to CPU, rc=" << rc
                  << std::endl;
      }
    }

    int32_t ServingWorker::getSearchResourceInfo(
        std::shared_ptr<index::IndexAbs<int32_t>> index_ptr,
        const index::SearchParam &search_param,
        std::shared_ptr<resource::SearchTempResource> &return_resource)
    {
      uint32_t worker_id = m_worker_id;
      resource::SearchResourceConfig config;
      config.vec_type = collection::VecType::INT8;
      config.index_type = index_ptr->getIndexType();
      switch (config.index_type)
      {
      case collection::IndexType::HV_CONST:
      {
        return_resource =
            std::make_shared<resource::MiniHyperVecConstSearchResource<int32_t>>();
        auto *search_param_hvc =
            dynamic_cast<const index::MiniHyperVecConstSearchParam *>(&search_param);
        auto *hvc_index_ptr =
            dynamic_cast<index::HyperConstImp<int32_t> *>(index_ptr.get());
        if (search_param_hvc == nullptr || hvc_index_ptr == nullptr)
        {
          std::cerr
              << "ServingWorker::getSearchResourceInfo: "
                 "search_param cast or index_ptr cast error, during cast to HVC."
              << std::endl;
          return -1;
        }
        uint64_t vec_scan_per_query =
            search_param_hvc->cluster_nprobe *
            dynamic_cast<index::MiniHyperVecConstBuildParam *>(
                hvc_index_ptr->m_collection_meta.index_meta.build_param.get())
                ->cluster_size;

        config.io_buf_bytes_required =
            vec_scan_per_query *
            dynamic_cast<index::MiniHyperVecConstBuildParam *>(
                hvc_index_ptr->m_collection_meta.index_meta.build_param.get())
                ->dim *
            sizeof(int8_t);
        config.dis_buf_bytes_required = vec_scan_per_query * sizeof(int32_t);
        break;
      }
      default:
        std::cerr << "ServingWorker::getSearchResourceInfo: unsupported "
                     "index_type"
                  << std::endl;
        return -1;
      }
      g_search_rsrc_pool->getSearchResource(worker_id, config,
                                            return_resource.get());
      return 0;
    }

    int32_t ServingWorker::searchKnn(
        const std::string &collection_name,
        const std::vector<int8_t> &query,
        const index::SearchParam &search_param,
        std::vector<std::pair<uint64_t, int32_t>> &res)
    {
      std::shared_ptr<index::IndexAbs<int32_t>> index_ptr = nullptr;
      if (index_holder->getIndex(collection_name, index_ptr) != 0)
      {
        std::cout << "ServingWorker::searchKnn: failed to get index ptr for collection " << collection_name << std::endl;
        return -1;
      }
      std::shared_ptr<resource::SearchTempResource> search_resource = nullptr;
      int32_t ret = getSearchResourceInfo(index_ptr, search_param, search_resource);
      if (ret != 0)
      {
        std::cerr
            << "ServingWorker::searchKnn of Int8: failed to get search resource."
            << std::endl;
        return ret;
      }
      ret = index_ptr->searchKnn(search_param, query, res, search_resource.get());
      return ret;
    }

    ServingWorkerPool *ServingWorkerPool::getInstance()
    {
      static ServingWorkerPool g_servingworker_pool_instance;
      return &g_servingworker_pool_instance;
    }

    int32_t ServingWorkerPool::init(uint32_t worker_cnt,
                                    const std::vector<int32_t> &cpu_ids)
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!workers_.empty())
      {
        std::cerr << "ServingWorkerPool::init: already initialized." << std::endl;
        return -1;
      }
      if (worker_cnt != cpu_ids.size())
      {
        std::cerr << "ServingWorkerPool::init: worker_cnt and cpu_ids size "
                     "mismatch."
                  << std::endl;
        return -1;
      }
      workers_.reserve(worker_cnt);
      for (uint32_t i = 0; i < worker_cnt; ++i)
      {
        auto worker = std::make_unique<ServingWorker>();
        int32_t core_id = -1;
        if (i < cpu_ids.size())
        {
          core_id = cpu_ids[i];
        }
        worker->init(i, core_id);
        workers_.emplace_back(std::move(worker));
        free_idx_.push(i);
      }
      std::cout << "Initialized ServingWorkerPool with " << worker_cnt << " workers. CPU IDs: ";
      for (const auto &cpu_id : cpu_ids)
      {
        std::cout << cpu_id << " ";
      }
      return 0;
    }

    int32_t ServingWorkerPool::shutdown()
    {
      std::lock_guard<std::mutex> lk(mu_);
      while (!free_idx_.empty())
        free_idx_.pop();
      workers_.clear();
      cv_.notify_all();
      return 0;
    }

    ServingWorker *ServingWorkerPool::acquire()
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&]
               { return !free_idx_.empty() || workers_.empty(); });
      if (workers_.empty())
        return nullptr;
      size_t idx = free_idx_.front();
      free_idx_.pop();
      return workers_[idx].get();
    }

    void ServingWorkerPool::release(ServingWorker *worker)
    {
      if (!worker)
        return;
      std::lock_guard<std::mutex> lk(mu_);
      for (size_t i = 0; i < workers_.size(); ++i)
      {
        if (workers_[i].get() == worker)
        {
          free_idx_.push(i);
          cv_.notify_one();
          return;
        }
      }
    }

    uint32_t ServingWorkerPool::size() const
    {
      std::lock_guard<std::mutex> lk(mu_);
      return static_cast<uint32_t>(workers_.size());
    }

    uint32_t ServingWorkerPool::available() const
    {
      std::lock_guard<std::mutex> lk(mu_);
      return static_cast<uint32_t>(free_idx_.size());
    }

  } // namespace runtime
} // namespace minihypervec