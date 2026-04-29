#pragma once

#include "runtime/env/index_holder.hpp"
#include "runtime/resource/search_rsrcpool.hpp"

namespace minihypervec
{
  namespace runtime
  {
    class ServingWorker
    {
    public:
      uint32_t m_worker_id;
      int32_t m_running_core_id;
      resource::SearchResourcePoolLockFree *g_search_rsrc_pool;
      runtime::IndexHolder *index_holder;

    public:
      ServingWorker() = default;
      ~ServingWorker() = default;

      int32_t init(uint32_t worker_id, int32_t core_id);
      void bindToCPU() const;

    public:
      int32_t searchKnn(const std::string &collection_name,
                        const std::vector<int8_t> &query,
                        const index::SearchParam &search_param,
                        std::vector<std::pair<uint64_t, int32_t>> &res);

    private:
      int32_t getSearchResourceInfo(
          std::shared_ptr<index::IndexAbs<int32_t>> index_ptr,
          const index::SearchParam &search_param,
          std::shared_ptr<resource::SearchTempResource> &return_resource);
    };

    class ServingWorkerPool
    {
    public:
      static ServingWorkerPool *getInstance();

      int32_t init(uint32_t worker_cnt, const std::vector<int32_t> &cpu_ids);
      int32_t shutdown();

      ServingWorker *acquire();
      void release(ServingWorker *worker);

      uint32_t size() const;
      uint32_t available() const;

      class WorkerHandle
      {
      public:
        WorkerHandle(ServingWorkerPool &pool, ServingWorker *w)
            : pool_(&pool), w_(w) {}
        WorkerHandle(WorkerHandle &&other) noexcept
            : pool_(other.pool_), w_(other.w_)
        {
          other.pool_ = nullptr;
          other.w_ = nullptr;
        }
        WorkerHandle &operator=(WorkerHandle &&other) noexcept
        {
          if (this != &other)
          {
            reset();
            pool_ = other.pool_;
            w_ = other.w_;
            other.pool_ = nullptr;
            other.w_ = nullptr;
          }
          return *this;
        }
        ~WorkerHandle() { reset(); }

        ServingWorker *operator->() const { return w_; }
        ServingWorker &operator*() const { return *w_; }
        ServingWorker *get() const { return w_; }
        explicit operator bool() const { return w_ != nullptr; }

      private:
        void reset()
        {
          if (pool_ && w_)
            pool_->release(w_);
          pool_ = nullptr;
          w_ = nullptr;
        }
        ServingWorkerPool *pool_{nullptr};
        ServingWorker *w_{nullptr};
      };

      WorkerHandle acquireHandle() { return WorkerHandle(*this, acquire()); }

    private:
      ServingWorkerPool() = default;

      std::vector<std::unique_ptr<ServingWorker>> workers_;
      std::queue<size_t> free_idx_;
      mutable std::mutex mu_;
      std::condition_variable cv_;
    };
  } // namespace runtime
} // namespace minihypervec