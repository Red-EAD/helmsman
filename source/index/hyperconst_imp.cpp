#include "index/hyperconst_imp.hpp"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace minihypervec
{
  namespace index
  {
    index::MiniHyperVecConstBuildParam *HyperConstImp<int32_t>::getBuildParamPtr()
        const
    {
      auto *hv_const_build_param = dynamic_cast<index::MiniHyperVecConstBuildParam *>(
          m_collection_meta.index_meta.build_param.get());
      return hv_const_build_param;
    }

    int32_t HyperConstImp<int32_t>::loadCollectionMetaDuringDeploy(
        const std::string &collection_name)
    {
      int32_t ret = 0;
      const std::string meta_path =
          release::constants::getIndexMetaPath(collection_name);
      ret = collection::CollectionMeta::loadCollectionMeta(meta_path,
                                                           m_collection_meta);
      if (ret != 0)
      {
        std::cerr << "Failed to load collection meta during deploy for "
                  << collection_name << ", path: " << meta_path << std::endl;
      }
      return ret;
    }

    int32_t HyperConstImp<int32_t>::initClusterExtraDuringDeploy(
        const std::string &collection_name)
    {
      int32_t ret = 0;
      auto *hv_const_build_param = getBuildParamPtr();
      if (hv_const_build_param == nullptr)
      {
        std::cerr << "Failed to init cluster extra during deploy for "
                  << collection_name
                  << ", invalid build param type in collection meta." << std::endl;
        return -1;
      }
      m_cluster_extra.init(
          hv_const_build_param->dim, hv_const_build_param->centroid_num,
          hv_const_build_param->cluster_size, hv_const_build_param->metric);
      const std::string cluster_ids_path =
          release::constants::getClusterIDsPath(collection_name);
      util::Dataset<uint64_t> cluster_ids(cluster_ids_path);
      if (cluster_ids.total_cnt != m_cluster_extra.m_max_cluster_cnt)
      {
        std::cerr << "Failed to init cluster extra during deploy for "
                  << collection_name << ", cluster ids count mismatch: expected "
                  << m_cluster_extra.m_max_cluster_cnt << ", got "
                  << cluster_ids.total_cnt << std::endl;
        return -1;
      }
      std::vector<uint64_t> cluster_ids_buf(cluster_ids.total_cnt);
      std::vector<uint64_t> cluster_lists_ids;
      std::vector<int32_t> cluster_lists_norms;
      std::iota(cluster_ids_buf.begin(), cluster_ids_buf.end(), 0);
      ret = cluster_ids.getVecs(cluster_ids_buf, cluster_lists_ids);
      if (ret != 0)
      {
        std::cerr << "Failed to read cluster ids during deploy for "
                  << collection_name << std::endl;
      }
      switch (hv_const_build_param->metric)
      {
      case collection::DisType::L2:
      {
        const std::string cluster_norms_path =
            release::constants::getClusterNormsPath(collection_name);
        util::Dataset<int32_t> cluster_norms(cluster_norms_path);
        ret = cluster_norms.getVecs(cluster_ids_buf, cluster_lists_norms);
        if (ret != 0)
        {
          std::cerr << "Failed to read cluster norms during deploy for "
                    << collection_name << std::endl;
          return ret;
        }
        break;
      }
      default:
        break;
      }
      ret = m_cluster_extra.putClusterIDsNormsBatch(
          cluster_ids_buf, cluster_lists_ids, cluster_lists_norms);
      ret = m_cluster_extra.saveExtraInfo(
          release::constants::getClusterExtraIDsPath(collection_name),
          release::constants::getClusterExtraNormsPath(collection_name));
      return ret;
    }

    int32_t HyperConstImp<int32_t>::allocateNVMeSpaceDuringDeploy(
        const std::string &collection_name)
    {
      auto *allocator = nvme::NVMeAllocator::getInstance();
      auto *hv_const_build_param = getBuildParamPtr();
      const uint64_t each_cluster_bytes = hv_const_build_param->cluster_size *
                                          hv_const_build_param->dim *
                                          sizeof(int8_t);
      if (allocator->chunkBytes() % each_cluster_bytes != 0)
      {
        std::cerr << "Each cluster size " << each_cluster_bytes
                  << " must align to chunk size " << allocator->chunkBytes()
                  << std::endl;
        return -1;
      }
      uint64_t clusters_per_chunk = allocator->chunkBytes() / each_cluster_bytes;
      uint64_t total_chunks =
          (hv_const_build_param->centroid_num + clusters_per_chunk - 1) /
          clusters_per_chunk;
      nvme::AllocationPlan plan;
      int32_t ret =
          allocator->allocate(total_chunks * allocator->chunkBytes(), plan);
      if (ret != 0)
      {
        std::cerr << "Failed to allocate NVMe space during load for "
                  << collection_name << ", chunks needed: " << total_chunks
                  << std::endl;
        return ret;
      }

      m_cluster_map.init(hv_const_build_param->centroid_num,
                         each_cluster_bytes / allocator->pageSize());
      m_cluster_map.allocateChunks(plan.chunks);

      return 0;
    }

    namespace
    {
      using DevQueue = std::pair<uint32_t, uint16_t>;

      void bindCurrentThreadToCore(uint32_t core_id)
      {
        const long cpu_cnt = ::sysconf(_SC_NPROCESSORS_ONLN);
        if (cpu_cnt <= 0)
        {
          return;
        }
        const uint32_t target_u32 = core_id % static_cast<uint32_t>(cpu_cnt);
        const int target = static_cast<int>(target_u32);
        if (target < 0 || target >= CPU_SETSIZE)
        {
          return;
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(target, &cpuset);
        (void)::pthread_setaffinity_np(::pthread_self(), sizeof(cpu_set_t), &cpuset);
      }

      void memcpyBatchToBuffer(const std::vector<char *> &src, char *dst, size_t begin,
                               size_t count, uint64_t bytes_per_item)
      {
        for (size_t i = 0; i < count; ++i)
        {
          std::memcpy(dst + i * bytes_per_item, src[begin + i], bytes_per_item);
        }
      }

      void memcpyBatchToBuffer(char *const *src, char *dst, size_t begin,
                               size_t count, uint64_t bytes_per_item)
      {
        for (size_t i = 0; i < count; ++i)
        {
          std::memcpy(dst + i * bytes_per_item, src[begin + i], bytes_per_item);
        }
      }

      void submitRange(const resource::DeployTempResource &deploy_resource,
                       const runtime::ClusterStripe *stripes, char *write_buffer_base,
                       size_t begin, size_t count, uint64_t lba_cnt,
                       std::map<DevQueue, uint64_t> &submit_cnt,
                       uint32_t que_id_offset = 0)
      {
        const uint64_t page_size = nvme::NVMeAllocator::getInstance()->pageSize();
        for (size_t i = 0; i < count; ++i)
        {
          const runtime::ClusterStripe &stripe = stripes[begin + i];
          const uint16_t que_id = static_cast<uint16_t>(
              deploy_resource.dev_que_id[stripe.nvme_id_][0ul + que_id_offset]);
          char *buf_i = write_buffer_base + i * lba_cnt * page_size;
          nvme::NVMeManager::getInstance()->writeSubmit(
              stripe.nvme_id_, buf_i, stripe.lba_id_, lba_cnt, que_id);
          submit_cnt[{stripe.nvme_id_, que_id}]++;
        }
      }

      void pollUntilComplete(const std::map<DevQueue, uint64_t> &submit_cnt)
      {
        bool all_done = false;
        while (!all_done)
        {
          all_done = true;
          for (const auto &kv : submit_cnt)
          {
            const uint32_t nvme_id = kv.first.first;
            const uint16_t qid = kv.first.second;
            const uint64_t need = kv.second;

            nvme::NVMeManager::getInstance()->pollCompletions(nvme_id, qid);
            const uint64_t finished =
                nvme::NVMeManager::getInstance()->getFinishedQue(nvme_id, qid);
            if (finished != need)
              all_done = false;
          }
        }
      }
      void resetFinishedQueues(const std::map<DevQueue, uint64_t> &submit_cnt)
      {
        for (const auto &kv : submit_cnt)
        {
          const uint32_t nvme_id = kv.first.first;
          const uint16_t qid = kv.first.second;
          nvme::NVMeManager::getInstance()->resetFinishedQue(nvme_id, qid);
        }
      }

      uint32_t computeMinQueueCntPerDev(
          const resource::DeployTempResource &deploy_resource)
      {
        if (deploy_resource.dev_que_id.empty())
        {
          return 0;
        }
        uint32_t min_cnt = UINT32_MAX;
        for (const auto &dev_ques : deploy_resource.dev_que_id)
        {
          min_cnt = std::min<uint32_t>(min_cnt, (uint32_t)dev_ques.size());
        }
        return (min_cnt == UINT32_MAX) ? 0 : min_cnt;
      }

      class DeployFlushWorkerPool
      {
      public:
        DeployFlushWorkerPool(uint32_t thread_cnt,
                              const resource::DeployTempResource &deploy_resource,
                              uint64_t per_thread_buf_bytes,
                              uint64_t each_cluster_bytes,
                              uint64_t one_stripe_page_cnt)
            : m_thread_cnt(thread_cnt),
              m_deploy_resource(deploy_resource),
              m_per_thread_buf_bytes(per_thread_buf_bytes),
              m_each_cluster_bytes(each_cluster_bytes),
              m_one_stripe_page_cnt(one_stripe_page_cnt)
        {
          m_threads.reserve(m_thread_cnt);
          for (uint32_t tid = 0; tid < m_thread_cnt; ++tid)
          {
            m_threads.emplace_back([this, tid]()
                                   { this->threadMain(tid); });
          }
        }

        ~DeployFlushWorkerPool()
        {
          {
            std::lock_guard<std::mutex> lg(m_mu);
            m_stop = true;
            ++m_generation;
          }
          m_cv_work.notify_all();
          for (auto &th : m_threads)
          {
            if (th.joinable())
            {
              th.join();
            }
          }
        }

        DeployFlushWorkerPool(const DeployFlushWorkerPool &) = delete;
        DeployFlushWorkerPool &operator=(const DeployFlushWorkerPool &) = delete;

        int32_t flushBatch(const runtime::ClusterStripe *stripes,
                           char *const *raw_vecs_ptrs, uint64_t batch_size)
        {
          if (batch_size == 0)
          {
            return 0;
          }
          m_failed.store(0, std::memory_order_relaxed);
          {
            std::lock_guard<std::mutex> lg(m_mu);
            m_ctx.stripes = stripes;
            m_ctx.raw_vecs_ptrs = raw_vecs_ptrs;
            m_ctx.batch_size = (size_t)batch_size;
            m_completed = 0;
            ++m_generation;
          }
          m_cv_work.notify_all();
          {
            std::unique_lock<std::mutex> ul(m_mu);
            m_cv_done.wait(ul, [&]()
                           { return m_completed == m_thread_cnt ||
                                    m_failed.load(std::memory_order_relaxed) != 0; });
            if (m_failed.load(std::memory_order_relaxed) != 0)
            {
              m_cv_done.wait(ul, [&]()
                             { return m_completed == m_thread_cnt; });
              return -1;
            }
          }
          return 0;
        }

      private:
        struct BatchContext
        {
          const runtime::ClusterStripe *stripes = nullptr;
          char *const *raw_vecs_ptrs = nullptr;
          size_t batch_size = 0;
        };

        void threadMain(uint32_t tid)
        {
          bindCurrentThreadToCore(tid);
          uint64_t local_gen = 0;

          while (true)
          {
            BatchContext ctx;
            uint64_t gen = 0;
            {
              std::unique_lock<std::mutex> ul(m_mu);
              m_cv_work.wait(ul,
                             [&]()
                             { return m_stop || m_generation != local_gen; });
              if (m_stop)
              {
                return;
              }
              local_gen = m_generation;
              gen = local_gen;
              ctx = m_ctx;
            }

            (void)gen;
            if (m_failed.load(std::memory_order_relaxed) == 0)
            {
              doFlushWork(tid, ctx);
            }

            {
              std::lock_guard<std::mutex> lg(m_mu);
              ++m_completed;
              if (m_completed == m_thread_cnt)
              {
                m_cv_done.notify_one();
              }
            }
          }
        }

        void doFlushWork(uint32_t tid, const BatchContext &ctx)
        {
          try
          {
            if (ctx.batch_size == 0)
            {
              return;
            }

            const uint64_t page_size =
                nvme::NVMeAllocator::getInstance()->pageSize();
            const uint64_t queue_depth =
                (uint64_t)nvme::NVMeMetaHandler::getInstance()
                    ->g_nvme_meta.global_meta.queue_depth;
            const uint64_t per_thread_by_buf =
                (m_each_cluster_bytes == 0)
                    ? 0
                    : (m_per_thread_buf_bytes / m_each_cluster_bytes);
            const uint64_t per_thread_max_per_loop = std::max<uint64_t>(
                1ull, std::min<uint64_t>(per_thread_by_buf, queue_depth));

            const size_t seg_size =
                (ctx.batch_size + m_thread_cnt - 1) / m_thread_cnt;
            const size_t seg_begin = (size_t)tid * seg_size;
            const size_t seg_end = std::min(ctx.batch_size, seg_begin + seg_size);
            if (seg_begin >= seg_end)
            {
              return;
            }

            char *thread_buf_base = m_deploy_resource.write_buffer +
                                    (uint64_t)tid * m_per_thread_buf_bytes;

            size_t pos = seg_begin;
            while (pos < seg_end)
            {
              if (m_failed.load(std::memory_order_relaxed) != 0)
              {
                return;
              }
              const size_t remain = seg_end - pos;
              const size_t count =
                  (size_t)std::min<uint64_t>(remain, per_thread_max_per_loop);

              const uint64_t need_bytes = (uint64_t)count * m_each_cluster_bytes;
              if (need_bytes > m_per_thread_buf_bytes)
              {
                m_failed.store(1, std::memory_order_relaxed);
                return;
              }

              memcpyBatchToBuffer(ctx.raw_vecs_ptrs, thread_buf_base, pos, count,
                                  m_each_cluster_bytes);
              std::map<DevQueue, uint64_t> submit_cnt;
              submitRange(m_deploy_resource, ctx.stripes, thread_buf_base, pos, count,
                          m_one_stripe_page_cnt, submit_cnt, tid);
              pollUntilComplete(submit_cnt);
              resetFinishedQueues(submit_cnt);

              pos += count;
            }
            (void)page_size;
          }
          catch (const std::exception &e)
          {
            m_failed.store(1, std::memory_order_relaxed);
            std::cerr << "deploy flush worker failed: " << e.what() << std::endl;
          }
          catch (...)
          {
            m_failed.store(1, std::memory_order_relaxed);
            std::cerr << "deploy flush worker failed: unknown exception" << std::endl;
          }
        }

      private:
        const uint32_t m_thread_cnt;
        const resource::DeployTempResource &m_deploy_resource;
        const uint64_t m_per_thread_buf_bytes;
        const uint64_t m_each_cluster_bytes;
        const uint64_t m_one_stripe_page_cnt;

        std::vector<std::thread> m_threads;
        std::mutex m_mu;
        std::condition_variable m_cv_work;
        std::condition_variable m_cv_done;
        bool m_stop = false;
        uint64_t m_generation = 0;
        uint32_t m_completed = 0;
        BatchContext m_ctx;
        std::atomic<int32_t> m_failed{0};
      };

    } // namespace

    int32_t HyperConstImp<int32_t>::flushClustersToNVMeDuringDeploy(
        const std::string &collection_name,
        const resource::DeployTempResource &deploy_resource,
        uint64_t flush_clusters_per_batch)
    {
      util::Dataset<uint64_t> cluster_ids(
          release::constants::getClusterIDsPath(collection_name));
      util::Dataset<int8_t> raw_dataset(
          release::constants::getRawdataPath(collection_name));
      if (raw_dataset.total_cnt != getBuildParamPtr()->max_elements)
      {
        std::cerr << "Failed to flush clusters to NVMe during deploy for "
                  << collection_name << ", raw data count mismatch: expected "
                  << getBuildParamPtr()->max_elements << ", got "
                  << raw_dataset.total_cnt << std::endl;
        return -1;
      }

      auto *hv_const_build_param = getBuildParamPtr();
      const uint64_t each_cluster_bytes = hv_const_build_param->cluster_size *
                                          hv_const_build_param->dim *
                                          sizeof(int8_t);
      const uint64_t page_size = nvme::NVMeAllocator::getInstance()->pageSize();
      if (each_cluster_bytes == 0 || (each_cluster_bytes % page_size) != 0)
      {
        std::cerr << "Invalid cluster bytes/page alignment during deploy flush for "
                  << collection_name << std::endl;
        return -1;
      }
      const uint64_t one_stripe_page_cnt = each_cluster_bytes / page_size;

      std::vector<uint64_t> all_cluster_ids(cluster_ids.total_cnt);
      std::iota(all_cluster_ids.begin(), all_cluster_ids.end(), 0);
      std::vector<runtime::ClusterStripe> all_stripes(cluster_ids.total_cnt);
      m_cluster_map.putClusterStripeBatch(all_cluster_ids, all_stripes);

      const uint32_t thread_cnt = computeMinQueueCntPerDev(deploy_resource);
      if (thread_cnt == 0)
      {
        std::cerr << "No available NVMe queues in deploy_resource for "
                  << collection_name << std::endl;
        return -1;
      }

      const uint64_t max_threads_by_buf =
          deploy_resource.max_write_bytes_once / each_cluster_bytes;
      if (max_threads_by_buf == 0)
      {
        std::cerr
            << "write_buffer too small for one cluster during deploy flush for "
            << collection_name << std::endl;
        return -1;
      }
      const uint32_t effective_thread_cnt = std::max<uint32_t>(
          1u, std::min<uint32_t>(thread_cnt, (uint32_t)max_threads_by_buf));
      uint64_t per_thread_buf_bytes =
          deploy_resource.max_write_bytes_once / effective_thread_cnt;
      per_thread_buf_bytes = (per_thread_buf_bytes / page_size) * page_size;
      if (per_thread_buf_bytes < each_cluster_bytes)
      {
        std::cerr << "per-thread write_buffer too small during deploy flush for "
                  << collection_name << std::endl;
        return -1;
      }

      DeployFlushWorkerPool worker_pool(effective_thread_cnt, deploy_resource,
                                        per_thread_buf_bytes, each_cluster_bytes,
                                        one_stripe_page_cnt);

      for (uint64_t batch_start = 0; batch_start < cluster_ids.total_cnt;
           batch_start += flush_clusters_per_batch)
      {
        uint64_t batch_end =
            std::min(batch_start + flush_clusters_per_batch, cluster_ids.total_cnt);
        uint64_t batch_size = batch_end - batch_start;
        std::vector<uint64_t> batch_cluster_ids(batch_size);
        std::vector<std::vector<uint64_t>> lists_ids_percluster(batch_size);
        for (uint64_t cluster_id = 0; cluster_id < batch_size; ++cluster_id)
        {
          batch_cluster_ids[cluster_id] = batch_start + cluster_id;
          cluster_ids.getVecs({batch_cluster_ids[cluster_id]},
                              lists_ids_percluster[cluster_id]);
        }
        std::vector<std::vector<int8_t>> raw_vecs_percluster(batch_size);
        std::vector<char *> raw_vecs_ptrs(batch_size);
        for (uint64_t i = 0; i < batch_size; ++i)
        {
          if (raw_dataset.getVecs(lists_ids_percluster[i],
                                  raw_vecs_percluster[i]) != 0)
          {
            std::cerr << "Failed to get raw vecs during flush to NVMe for "
                      << collection_name << ", cluster id: " << batch_cluster_ids[i]
                      << std::endl;
            return -1;
          }
          raw_vecs_ptrs[i] = reinterpret_cast<char *>(raw_vecs_percluster[i].data());
        }
        const runtime::ClusterStripe *batch_stripes =
            all_stripes.data() + (size_t)batch_start;
        if (worker_pool.flushBatch(batch_stripes, raw_vecs_ptrs.data(),
                                   batch_size) != 0)
        {
          return -1;
        }
      }

      if (m_cluster_map.saveClusterMap(
              release::constants::getClusterMapPath(collection_name)) != 0)
      {
        std::cerr << "Failed to save cluster map during deploy for "
                  << collection_name << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t HyperConstImp<int32_t>::syncIndexDuringDeploy(
        const std::string &collection_name)
    {
      int32_t res = 0;
      if (nvme::NVMeMetaHandler::getInstance()->sync(nvme::getNVMeMetaPath()) !=
          0)
      {
        std::cerr << "Failed to sync NVMe meta during deploy for "
                  << collection_name << std::endl;
        return -1;
      }
      return res;
    }

    int32_t HyperConstImp<int32_t>::deployIndex(
        const std::string &collection_name,
        const resource::DeployTempResource &deploy_resource)
    {
      int32_t res = 0;
      res = loadCollectionMetaDuringDeploy(collection_name);
      if (res != 0)
      {
        return res;
      }

      res = initClusterExtraDuringDeploy(collection_name);
      if (res != 0)
      {
        return res;
      }

      res = allocateNVMeSpaceDuringDeploy(collection_name);
      if (res != 0)
      {
        return res;
      }

      res = flushClustersToNVMeDuringDeploy(collection_name, deploy_resource);
      if (res != 0)
      {
        return res;
      }

      res = syncIndexDuringDeploy(collection_name);
      if (res != 0)
      {
        return res;
      }
      return res;
    }

    int32_t HyperConstImp<int32_t>::findNearestClustersDuringSearch(
        const SearchParam &search_param, const std::vector<int8_t> &query,
        std::vector<uint64_t> &probe_ids)
    {
      auto *hvc_search_param =
          dynamic_cast<const index::MiniHyperVecConstSearchParam *>(&search_param);
      uint32_t probe_count =
          std::max(hvc_search_param->centroid_search_param->topk_value,
                   hvc_search_param->topk_value /
                       (uint32_t)10);
      std::vector<std::pair<uint64_t, int32_t>> centroid_dists;
      centroid_index->searchKnn(*hvc_search_param->centroid_search_param, query,
                                centroid_dists, nullptr);

      uint32_t out_probe_prune = 0;
      prune_tool->pruneScan(search_param, query, centroid_dists, out_probe_prune);
      probe_count = std::min(probe_count, out_probe_prune);
      probe_ids.resize(probe_count);
      for (uint32_t i = 0; i < probe_count; ++i)
      {
        probe_ids[i] = centroid_dists[i].first;
      }
      return 0;
    }

    int32_t HyperConstImp<int32_t>::launchLoadClustersFromNVMeDuringSearch(
        const std::vector<uint64_t> &probe_ids,
        const std::vector<runtime::ClusterStripe> &cluster_stripe,
        resource::SearchTempResource *search_resource,
        std::vector<std::pair<uint32_t, size_t>> &nvme_que_submits)
    {
      auto *g_nvme_manager = nvme::NVMeManager::getInstance();
      auto *hvc_srch_rsrc =
          dynamic_cast<resource::MiniHyperVecConstSearchResource<int32_t> *>(
              search_resource);
      nvme_que_submits.resize(g_nvme_manager->getNVMeDevNum());
      for (size_t nvme_id = 0; nvme_id < g_nvme_manager->getNVMeDevNum();
           ++nvme_id)
      {
        nvme_que_submits[nvme_id] = {hvc_srch_rsrc->dev_que_id[nvme_id][0], 0};
      }
      auto *m_build_param = dynamic_cast<index::MiniHyperVecConstBuildParam *>(
          m_collection_meta.index_meta.build_param.get());
      uint64_t per_cluster_size =
          m_build_param->dim * sizeof(int8_t) * m_build_param->cluster_size;
      uint64_t lba_cnt =
          per_cluster_size / nvme::NVMeAllocator::getInstance()->pageSize();
      for (size_t i = 0; i < probe_ids.size(); i++)
      {
        uint64_t cluster_id = probe_ids[i];
        const runtime::ClusterStripe &stripe = cluster_stripe[i];
        if (stripe.nvme_id_ == runtime::ClusterMap::kEmptyNvmeId)
        {
          std::cerr << "Error: cluster " << cluster_id
                    << " pos is empty, cannot read" << std::endl;
          return -1;
        }
        uint32_t que_id =
            static_cast<uint32_t>(hvc_srch_rsrc->dev_que_id[stripe.nvme_id_][0]);
        char *io_buf = (char *)(hvc_srch_rsrc->io_read_buf + i * per_cluster_size);
        g_nvme_manager->readSubmit(stripe.nvme_id_, io_buf, stripe.lba_id_, lba_cnt,
                                   que_id);
        nvme_que_submits[stripe.nvme_id_].second++;
      }
      for (size_t nvme_id = 0; nvme_id < g_nvme_manager->getNVMeDevNum();
           ++nvme_id)
      {
        g_nvme_manager->pollCompletions(nvme_id,
                                        hvc_srch_rsrc->dev_que_id[nvme_id][0]);
      }
      return 0;
    }

    int32_t HyperConstImp<int32_t>::prepareExtrasAndRankPairsDuringSearch(
        const std::vector<uint64_t> &probe_ids,
        resource::SearchTempResource *search_resource)
    {
      int32_t ret = 0;
      auto *hvc_srch_rsrc =
          dynamic_cast<resource::MiniHyperVecConstSearchResource<int32_t> *>(
              search_resource);
      ret = m_cluster_extra.getClusterIDsNormsBatch(
          probe_ids, hvc_srch_rsrc->cluster_ids, hvc_srch_rsrc->cluster_norms);
      auto *m_build_param = dynamic_cast<index::MiniHyperVecConstBuildParam *>(
          m_collection_meta.index_meta.build_param.get());
      switch (m_build_param->metric)
      {
      case collection::DisType::L2:
      {
        compute::SearchCPUFuncL2<int32_t>::prepareRankPairs(
            hvc_srch_rsrc->cluster_ids, hvc_srch_rsrc->rank_pairs);
        break;
      }
      default:
        std::cerr
            << "Unsupported metric type in prepareExtrasAndRankPairsDuringSearch."
            << std::endl;
        return -1;
      }
      return ret;
    }

    int32_t HyperConstImp<int32_t>::pollNVMeCompletionsDuringSearch(
        const std::vector<std::pair<uint32_t, size_t>> &nvme_que_submits)
    {
      auto *g_nvme_manager = nvme::NVMeManager::getInstance();
      for (size_t nvme_id = 0; nvme_id < g_nvme_manager->getNVMeDevNum();
           ++nvme_id)
      {
        uint32_t que_id = nvme_que_submits[nvme_id].first;
        uint32_t submit_cnt = nvme_que_submits[nvme_id].second;
        uint64_t finished = g_nvme_manager->getFinishedQue(nvme_id, que_id);
        while (finished < submit_cnt)
        {
          finished += g_nvme_manager->pollCompletions(nvme_id, que_id);
        }
        g_nvme_manager->resetFinishedQue(nvme_id, que_id);
      }
      return 0;
    }

    int32_t HyperConstImp<int32_t>::calculateInnerProductDuringSearch(
        const std::vector<int8_t> &query, uint64_t total_clusters,
        resource::SearchTempResource *search_resource)
    {
      auto *hvc_srch_rsrc =
          dynamic_cast<resource::MiniHyperVecConstSearchResource<int32_t> *>(
              search_resource);
      auto *m_build_param = dynamic_cast<index::MiniHyperVecConstBuildParam *>(
          m_collection_meta.index_meta.build_param.get());
      const uint64_t total_vecs = total_clusters * m_build_param->cluster_size;
      const int8_t *query_ptr = query.data();
      const int8_t *data_ptr =
          reinterpret_cast<const int8_t *>(hvc_srch_rsrc->io_read_buf);
      compute::InnerProductInt8InBatch(query_ptr, data_ptr, hvc_srch_rsrc->dis_addr,
                                       m_build_param->dim, total_vecs);
      return 0;
    }

    int32_t HyperConstImp<int32_t>::rankFinalTopKDuringSearch(
        const std::vector<int8_t> &query, uint64_t total_clusters,
        const SearchParam &search_param,
        resource::SearchTempResource *search_resource,
        std::vector<std::pair<uint64_t, int32_t>> &res)
    {
      std::vector<std::pair<uint64_t, int32_t>> cluster_res;
      auto *m_build_param = dynamic_cast<index::MiniHyperVecConstBuildParam *>(
          m_collection_meta.index_meta.build_param.get());
      auto *hvc_srch_rsrc =
          dynamic_cast<resource::MiniHyperVecConstSearchResource<int32_t> *>(
              search_resource);
      auto *hvc_search_param =
          dynamic_cast<const index::MiniHyperVecConstSearchParam *>(&search_param);
      switch (m_build_param->metric)
      {
      case collection::DisType::L2:
      {
        int32_t q_norm = 0;
        for (uint32_t d = 0; d < m_build_param->dim; d++)
        {
          q_norm +=
              static_cast<int32_t>(query[d]) * static_cast<int32_t>(query[d]);
        }
        compute::SearchCPUFuncL2<int32_t>::rankTopK(
            hvc_srch_rsrc->dis_addr, q_norm, hvc_srch_rsrc->cluster_norms,
            m_build_param->cluster_size * total_clusters,
            hvc_srch_rsrc->rank_pairs, hvc_search_param->topk_value, cluster_res);
        res = std::move(cluster_res);
        if (res.size() > hvc_search_param->topk_value)
        {
          res.resize(hvc_search_param->topk_value);
        }
        break;
      }
      default:
        std::cerr << "Unsupported metric type in rankFinalTopKDuringSearch."
                  << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t HyperConstImp<int32_t>::searchKnn(
        const SearchParam &search_param, const std::vector<int8_t> &query,
        std::vector<std::pair<uint64_t, int32_t>> &res,
        resource::SearchTempResource *search_resource)
    {
      int32_t ret = 0;
      std::vector<uint64_t> probe_ids;
      ret = findNearestClustersDuringSearch(search_param, query, probe_ids);
      if (ret != 0)
      {
        std::cerr << "Failed to find nearest clusters during search." << std::endl;
        return ret;
      }
      std::vector<runtime::ClusterStripe> cluster_stripes_pos;
      ret = m_cluster_map.getClusterStripeBatch(probe_ids, cluster_stripes_pos);
      if (ret != 0)
      {
        std::cerr << "Failed to get cluster stripes during search." << std::endl;
        return ret;
      }
      std::vector<std::pair<uint32_t, size_t>> nvme_que_submits;
      ret = launchLoadClustersFromNVMeDuringSearch(
          probe_ids, cluster_stripes_pos, search_resource, nvme_que_submits);
      ret = prepareExtrasAndRankPairsDuringSearch(probe_ids, search_resource);
      if (ret != 0)
      {
        std::cerr << "Failed to prepare extras and rank pairs during search."
                  << std::endl;
        return ret;
      }
      ret = pollNVMeCompletionsDuringSearch(nvme_que_submits);
      if (ret != 0)
      {
        std::cerr << "Failed to poll NVMe completions during search." << std::endl;
        return ret;
      }
      ret = calculateInnerProductDuringSearch(query, probe_ids.size(),
                                              search_resource);
      if (ret != 0)
      {
        std::cerr << "Failed to calculate inner product during search."
                  << std::endl;
        return ret;
      }
      ret = rankFinalTopKDuringSearch(query, probe_ids.size(),
                                      search_param, search_resource, res);
      return ret;
    }

    int32_t HyperConstImp<int32_t>::loadMetaDuringLoad(
        const std::string &collection_name)
    {
      const std::string meta_path =
          release::constants::getIndexMetaPath(collection_name);
      const int32_t ret = collection::CollectionMeta::loadCollectionMeta(
          meta_path, m_collection_meta);
      if (ret != 0)
      {
        std::cerr << "Failed to load collection meta during load for "
                  << collection_name << ", path: " << meta_path << std::endl;
        return ret;
      }
      if (m_collection_meta.index_meta.build_param == nullptr)
      {
        std::cerr << "Invalid collection meta during load for "
                  << collection_name << ", path: " << meta_path << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t HyperConstImp<int32_t>::loadInmemIndexDuringLoad(
        const std::string &collection_name)
    {
      auto *m_build_param = dynamic_cast<index::MiniHyperVecConstBuildParam *>(
          m_collection_meta.index_meta.build_param.get());
      std::string centroids_index_path =
          release::constants::getCentroidsIndexPath(collection_name);
      collection::IndexType centroid_index_type =
          m_build_param->centroid_index_type;
      switch (centroid_index_type)
      {
      case collection::IndexType::HNSW:
      {
        centroid_index = std::make_unique<HnswImp<int32_t>>();
        auto *centroid_index_imp =
            dynamic_cast<HnswImp<int32_t> *>(centroid_index.get());
        if (centroid_index_imp->loadIndex(
                centroids_index_path, *(m_build_param->centroid_build_param)) != 0)
        {
          std::cerr << "Failed to load centroid HNSW index during load for "
                    << collection_name << std::endl;
          return -1;
        }
        break;
      }
      default:
        std::cerr << "Unsupported centroid index type during load for "
                  << collection_name << std::endl;
        return -1;
        break;
      }
      return 0;
    }

    int32_t HyperConstImp<int32_t>::loadClusterExtraDuringLoad(
        const std::string &collection_name)
    {
      std::string cluster_extra_ids_path =
          release::constants::getClusterExtraIDsPath(collection_name);
      std::string cluster_extra_norms_path =
          release::constants::getClusterExtraNormsPath(collection_name);
      if (m_cluster_extra.loadExtraInfo(cluster_extra_ids_path,
                                        cluster_extra_norms_path) != 0)
      {
        std::cerr << "Failed to load cluster extra during load for "
                  << collection_name << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t HyperConstImp<int32_t>::loadClusterMapDuringLoad(
        const std::string &collection_name)
    {
      std::string cluster_map_path =
          release::constants::getClusterMapPath(collection_name);
      if (m_cluster_map.loadClusterMap(cluster_map_path) != 0)
      {
        std::cerr << "Failed to load cluster map during load for "
                  << collection_name << ", path: " << cluster_map_path << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t HyperConstImp<int32_t>::loadPruneToolDuringLoad(
        const std::string &collection_name)
    {
      prune_tool = std::make_unique<prune::PruningToolNaive<int32_t>>();
      return 0;
    }

    int32_t HyperConstImp<int32_t>::loadIndex(const std::string &collection_name)
    {
      int32_t ret = 0;
      ret = loadMetaDuringLoad(collection_name);
      if (ret != 0)
      {
        return ret;
      }
      ret = loadInmemIndexDuringLoad(collection_name);
      if (ret != 0)
      {
        return ret;
      }
      ret = loadClusterExtraDuringLoad(collection_name);
      if (ret != 0)
      {
        return ret;
      }

      ret = loadClusterMapDuringLoad(collection_name);
      if (ret != 0)
      {
        return ret;
      }
      ret = loadPruneToolDuringLoad(collection_name);
      if (ret != 0)
      {
        return ret;
      }
      return 0;
    }

    collection::IndexType HyperConstImp<int32_t>::getIndexType() const
    {
      return collection::IndexType::HV_CONST;
    }

  } // namespace index
} // namespace minihypervec