#include "index/hnsw_imp.hpp"

namespace minihypervec
{
  namespace index
  {
    collection::IndexType HnswImp<int32_t>::getIndexType() const
    {
      return collection::IndexType::HNSW;
    }

    int32_t HnswImp<int32_t>::initBuildParam(const BuildParam &build_param)
    {
      auto *hnsw_build_param = dynamic_cast<const HnswBuildParam *>(&build_param);
      if (hnsw_build_param == nullptr)
      {
        std::cerr << "HnswImp<float>::initParam: build_param cast error"
                  << std::endl;
        return -1;
      }
      m_dim = hnsw_build_param->dim;
      m_max_elements = hnsw_build_param->max_elements;
      m_inner_M = hnsw_build_param->inner_M;
      m_build_ef = hnsw_build_param->build_ef;
      m_search_ef = hnsw_build_param->search_ef;
      switch (hnsw_build_param->metric)
      {
      case collection::DisType::L2:
        m_space = std::make_unique<hnswlib::L2SpaceI_int8>(m_dim);
        break;
      default:
        std::cerr << "HnswImp<float>::initParam: unsupported metric type"
                  << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t HnswImp<int32_t>::initIndex(const std::string &index_path,
                                        const BuildParam &build_param)
    {
      m_index_path = index_path;
      if (initBuildParam(build_param) != 0)
      {
        return -1;
      }
      m_hnsw = std::make_unique<hnswlib::HierarchicalNSW<int32_t>>(
          m_space.get(), m_max_elements, m_inner_M, m_build_ef);
      m_hnsw->setEf(m_search_ef);
      return 0;
    }

    int32_t HnswImp<int32_t>::checkBuildParamConsistency(
        const BuildParam &build_param)
    {
      auto *hnsw_build_param = dynamic_cast<const HnswBuildParam *>(&build_param);
      if (m_dim != hnsw_build_param->dim)
      {
        std::cerr << "HnswImp<int32_t>::checkParamConsistency: dim mismatch"
                  << std::endl;
        return -1;
      }
      if (m_max_elements != hnsw_build_param->max_elements)
      {
        std::cerr
            << "HnswImp<int32_t>::checkParamConsistency: max_elements mismatch"
            << std::endl;
        return -1;
      }
      if (m_inner_M != hnsw_build_param->inner_M)
      {
        std::cerr << "HnswImp<int32_t>::checkParamConsistency: inner_M mismatch"
                  << std::endl;
        return -1;
      }
      if (m_build_ef != hnsw_build_param->build_ef)
      {
        std::cerr << "HnswImp<int32_t>::checkParamConsistency: build_ef mismatch"
                  << std::endl;
        return -1;
      }
      return 0;
    }

    int32_t HnswImp<int32_t>::loadIndex(const std::string &index_path,
                                        const BuildParam &build_param)
    {
      m_index_path = index_path;
      if (initBuildParam(build_param) != 0)
      {
        return -1;
      }
      m_hnsw = std::make_unique<hnswlib::HierarchicalNSW<int32_t>>(m_space.get(),
                                                                   m_index_path);
      if (checkBuildParamConsistency(build_param) != 0)
      {
        return -1;
      }
      m_hnsw->setEf(m_search_ef);
      return 0;
    }

    int32_t HnswImp<int32_t>::saveIndex(const std::string &index_path)
    {
      if (m_hnsw == nullptr)
      {
        std::cerr << "HnswImp<int32_t>::saveIndex: index not initialized"
                  << std::endl;
        return -1;
      }
      m_hnsw->saveIndex(index_path);
      return 0;
    }

    int32_t HnswImp<int32_t>::addVecInBatch(const int8_t *vecs,
                                            const std::vector<uint64_t> &seq_ids,
                                            uint64_t count, uint64_t start,
                                            bool verbose)
    {
      if (m_hnsw == nullptr)
      {
        std::cerr << "HnswImp<int32_t>::addVecInBatch: index not initialized"
                  << std::endl;
        return -1;
      }
      for (uint64_t cnt = 0; cnt < count; cnt++)
      {
        m_hnsw->addPoint((void *)(vecs + (start + cnt) * m_dim),
                         (hnswlib::labeltype)seq_ids[start + cnt]);
        if (verbose && count > 100 && (cnt) % (count / 100UL) == 0)
        {
          std::cout << "build hnsw process: " << (100UL * cnt) / (count) << "%"
                    << std::endl;
        }
      }
      return 0;
    }

    int32_t HnswImp<int32_t>::addVec(const std::vector<int8_t> &vec,
                                     uint64_t seq_id)
    {
      if (m_hnsw == nullptr)
      {
        std::cerr << "HnswImp<int8_t>::addVec: index not initialized" << std::endl;
        return -1;
      }
      m_hnsw->addPoint((void *)(vec.data()), (hnswlib::labeltype)seq_id);
      return 0;
    }

    int32_t HnswImp<int32_t>::buildIndex(const int8_t *vecs,
                                         const std::vector<uint64_t> &seq_ids,
                                         uint32_t thread_cnt, bool verbose)
    {
      if (m_hnsw == nullptr)
      {
        std::cerr << "HnswImp<int32_t>::buildIndex: index not initialized"
                  << std::endl;
        return -1;
      }
      thread_cnt = std::min(thread_cnt, std::thread::hardware_concurrency());
      std::vector<std::thread> threads;
      uint64_t total = seq_ids.size();
      uint64_t batch_size = (total + thread_cnt - 1) / thread_cnt;
      for (uint32_t t = 0; t < thread_cnt; t++)
      {
        uint64_t start = t * batch_size;
        uint64_t end = std::min((t + 1) * batch_size, total);
        if (start >= end)
        {
          continue;
        }
        threads.emplace_back(&HnswImp<int32_t>::addVecInBatch, this, vecs, seq_ids,
                             end - start, start, verbose);
        verbose = false;
      }
      for (auto &thread : threads)
      {
        thread.join();
      }
      return 0;
    }

    int32_t HnswImp<int32_t>::searchKnn(
        const SearchParam &search_param, const std::vector<int8_t> &query,
        std::vector<std::pair<uint64_t, int32_t>> &res,
        resource::SearchTempResource *search_resource)
    {
      auto *hnsw_search_param = dynamic_cast<const HnswSearchParam *>(&search_param);
      if (hnsw_search_param == nullptr)
      {
        std::cerr << "HnswImp<int32_t>::searchKnn: search_param cast error"
                  << std::endl;
        return -1;
      }
      uint32_t topk = hnsw_search_param->topk_value;
      uint32_t max_visits = hnsw_search_param->max_visits;
      auto hnsw_res = m_hnsw->searchKnn((void *)(query.data()), topk, max_visits);
      res.resize(std::min(topk, (uint32_t)hnsw_res.size()));
      uint32_t pos = (uint32_t)res.size() - 1;
      while (!hnsw_res.empty())
      {
        res[pos] = {hnsw_res.top().second, hnsw_res.top().first};
        pos--;
        hnsw_res.pop();
      }
      return 0;
    }

  } // namespace index
} // namespace minihypervec