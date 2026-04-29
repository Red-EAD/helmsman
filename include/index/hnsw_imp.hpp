#pragma once

#include "index/index_abs.hpp"
#include "util/hnswlib/hnswlib.hpp"

namespace minihypervec
{
    namespace index
    {
        template <typename T>
        class HnswImp;

        template <>
        class HnswImp<int32_t> : public IndexAbs<int32_t>
        {
        public:
            std::string m_index_path;
            std::unique_ptr<hnswlib::SpaceInterface<int32_t>>
                m_space;
            std::unique_ptr<hnswlib::HierarchicalNSW<int32_t>> m_hnsw;
            uint32_t m_dim;
            uint64_t m_max_elements;
            uint32_t m_inner_M;
            uint32_t m_build_ef;
            uint32_t m_search_ef;

        public:
            HnswImp() = default;
            ~HnswImp() override = default;

        public:
            int32_t initIndex(const std::string &index_path,
                              const BuildParam &build_param);
            int32_t loadIndex(const std::string &index_path,
                              const BuildParam &build_param);
            int32_t saveIndex(const std::string &index_path);
            int32_t buildIndex(const int8_t *vecs, const std::vector<uint64_t> &seq_ids,
                               uint32_t thread_cnt, bool verbose = true);
            int32_t searchKnn(const SearchParam &search_param,
                              const std::vector<int8_t> &query,
                              std::vector<std::pair<uint64_t, int32_t>> &res,
                              resource::SearchTempResource *search_resource = nullptr)
                override;
            int32_t addVecInBatch(const int8_t *vecs,
                                  const std::vector<uint64_t> &seq_ids, uint64_t count,
                                  uint64_t start, bool verbose = false);
            int32_t addVec(const std::vector<int8_t> &vec, uint64_t seq_id);
            collection::IndexType getIndexType() const override;

        private:
            int32_t initBuildParam(const BuildParam &build_param);
            int32_t checkBuildParamConsistency(const BuildParam &build_param);
        };

    } // namespace index
} // namespace minihypervec