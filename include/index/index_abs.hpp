#pragma once

#include "collection/collection_meta.hpp"
#include "index/params.hpp"

namespace minihypervec
{
    namespace index
    {
        template <typename T>
        class IndexAbs;

        template <>
        class IndexAbs<int32_t>
        {
        public:
            virtual int32_t searchKnn(
                const SearchParam &search_param, const std::vector<int8_t> &query,
                std::vector<std::pair<uint64_t, int32_t>> &res,
                resource::SearchTempResource *search_resource = nullptr);
            virtual int32_t deployIndex(
                const std::string &collection_name,
                const resource::DeployTempResource &deploy_resource);
            virtual ~IndexAbs() noexcept = default;
            virtual collection::IndexType getIndexType() const;
            collection::VecType getVecType() const { return collection::VecType::INT8; }
        };

    } // namespace index
} // namespace minihypervec