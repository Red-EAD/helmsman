#pragma once

#include "index/params.hpp"
#include "root.hpp"

namespace minihypervec
{
    namespace prune
    {
        template <typename T>
        class PruningTool;

        template <>
        class PruningTool<int32_t>
        {
        public:
            PruningTool() = default;
            virtual ~PruningTool() = default;
            virtual int32_t pruneScan(
                const index::SearchParam &search_param, const std::vector<int8_t> &query,
                const std::vector<std::pair<uint64_t, int32_t>> &centroid_res,
                uint32_t &out_probe);
        };

        template <typename T>
        class PruningToolNaive;

        template <>
        class PruningToolNaive<int32_t> : public PruningTool<int32_t>
        {
        public:
            uint32_t max_probe = 1024;

        public:
            PruningToolNaive() = default;
            ~PruningToolNaive() override = default;
            int32_t pruneScan(
                const index::SearchParam &search_param, const std::vector<int8_t> &query,
                const std::vector<std::pair<uint64_t, int32_t>> &centroid_res,
                uint32_t &out_probe) override;
        };

    } // namespace prune
} // namespace minihypervec