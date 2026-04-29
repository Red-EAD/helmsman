#pragma once

#include "collection/types.hpp"
#include "compute/rank_cal.hpp"

namespace minihypervec
{
  namespace resource
  {
    using compute::RankPair;

    enum class SearchResourceType : uint32_t
    {
      UNKNOWN = 0,
      HNSW_RESOURCE = 1,
      MINI_HYPERVEC_CONST_CPU_RESOURCE = 2,
    };
    struct SearchTempResource
    {
      SearchResourceType resource_type = SearchResourceType::UNKNOWN;
      virtual ~SearchTempResource() = default;
    };

    template <typename T>
    struct MiniHyperVecConstSearchResource;

    template <>
    struct MiniHyperVecConstSearchResource<int32_t> : public SearchTempResource
    {
      MiniHyperVecConstSearchResource()
      {
        resource_type = SearchResourceType::MINI_HYPERVEC_CONST_CPU_RESOURCE;
      }
      int8_t *io_read_buf;
      int32_t *dis_addr;
      std::vector<int32_t> cluster_norms;
      std::vector<uint64_t> cluster_ids;
      std::vector<RankPair> rank_pairs;
      std::vector<std::vector<uint64_t>> dev_que_id;
    };

    struct DeployTempResource
    {
      char *write_buffer{nullptr};
      uint64_t max_write_bytes_once{0};
      std::vector<std::vector<uint64_t>> dev_que_id;
    };
  } // namespace resource
} // namespace minihypervec