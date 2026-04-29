#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "collection/collection_meta.hpp"
#include "util/file/dataset.hpp"

namespace minihypervec
{
  namespace runtime
  {

    template <typename T>
    class ClusterExtra;

    template <>
    class ClusterExtra<int32_t>
    {
    public:
      std::vector<uint64_t> m_cluster_ids;
      std::vector<int32_t> m_cluster_norms;

      uint32_t m_dim;
      std::atomic<uint64_t> m_cluster_cnt;
      uint64_t m_cluster_size;
      collection::DisType m_dis_type;

      uint64_t m_max_cluster_cnt;

      std::vector<uint8_t> m_present;

    public:
      ClusterExtra() = default;
      ~ClusterExtra() = default;

      void init(uint32_t dim,
                uint64_t max_cluster_cnt,
                uint64_t cluster_size,
                collection::DisType dis_type);

      int32_t loadExtraInfo(const std::string &lists_ids_path,
                            const std::string &lists_norms_path);

      int32_t saveExtraInfo(const std::string &lists_ids_path,
                            const std::string &lists_norms_path);

      int32_t getClusterIDsNormsAddr(uint64_t cluster_id,
                                     std::vector<uint64_t> &list_ids,
                                     std::vector<int32_t> &list_norms);

      int32_t putClusterIDsNorms(uint64_t cluster_id,
                                 const std::vector<uint64_t> &list_ids,
                                 const std::vector<int32_t> &list_norms);

      int32_t getClusterIDsNormsBatch(const std::vector<uint64_t> &cluster_ids,
                                      std::vector<uint64_t> &list_ids,
                                      std::vector<int32_t> &list_norms);

      int32_t putClusterIDsNormsBatch(const std::vector<uint64_t> &cluster_ids,
                                      const std::vector<uint64_t> &list_ids,
                                      const std::vector<int32_t> &list_norms);
    };

  } // namespace runtime
} // namespace minihypervec
