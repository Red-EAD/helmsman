#pragma once

#include "nvme/nvme_allocator.hpp"

namespace minihypervec
{
  namespace runtime
  {
#pragma pack(4)
    struct ClusterStripe
    {
      ClusterStripe() = default;
      ClusterStripe(uint32_t nvme_id, uint64_t lba_id);
      bool operator==(const ClusterStripe &other) const;
      uint32_t nvme_id_;
      uint64_t lba_id_;
    };
#pragma pack()

    class ClusterMap
    {
    public:
      static constexpr uint32_t kEmptyNvmeId = std::numeric_limits<uint32_t>::max();

    public:
      void init(uint64_t max_cluster_cnt, uint64_t one_cluster_page_cnt);
      void allocateChunks(const std::vector<nvme::Chunk> &chunks);
      uint64_t getClusterCnt();

      int32_t putClusterStripe(uint64_t cluster_id, ClusterStripe &pos, bool lock_inside = false);
      int32_t getClusterStripe(uint64_t cluster_id, ClusterStripe &pos, bool lock_inside = false);

      int32_t putClusterStripeBatch(const std::vector<uint64_t> &cluster_id,
                                    std::vector<ClusterStripe> &pos, bool lock_inside = false);
      int32_t getClusterStripeBatch(const std::vector<uint64_t> &cluster_id,
                                    std::vector<ClusterStripe> &pos, bool lock_inside = false);

      int32_t saveClusterMap(const std::string &path);
      int32_t loadClusterMap(const std::string &path);

    public:
      tbb::spin_rw_mutex m_inside_rw_mutex;
      uint64_t m_cluster_cnt = 0;
      uint64_t m_max_cluster_cnt = 0;
      uint64_t m_one_page_cnt = 0;
      std::vector<ClusterStripe> m_map_impl;

      std::unordered_map<uint32_t, std::vector<uint64_t>> m_free_by_dev_;
      uint64_t m_free_cnt_ = 0;

      std::vector<uint32_t> m_rr_devices_;
      uint64_t m_rr_index_dev_ = 0;
    };

  } // namespace runtime
} // namespace minihypervec