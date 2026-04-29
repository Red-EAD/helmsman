#include "runtime/cluster/cluster_extra.hpp"

namespace minihypervec
{
  namespace runtime
  {
    namespace
    {
      constexpr uint64_t kEmptyId = std::numeric_limits<uint64_t>::max();
      constexpr uint32_t kMagicIDs = 0x49445843; // 'CXDI' - IDs file
      constexpr uint32_t kMagicNRM = 0x4E524D43; // 'CMRN' - Norms file

#pragma pack(1)
      struct FileHeader
      {
        uint32_t magic;
        uint32_t dim;
        uint32_t dis_type_u32;
        uint64_t cluster_cnt;
        uint64_t cluster_size;
        uint64_t max_cluster_cnt;
      };
#pragma pack()

      inline bool is_L2(collection::DisType t)
      {
        return t == collection::DisType::L2;
      }
      inline uint64_t base_index(uint64_t cid, uint64_t cluster_size)
      {
        return static_cast<uint64_t>(cid * cluster_size);
      }

      static int force_fsync_path_linux(const std::string &path)
      {
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
          return -1;
        int rc = ::fsync(fd);
        ::close(fd);
        return (rc == 0) ? 0 : -1;
      }
    } // namespace

    void ClusterExtra<int32_t>::init(uint32_t dim,
                                     uint64_t max_cluster_cnt,
                                     uint64_t cluster_size,
                                     collection::DisType dis_type)
    {
      m_dim = dim;
      m_cluster_cnt.store(0, std::memory_order_relaxed);
      m_max_cluster_cnt = max_cluster_cnt;
      m_cluster_size = cluster_size;
      m_dis_type = dis_type;

      const uint64_t cap_elems =
          static_cast<uint64_t>(m_max_cluster_cnt * m_cluster_size);
      m_cluster_ids.assign(cap_elems, kEmptyId);
      if (is_L2(m_dis_type))
      {
        m_cluster_norms.assign(cap_elems, 0);
      }
      else
      {
        m_cluster_norms.clear();
        m_cluster_norms.shrink_to_fit();
        std::cerr << "only support L2 metric for norms." << std::endl;
      }
      m_present.assign(m_max_cluster_cnt, 0);
    }

    int32_t ClusterExtra<int32_t>::saveExtraInfo(
        const std::string &lists_ids_path, const std::string &lists_norms_path)
    {
      const uint64_t cap_elems =
          static_cast<uint64_t>(m_max_cluster_cnt * m_cluster_size);
      if (m_cluster_ids.size() != cap_elems)
        return -11;

      {
        std::ofstream ofs(lists_ids_path, std::ios::binary | std::ios::trunc);
        if (!ofs)
          return -1;
        FileHeader hdr{kMagicIDs,
                       m_dim,
                       static_cast<uint32_t>(m_dis_type),
                       m_cluster_cnt.load(std::memory_order_relaxed),
                       m_cluster_size,
                       m_max_cluster_cnt};
        ofs.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
        ofs.write(reinterpret_cast<const char *>(m_cluster_ids.data()),
                  cap_elems * sizeof(uint64_t));
        ofs.flush();
        if (!ofs)
          return -1;
      }

      if (is_L2(m_dis_type))
      {
        if (m_cluster_norms.size() != cap_elems)
          return -12;
        std::ofstream ofs(lists_norms_path, std::ios::binary | std::ios::trunc);
        if (!ofs)
          return -2;
        FileHeader hdr{kMagicNRM,
                       m_dim,
                       static_cast<uint32_t>(m_dis_type),
                       m_cluster_cnt.load(std::memory_order_relaxed),
                       m_cluster_size,
                       m_max_cluster_cnt};
        ofs.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
        ofs.write(reinterpret_cast<const char *>(m_cluster_norms.data()),
                  cap_elems * sizeof(int32_t));
        ofs.flush();
        if (!ofs)
          return -2;
      }
      force_fsync_path_linux(lists_ids_path);
      if (is_L2(m_dis_type))
      {
        force_fsync_path_linux(lists_norms_path);
      }
      return 0;
    }

    int32_t ClusterExtra<int32_t>::loadExtraInfo(
        const std::string &lists_ids_path, const std::string &lists_norms_path)
    {
      FileHeader hdr{};
      {
        std::ifstream ifs(lists_ids_path, std::ios::binary);
        if (!ifs)
          return -1;
        ifs.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
        if (!ifs || hdr.magic != kMagicIDs)
          return -1;

        m_dim = hdr.dim;
        m_dis_type = static_cast<collection::DisType>(hdr.dis_type_u32);
        m_cluster_size = hdr.cluster_size;
        m_max_cluster_cnt = hdr.max_cluster_cnt;

        const uint64_t cap_elems =
            static_cast<uint64_t>(m_max_cluster_cnt * m_cluster_size);
        m_cluster_ids.resize(cap_elems);
        ifs.read(reinterpret_cast<char *>(m_cluster_ids.data()),
                 cap_elems * sizeof(uint64_t));
        if (!ifs)
          return -1;
      }

      if (is_L2(m_dis_type))
      {
        FileHeader nh{};
        std::ifstream ifn(lists_norms_path, std::ios::binary);
        if (!ifn)
          return -2;
        ifn.read(reinterpret_cast<char *>(&nh), sizeof(nh));
        if (!ifn || nh.magic != kMagicNRM)
          return -2;

        if (nh.dim != m_dim ||
            nh.dis_type_u32 != static_cast<uint32_t>(m_dis_type) ||
            nh.cluster_size != m_cluster_size ||
            nh.max_cluster_cnt != m_max_cluster_cnt)
        {
          return -2;
        }

        const uint64_t cap_elems =
            static_cast<uint64_t>(m_max_cluster_cnt * m_cluster_size);
        m_cluster_norms.resize(cap_elems);
        ifn.read(reinterpret_cast<char *>(m_cluster_norms.data()),
                 cap_elems * sizeof(int32_t));
        if (!ifn)
          return -2;
      }
      else
      {
        m_cluster_norms.clear();
        m_cluster_norms.shrink_to_fit();
      }

      m_present.assign(m_max_cluster_cnt, 0);
      uint64_t occ = 0;
      for (uint64_t cid = 0; cid < m_max_cluster_cnt; ++cid)
      {
        const uint64_t base = base_index(cid, m_cluster_size);
        bool occupied = false;
        for (uint64_t j = 0; j < m_cluster_size; ++j)
        {
          if (m_cluster_ids[base + j] != kEmptyId)
          {
            occupied = true;
            break;
          }
        }
        m_present[cid] = occupied ? 1 : 0;
        if (occupied)
          ++occ;
      }
      m_cluster_cnt.store(occ, std::memory_order_relaxed);
      return 0;
    }

    int32_t ClusterExtra<int32_t>::getClusterIDsNormsAddr(
        uint64_t cluster_id,
        std::vector<uint64_t> &list_ids,
        std::vector<int32_t> &list_norms)
    {
      if (cluster_id >= m_max_cluster_cnt)
        return -1;

      list_ids.resize(m_cluster_size);
      if (is_L2(m_dis_type))
      {
        list_norms.resize(m_cluster_size);
      }
      else
      {
        list_norms.clear();
        list_norms.shrink_to_fit();
      }

      const uint64_t base = base_index(cluster_id, m_cluster_size);

      std::copy(m_cluster_ids.data() + base,
                m_cluster_ids.data() + base + m_cluster_size, list_ids.data());
      if (is_L2(m_dis_type))
      {
        std::copy(m_cluster_norms.data() + base,
                  m_cluster_norms.data() + base + m_cluster_size,
                  list_norms.data());
      }
      else
      {
        list_norms.clear();
        list_norms.shrink_to_fit();
      }
      return 0;
    }

    int32_t ClusterExtra<int32_t>::putClusterIDsNorms(
        uint64_t cluster_id, const std::vector<uint64_t> &list_ids,
        const std::vector<int32_t> &list_norms)
    {
      if (cluster_id >= m_max_cluster_cnt)
        return -1;
      if (list_ids.size() != m_cluster_size)
        return -2;
      if (is_L2(m_dis_type))
      {
        if (list_norms.size() != m_cluster_size)
          return -3;
      }
      const uint64_t base = base_index(cluster_id, m_cluster_size);

      std::copy(list_ids.data(), list_ids.data() + m_cluster_size,
                m_cluster_ids.data() + base);
      if (is_L2(m_dis_type))
      {
        std::copy(list_norms.data(), list_norms.data() + m_cluster_size,
                  m_cluster_norms.data() + base);
      }

      if (m_present[cluster_id] == 0)
      {
        m_present[cluster_id] = 1;
        m_cluster_cnt.fetch_add(1, std::memory_order_relaxed);
      }
      return 0;
    }

    int32_t ClusterExtra<int32_t>::getClusterIDsNormsBatch(
        const std::vector<uint64_t> &cluster_ids, std::vector<uint64_t> &list_ids,
        std::vector<int32_t> &list_norms)
    {
      if (cluster_ids.empty())
      {
        list_ids.clear();
        list_ids.shrink_to_fit();
        list_norms.clear();
        list_norms.shrink_to_fit();
        return 0;
      }

      for (auto cid : cluster_ids)
      {
        if (cid >= m_max_cluster_cnt)
          return -1;
      }

      const size_t n = cluster_ids.size();
      list_ids.resize(n * m_cluster_size);
      if (is_L2(m_dis_type))
      {
        list_norms.resize(n * m_cluster_size);
      }
      else
      {
        list_norms.clear();
        list_norms.shrink_to_fit();
      }

      for (size_t i = 0; i < n; ++i)
      {
        uint64_t cid = cluster_ids[i];
        const uint64_t base_src = base_index(cid, m_cluster_size);
        const uint64_t base_dst = static_cast<uint64_t>(i * m_cluster_size);

        std::copy(m_cluster_ids.data() + base_src,
                  m_cluster_ids.data() + base_src + m_cluster_size,
                  list_ids.data() + base_dst);
        if (is_L2(m_dis_type))
        {
          std::copy(m_cluster_norms.data() + base_src,
                    m_cluster_norms.data() + base_src + m_cluster_size,
                    list_norms.data() + base_dst);
        }
      }
      return 0;
    }

    int32_t ClusterExtra<int32_t>::putClusterIDsNormsBatch(
        const std::vector<uint64_t> &cluster_ids,
        const std::vector<uint64_t> &list_ids,
        const std::vector<int32_t> &list_norms)
    {
      if (cluster_ids.empty())
        return 0;

      for (auto cid : cluster_ids)
      {
        if (cid >= m_max_cluster_cnt)
          return -1;
      }
      const size_t n = cluster_ids.size();
      if (list_ids.size() != n * m_cluster_size)
        return -2;
      if (is_L2(m_dis_type))
      {
        if (list_norms.size() != n * m_cluster_size)
          return -3;
      }

      for (size_t i = 0; i < n; ++i)
      {
        uint64_t cid = cluster_ids[i];
        const uint64_t base_dst = base_index(cid, m_cluster_size);
        const uint64_t base_src = static_cast<uint64_t>(i * m_cluster_size);

        std::copy(list_ids.data() + base_src,
                  list_ids.data() + base_src + m_cluster_size,
                  m_cluster_ids.data() + base_dst);
        if (is_L2(m_dis_type))
        {
          std::copy(list_norms.data() + base_src,
                    list_norms.data() + base_src + m_cluster_size,
                    m_cluster_norms.data() + base_dst);
        }
      }

      std::unordered_set<uint64_t> seen;
      seen.reserve(n * 2);
      uint64_t inc = 0;
      for (auto cid : cluster_ids)
      {
        if (seen.insert(cid).second)
        {
          if (m_present[cid] == 0)
          {
            m_present[cid] = 1;
            ++inc;
          }
        }
      }
      if (inc)
        m_cluster_cnt.fetch_add(inc, std::memory_order_relaxed);
      return 0;
    }

  } // namespace runtime
} // namespace minihypervec
