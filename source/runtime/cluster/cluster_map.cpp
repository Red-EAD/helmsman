#include "runtime/cluster/cluster_map.hpp"

namespace minihypervec
{
  namespace runtime
  {

    ClusterStripe::ClusterStripe(uint32_t nvme_id, uint64_t lba_id)
        : nvme_id_(nvme_id), lba_id_(lba_id) {}
    bool ClusterStripe::operator==(const ClusterStripe &o) const
    {
      return nvme_id_ == o.nvme_id_ && lba_id_ == o.lba_id_;
    }

    namespace
    {
      constexpr uint32_t kEmptyNvmeId = std::numeric_limits<uint32_t>::max();

      inline bool IsEmpty(const ClusterStripe &cs)
      {
        return cs.nvme_id_ == kEmptyNvmeId;
      }
      inline ClusterStripe EmptyStripe() { return ClusterStripe{kEmptyNvmeId, 0}; }

      struct FileHeader
      {
        uint32_t magic;    // 'CMAP' = 0x50414D43
        uint32_t version;  // 1
        uint64_t one_page; // m_one_page_cnt
        uint64_t max_cnt;  // m_max_cluster_cnt
        uint64_t rr_index; // m_rr_index_dev_
      };

      struct RangeRec
      {
        uint32_t nvme_id;
        uint64_t start_lba;
        uint64_t count;
      };

      constexpr uint32_t kMagic = 0x50414D43; // 'C' 'M' 'A' 'P'

      static void pack_ranges_for_save(
          const std::unordered_map<uint32_t, std::vector<uint64_t>> &free_by_dev,
          uint64_t one_page, std::vector<RangeRec> &out)
      {
        out.clear();
        if (one_page == 0)
          return;

        for (const auto &kv : free_by_dev)
        {
          uint32_t dev = kv.first;
          const auto &v = kv.second;
          if (v.empty())
            continue;

          std::vector<uint64_t> s = v;
          std::sort(s.begin(), s.end());

          uint64_t run_start = s[0];
          uint64_t run_count = 1;
          for (size_t i = 1; i < s.size(); ++i)
          {
            if (s[i] == s[i - 1] + one_page)
            {
              ++run_count;
            }
            else
            {
              out.push_back(RangeRec{dev, run_start, run_count});
              run_start = s[i];
              run_count = 1;
            }
          }
          out.push_back(RangeRec{dev, run_start, run_count});
        }
      }

      static void unpack_ranges_on_load(
          const std::vector<RangeRec> &ranges, uint64_t one_page,
          std::unordered_map<uint32_t, std::vector<uint64_t>> &free_by_dev,
          uint64_t &free_cnt)
      {
        free_by_dev.clear();
        free_cnt = 0;
        if (one_page == 0)
          return;

        for (const auto &r : ranges)
        {
          auto &vec = free_by_dev[r.nvme_id];
          vec.reserve(vec.size() + static_cast<size_t>(r.count));
          for (uint64_t i = 0; i < r.count; ++i)
          {
            vec.push_back(r.start_lba + i * one_page);
          }
          free_cnt += r.count;
        }
      }

      static bool pop_one_stripe_rr(
          std::unordered_map<uint32_t, std::vector<uint64_t>> &free_by_dev,
          std::vector<uint32_t> &rr_devices, uint64_t &rr_index, uint64_t &free_cnt,
          ClusterStripe &out)
      {
        if (rr_devices.empty() || free_cnt == 0)
          return false;

        const size_t n = rr_devices.size();
        for (size_t step = 0; step < n; ++step)
        {
          uint64_t idx = (rr_index + step) % n;
          uint32_t dev = rr_devices[idx];
          auto it = free_by_dev.find(dev);
          if (it == free_by_dev.end() || it->second.empty())
            continue;

          uint64_t lba = it->second.back();
          it->second.pop_back();
          if (it->second.empty())
          {
          }
          rr_index = (idx + 1) % n;
          --free_cnt;
          out = ClusterStripe{dev, lba};
          return true;
        }
        return false;
      }

    } // namespace

    void ClusterMap::init(uint64_t max_cluster_cnt, uint64_t one_cluster_page_cnt)
    {
      m_max_cluster_cnt = max_cluster_cnt;
      m_one_page_cnt = one_cluster_page_cnt;
      m_cluster_cnt = 0;

      m_map_impl.assign(m_max_cluster_cnt, EmptyStripe());

      m_free_by_dev_.clear();
      m_free_cnt_ = 0;
      m_rr_devices_.clear();
      m_rr_index_dev_ = 0;
    }

    void ClusterMap::allocateChunks(const std::vector<nvme::Chunk> &chunks)
    {
      if (m_one_page_cnt == 0)
        return;

      for (const auto &ck : chunks)
      {
        if (ck.page_count < m_one_page_cnt)
          continue;

        const uint64_t stripes = ck.page_count / m_one_page_cnt;
        auto &bucket = m_free_by_dev_[ck.nvme_id];
        bucket.reserve(bucket.size() + static_cast<size_t>(stripes));
        for (uint64_t s = 0; s < stripes; ++s)
        {
          const uint64_t lba = ck.start_page + s * m_one_page_cnt;
          bucket.push_back(lba);
        }
        m_free_cnt_ += stripes;

        if (std::find(m_rr_devices_.begin(), m_rr_devices_.end(), ck.nvme_id) ==
            m_rr_devices_.end())
        {
          m_rr_devices_.push_back(ck.nvme_id);
        }
      }
    }

    uint64_t ClusterMap::getClusterCnt() { return m_cluster_cnt; }

    int32_t ClusterMap::putClusterStripe(uint64_t cluster_id, ClusterStripe &pos,
                                         bool lock_inside)
    {
      tbb::spin_rw_mutex::scoped_lock lock;
      if (lock_inside)
      {
        lock.acquire(m_inside_rw_mutex, true);
      }
      if (m_one_page_cnt == 0 || m_max_cluster_cnt == 0)
        return nvme::kInvalidArg;
      if (cluster_id >= m_max_cluster_cnt)
        return nvme::kInvalidArg;
      if (!IsEmpty(m_map_impl[cluster_id]))
        return nvme::kInvalidArg;

      ClusterStripe chosen;
      if (!pop_one_stripe_rr(m_free_by_dev_, m_rr_devices_, m_rr_index_dev_,
                             m_free_cnt_, chosen))
      {
        return nvme::kNoSpace;
      }

      m_map_impl[cluster_id] = chosen;
      ++m_cluster_cnt;
      pos = chosen;
      return nvme::kOk;
    }

    int32_t ClusterMap::getClusterStripe(uint64_t cluster_id, ClusterStripe &pos,
                                         bool lock_inside)
    {
      tbb::spin_rw_mutex::scoped_lock lock;
      if (lock_inside)
      {
        lock.acquire(m_inside_rw_mutex, false);
      }
      if (cluster_id >= m_max_cluster_cnt)
        return nvme::kInvalidArg;
      const auto &s = m_map_impl[cluster_id];
      if (IsEmpty(s))
        return nvme::kDeviceMissing;
      pos = s;
      return nvme::kOk;
    }

    int32_t ClusterMap::saveClusterMap(const std::string &path)
    {
      std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
      if (!ofs)
        throw std::runtime_error("saveClusterMap: open failed: " + path);

      FileHeader hdr{kMagic, 1, m_one_page_cnt, m_max_cluster_cnt, m_rr_index_dev_};
      ofs.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));

      uint64_t bound_cnt = m_cluster_cnt;
      ofs.write(reinterpret_cast<const char *>(&bound_cnt), sizeof(bound_cnt));
      for (uint64_t cid = 0, emitted = 0;
           cid < m_map_impl.size() && emitted < bound_cnt; ++cid)
      {
        const auto &s = m_map_impl[cid];
        if (IsEmpty(s))
          continue;
        ofs.write(reinterpret_cast<const char *>(&cid), sizeof(cid));
        ofs.write(reinterpret_cast<const char *>(&s.nvme_id_), sizeof(s.nvme_id_));
        ofs.write(reinterpret_cast<const char *>(&s.lba_id_), sizeof(s.lba_id_));
        ++emitted;
      }

      std::vector<RangeRec> ranges;
      pack_ranges_for_save(m_free_by_dev_, m_one_page_cnt, ranges);

      uint64_t range_cnt = ranges.size();
      ofs.write(reinterpret_cast<const char *>(&range_cnt), sizeof(range_cnt));
      for (const auto &r : ranges)
      {
        ofs.write(reinterpret_cast<const char *>(&r.nvme_id), sizeof(r.nvme_id));
        ofs.write(reinterpret_cast<const char *>(&r.start_lba), sizeof(r.start_lba));
        ofs.write(reinterpret_cast<const char *>(&r.count), sizeof(r.count));
      }

      uint64_t dev_list_cnt = m_rr_devices_.size();
      ofs.write(reinterpret_cast<const char *>(&dev_list_cnt), sizeof(dev_list_cnt));
      for (uint32_t dev : m_rr_devices_)
      {
        ofs.write(reinterpret_cast<const char *>(&dev), sizeof(dev));
      }
      return 0;
    }

    int32_t ClusterMap::loadClusterMap(const std::string &path)
    {
      std::ifstream ifs(path, std::ios::binary);
      if (!ifs)
        throw std::runtime_error("loadClusterMap: open failed: " + path);

      FileHeader hdr{};
      ifs.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
      if (hdr.magic != kMagic || hdr.version != 1)
      {
        throw std::runtime_error("loadClusterMap: bad header");
      }

      if (m_max_cluster_cnt == 0 || m_one_page_cnt == 0)
      {
        m_max_cluster_cnt = hdr.max_cnt;
        m_one_page_cnt = hdr.one_page;
        m_map_impl.assign(m_max_cluster_cnt, EmptyStripe());
      }
      else if (m_max_cluster_cnt != hdr.max_cnt ||
               m_one_page_cnt != hdr.one_page)
      {
        throw std::runtime_error("loadClusterMap: incompatible config");
      }
      m_rr_index_dev_ = hdr.rr_index;

      uint64_t bound_cnt = 0;
      ifs.read(reinterpret_cast<char *>(&bound_cnt), sizeof(bound_cnt));
      std::fill(m_map_impl.begin(), m_map_impl.end(), EmptyStripe());
      m_cluster_cnt = 0;

      for (uint64_t i = 0; i < bound_cnt; ++i)
      {
        uint64_t cid;
        uint32_t dev;
        uint64_t lba;
        ifs.read(reinterpret_cast<char *>(&cid), sizeof(cid));
        ifs.read(reinterpret_cast<char *>(&dev), sizeof(dev));
        ifs.read(reinterpret_cast<char *>(&lba), sizeof(lba));
        if (cid < m_map_impl.size())
        {
          m_map_impl[cid] = ClusterStripe{dev, lba};
          ++m_cluster_cnt;
        }
      }

      uint64_t range_cnt = 0;
      ifs.read(reinterpret_cast<char *>(&range_cnt), sizeof(range_cnt));
      std::vector<RangeRec> ranges(range_cnt);
      for (uint64_t i = 0; i < range_cnt; ++i)
      {
        ifs.read(reinterpret_cast<char *>(&ranges[i].nvme_id),
                 sizeof(ranges[i].nvme_id));
        ifs.read(reinterpret_cast<char *>(&ranges[i].start_lba),
                 sizeof(ranges[i].start_lba));
        ifs.read(reinterpret_cast<char *>(&ranges[i].count),
                 sizeof(ranges[i].count));
      }
      unpack_ranges_on_load(ranges, m_one_page_cnt, m_free_by_dev_, m_free_cnt_);

      uint64_t dev_list_cnt = 0;
      ifs.read(reinterpret_cast<char *>(&dev_list_cnt), sizeof(dev_list_cnt));
      m_rr_devices_.resize(static_cast<size_t>(dev_list_cnt));
      for (uint64_t i = 0; i < dev_list_cnt; ++i)
      {
        ifs.read(reinterpret_cast<char *>(&m_rr_devices_[i]), sizeof(uint32_t));
      }
      return 0;
    }

    int32_t ClusterMap::putClusterStripeBatch(
        const std::vector<uint64_t> &cluster_id, std::vector<ClusterStripe> &pos,
        bool lock_inside)
    {
      tbb::spin_rw_mutex::scoped_lock lock;
      if (lock_inside)
      {
        lock.acquire(m_inside_rw_mutex, true);
      }
      if (m_one_page_cnt == 0 || m_max_cluster_cnt == 0)
        return nvme::kInvalidArg;
      if (cluster_id.empty())
      {
        pos.clear();
        return nvme::kOk;
      }

      std::unordered_set<uint64_t> seen;
      for (auto cid : cluster_id)
      {
        if (cid >= m_max_cluster_cnt)
          return nvme::kInvalidArg;
        if (!IsEmpty(m_map_impl[cid]))
          return nvme::kInvalidArg;
        if (!seen.insert(cid).second)
          return nvme::kInvalidArg;
      }
      if (m_free_cnt_ < cluster_id.size())
        return nvme::kNoSpace;

      const uint64_t old_rr = m_rr_index_dev_;
      const uint64_t old_free_cnt = m_free_cnt_;

      std::vector<ClusterStripe> picked;
      picked.reserve(cluster_id.size());
      for (size_t i = 0; i < cluster_id.size(); ++i)
      {
        ClusterStripe s{};
        if (!pop_one_stripe_rr(m_free_by_dev_, m_rr_devices_, m_rr_index_dev_,
                               m_free_cnt_, s))
        {
          for (size_t j = 0; j < picked.size(); ++j)
          {
            const auto &r = picked[picked.size() - 1 - j];
            m_free_by_dev_[r.nvme_id_].push_back(r.lba_id_);
          }
          m_free_cnt_ = old_free_cnt;
          m_rr_index_dev_ = old_rr;
          return nvme::kNoSpace;
        }
        picked.push_back(s);
      }

      pos.resize(cluster_id.size());
      for (size_t i = 0; i < cluster_id.size(); ++i)
      {
        m_map_impl[cluster_id[i]] = picked[i];
        pos[i] = picked[i];
      }
      m_cluster_cnt += static_cast<uint64_t>(cluster_id.size());
      return nvme::kOk;
    }

    int32_t ClusterMap::getClusterStripeBatch(
        const std::vector<uint64_t> &cluster_id, std::vector<ClusterStripe> &pos,
        bool lock_inside)
    {
      tbb::spin_rw_mutex::scoped_lock lock;
      if (lock_inside)
      {
        lock.acquire(m_inside_rw_mutex, false);
      }
      if (cluster_id.empty())
      {
        pos.clear();
        return nvme::kOk;
      }

      for (auto cid : cluster_id)
      {
        if (cid >= m_max_cluster_cnt)
          return nvme::kInvalidArg;
        if (IsEmpty(m_map_impl[cid]))
          return nvme::kDeviceMissing;
      }

      pos.resize(cluster_id.size());
      for (size_t i = 0; i < cluster_id.size(); ++i)
      {
        pos[i] = m_map_impl[cluster_id[i]];
      }
      return nvme::kOk;
    }

  } // namespace runtime
} // namespace minihypervec