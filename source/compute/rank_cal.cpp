#include "compute/rank_cal.hpp"

namespace minihypervec
{
  namespace compute
  {
    static inline uint64_t take_block(uint64_t sorted_idx, uint64_t total_vec,
                                      uint64_t k2)
    {
      const uint64_t remain = total_vec - sorted_idx;
      return remain < k2 ? remain : k2;
    }

    void SearchCPUFuncL2<int32_t>::prepareRankPairs(
        const std::vector<uint64_t> &list_ids, std::vector<RankPair> &rank_addr)
    {
      const uint64_t total_vec = list_ids.size();
      rank_addr.resize(total_vec);
      for (uint64_t i = 0; i < total_vec; ++i)
      {
        rank_addr[i].pos = static_cast<uint32_t>(i);
        rank_addr[i].rank_id = list_ids[i];
      }
    }

    void SearchCPUFuncL2<int32_t>::rankTopK(
        const int32_t *dis_addr, int32_t q_norm,
        const std::vector<int32_t> &h_norms, uint64_t total_vec,
        std::vector<RankPair> &rank_addr, uint64_t k,
        std::vector<std::pair<uint64_t, int32_t>> &res)
    {
      res.clear();
      if (k == 0 || total_vec == 0)
        return;

      std::unordered_set<uint64_t> seen;
      seen.reserve(2 * k);

      uint64_t unique_cnt = 0;
      uint64_t sorted_idx = 0;

      res.resize(static_cast<size_t>(k));

      while (unique_cnt < k && sorted_idx < total_vec)
      {
        const uint64_t take = take_block(sorted_idx, total_vec, 2 * k);
        if (take == 0)
          break;

        std::partial_sort(
            rank_addr.begin() + static_cast<std::ptrdiff_t>(sorted_idx),
            rank_addr.begin() + static_cast<std::ptrdiff_t>(sorted_idx + take),
            rank_addr.end(),
            cpu_cmp_noaxpy(dis_addr, q_norm, const_cast<int32_t *>(h_norms.data())));

        const uint64_t end_i = std::min<uint64_t>(sorted_idx + take, total_vec);
        for (uint64_t i = sorted_idx; i < end_i && unique_cnt < k; ++i)
        {
          const uint64_t cur_id = rank_addr[i].rank_id;
          const uint32_t cur_pos = rank_addr[i].pos;
          if (!seen.count(cur_id))
          {
            seen.insert(cur_id);
            const int32_t l2 = q_norm + h_norms[cur_pos] - 2 * dis_addr[cur_pos];
            res[static_cast<size_t>(unique_cnt)] = std::make_pair(cur_id, l2);
            ++unique_cnt;
          }
        }
        sorted_idx += take;
      }

      if (unique_cnt < k)
      {
        std::cerr << "[L2<int32_t>::rankTopK] unique ids found = " << unique_cnt
                  << " < k = " << k << "\n";
      }
      res.resize(static_cast<size_t>(unique_cnt));
      return;
    }

  } // namespace compute
} // namespace minihypervec