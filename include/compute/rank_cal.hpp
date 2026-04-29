#pragma once

#include "root.hpp"

namespace minihypervec
{
  namespace compute
  {
    struct RankPair
    {
      uint32_t pos;
      uint64_t rank_id;
    };

    template <typename T>
    class SearchCPUFuncL2;

    template <>
    class SearchCPUFuncL2<int32_t>
    {
    public:
      struct cpu_cmp_noaxpy
      {
        cpu_cmp_noaxpy(const int32_t *distances_, int32_t q_norm_,
                       int32_t *h_norms_)
            : distances(distances_), q_norm(q_norm_), h_norms(h_norms_) {}
        const int32_t *distances;
        int32_t q_norm;
        const int32_t *h_norms;
        bool operator()(const RankPair &lhs, const RankPair &rhs) const
        {
          return -2 * distances[lhs.pos] + q_norm + h_norms[lhs.pos] <
                 -2 * distances[rhs.pos] + q_norm + h_norms[rhs.pos];
        }
      };

    public:
      static void prepareRankPairs(const std::vector<uint64_t> &list_ids,
                                   std::vector<RankPair> &rank_addr);

      static void rankTopK(const int32_t *dis_addr, int32_t q_norm,
                           const std::vector<int32_t> &h_norms, uint64_t total_vec,
                           std::vector<RankPair> &rank_addr, uint64_t k,
                           std::vector<std::pair<uint64_t, int32_t>> &res);
    };

  } // namespace compute
} // namespace minihypervec