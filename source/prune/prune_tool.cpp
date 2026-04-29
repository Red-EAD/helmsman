#include "prune/prune_tool.hpp"

namespace minihypervec
{
  namespace prune
  {
    int32_t PruningTool<int32_t>::pruneScan(
        const index::SearchParam &search_param, const std::vector<int8_t> &query,
        const std::vector<std::pair<uint64_t, int32_t>> &centroid_res,
        uint32_t &out_probe)
    {
      std::cerr << "PruningTool<int32_t>::pruneScan: not implemented, need real "
                   "implementation"
                << std::endl;
      return -1;
    }

    int32_t PruningToolNaive<int32_t>::pruneScan(
        const index::SearchParam &search_param, const std::vector<int8_t> &query,
        const std::vector<std::pair<uint64_t, int32_t>> &centroid_res,
        uint32_t &out_probe)
    {
      out_probe = std::min(max_probe, static_cast<uint32_t>(centroid_res.size()));
      return 0;
    }

  } // namespace prune
} // namespace minihypervec