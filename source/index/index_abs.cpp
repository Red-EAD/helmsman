#include "index/index_abs.hpp"

namespace minihypervec
{
  namespace index
  {

    int32_t IndexAbs<int32_t>::searchKnn(
        const SearchParam &search_param, const std::vector<int8_t> &query,
        std::vector<std::pair<uint64_t, int32_t>> &res,
        resource::SearchTempResource *search_resource)
    {
      std::cerr << "IndexAbs<int32_t>::searchKnn: not implemented" << std::endl;
      return -1;
    }

    int32_t IndexAbs<int32_t>::deployIndex(
        const std::string &collection_name,
        const resource::DeployTempResource &deploy_resource)
    {
      std::cerr << "IndexAbs<int32_t>::deployIndex: not implemented" << std::endl;
      return -1;
    }

    collection::IndexType IndexAbs<int32_t>::getIndexType() const
    {
      return collection::IndexType::UNKNOWN;
    }

  } // namespace index
} // namespace minihypervec
