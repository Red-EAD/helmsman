#include "runtime/env/index_holder.hpp"

namespace minihypervec
{
  namespace runtime
  {
    IndexHolder *IndexHolder::getInstance()
    {
      static IndexHolder g_global_instance;
      return &g_global_instance;
    }

    int32_t IndexHolder::getIndex(
        const std::string &collection_name,
        std::shared_ptr<index::IndexAbs<int32_t>> &index)
    {
      if (held_index == nullptr || collection_name != this->collection_name)
      {
        std::cerr << "Error: INT8 index does not exist for collection: "
                  << collection_name << std::endl;
        return -1;
      }
      index = held_index;
      return 0;
    }

    int32_t IndexHolder::initIndex(
        const std::string &collection_name,
        const std::shared_ptr<index::IndexAbs<int32_t>> &index)
    {
      if (held_index != nullptr)
      {
        std::cerr << "Error: INT8 index already exists for collection: "
                  << collection_name << std::endl;
        return -1;
      }
      this->collection_name = collection_name;
      held_index = index;
      return 0;
    }

  } // namespace runtime
} // namespace minihypervec