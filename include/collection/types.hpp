#pragma once

#include "root.hpp"

namespace minihypervec
{
  namespace collection
  {
    enum class VecType : uint32_t
    {
      UNKNOWN = 0,
      INT8 = 1,
    };

    enum class DisType : uint32_t
    {
      UNKNOWN = 0,
      L2 = 1,
      IP = 2,
    };

    enum class IndexType : uint32_t
    {
      UNKNOWN = 0,
      HNSW = 1,
      HV_CONST = 2,
    };

    std::string stringFromDisType(collection::DisType d);
    collection::DisType disTypeFromString(const std::string &d);

    std::string stringFromIndexType(collection::IndexType index_type);
    collection::IndexType indexTypeFromString(const std::string &index_type_str);

    std::string stringFromVecType(collection::VecType vec_type);
    collection::VecType vecTypeFromString(const std::string &vec_type_str);

  } // namespace collection
} // namespace minihypervec