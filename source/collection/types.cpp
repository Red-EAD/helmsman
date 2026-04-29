#include "collection/types.hpp"

namespace minihypervec {
namespace collection {
std::string stringFromDisType(collection::DisType d) {
  switch (d) {
    case collection::DisType::UNKNOWN:
      return "UNKNOWN";
    case collection::DisType::L2:
      return "Euclidean";
    case collection::DisType::IP:
      return "InnerProduct";
    default:
      return "UNSPECIFIED";
  }
}

collection::DisType disTypeFromString(const std::string& d) {
  if (d == "UNKNOWN") return collection::DisType::UNKNOWN;
  if (d == "Euclidean") return collection::DisType::L2;
  if (d == "InnerProduct") return collection::DisType::IP;
  return collection::DisType::UNKNOWN;
}

std::string stringFromIndexType(collection::IndexType index_type) {
  switch (index_type) {
    case collection::IndexType::UNKNOWN:
      return "UNKNOWN";
    case collection::IndexType::HNSW:
      return "HNSW";
    case collection::IndexType::HV_CONST:
      return "HV_CONST";
    default:
      return "UNSPECIFIED";
  }
}

collection::IndexType indexTypeFromString(const std::string& index_type_str) {
  if (index_type_str == "UNKNOWN") return collection::IndexType::UNKNOWN;
  if (index_type_str == "HNSW") return collection::IndexType::HNSW;
  if (index_type_str == "HV_CONST") return collection::IndexType::HV_CONST;
  return collection::IndexType::UNKNOWN;
}

std::string stringFromVecType(collection::VecType vec_type) {
  switch (vec_type) {
    case collection::VecType::UNKNOWN:
      return "UNKNOWN";
    case collection::VecType::INT8:
      return "INT8";
    default:
      return "UNSPECIFIED";
  }
}

collection::VecType vecTypeFromString(const std::string& vec_type_str) {
  if (vec_type_str == "UNKNOWN") return collection::VecType::UNKNOWN;
  if (vec_type_str == "INT8") return collection::VecType::INT8;
  return collection::VecType::UNKNOWN;
}

}  // namespace collection
}  // namespace minihypervec