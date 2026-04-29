#pragma once

#include "collection/types.hpp"
#include "runtime/resource/resource.hpp"

namespace minihypervec
{

  namespace index
  {
    using json = nlohmann::json;

    struct SearchParam
    {
      uint32_t topk_value = 10ul;
      virtual ~SearchParam() = default;
    };

    struct HnswSearchParam : public SearchParam
    {
      uint32_t max_visits = 5000ul;
    };

    struct MiniHyperVecConstSearchParam : public SearchParam
    {
      std::unique_ptr<SearchParam> centroid_search_param = nullptr;
      uint32_t cluster_nprobe;
    };

    struct BuildParam
    {
      collection::DisType metric;
      uint32_t dim;
      virtual ~BuildParam() = default;
      virtual void printBuildParam() const;
      virtual json to_json() const;
      virtual void from_json(const json &j);
      virtual std::unique_ptr<BuildParam> clone() const;
    };

    struct HnswBuildParam : public BuildParam
    {
      uint32_t inner_M;
      uint32_t build_ef;
      uint32_t search_ef;
      uint64_t max_elements;
      void printBuildParam() const override;
      json to_json() const override;
      void from_json(const json &j) override;
      std::unique_ptr<BuildParam> clone() const override;
    };

    struct MiniHyperVecConstBuildParam : public BuildParam
    {
      uint32_t centroid_num;
      uint32_t cluster_size;
      uint64_t max_elements;
      collection::IndexType centroid_index_type;
      std::unique_ptr<BuildParam> centroid_build_param = nullptr;
      void printBuildParam() const override;
      json to_json() const override;
      void from_json(const json &j) override;
      MiniHyperVecConstBuildParam() = default;
      MiniHyperVecConstBuildParam(const MiniHyperVecConstBuildParam &o);
      MiniHyperVecConstBuildParam &operator=(const MiniHyperVecConstBuildParam &o);
      std::unique_ptr<BuildParam> clone() const override;
    };

  } // namespace index
} // namespace minihypervec