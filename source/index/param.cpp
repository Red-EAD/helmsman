#include "index/params.hpp"

namespace minihypervec
{
  namespace index
  {

    void BuildParam::printBuildParam() const
    {
      std::cout << "BuildParam: " << std::endl;
      std::cout << "  Metric: " << collection::stringFromDisType(metric)
                << std::endl;
      std::cout << "  Dimension: " << dim << std::endl;
    }

    void HnswBuildParam::printBuildParam() const
    {
      std::cout << "HnswBuildParam: " << std::endl;
      BuildParam::printBuildParam();
      std::cout << "  Inner M: " << inner_M << std::endl;
      std::cout << "  Build ef: " << build_ef << std::endl;
      std::cout << "  Search ef: " << search_ef << std::endl;
      std::cout << "  Max elements: " << max_elements << std::endl;
    }

    void MiniHyperVecConstBuildParam::printBuildParam() const
    {
      std::cout << "MiniHyperVecConstBuildParam: " << std::endl;
      BuildParam::printBuildParam();
      std::cout << "  Centroid num: " << centroid_num << std::endl;
      std::cout << "  Cluster size: " << cluster_size << std::endl;
      std::cout << "  Max elements: " << max_elements << std::endl;
      std::cout << "  Centroid index type: "
                << static_cast<uint32_t>(centroid_index_type) << std::endl;
      if (centroid_build_param)
      {
        std::cout << "  Centroid build param: " << std::endl;
        centroid_build_param->printBuildParam();
      }
    }

    json BuildParam::to_json() const
    {
      json j;
      j["metric"] = collection::stringFromDisType(metric);
      j["dim"] = dim;
      return j;
    }

    void BuildParam::from_json(const json &j)
    {
      metric = collection::disTypeFromString(j.at("metric").get<std::string>());
      dim = j.at("dim").get<uint32_t>();
    }

    json HnswBuildParam::to_json() const
    {
      json j = BuildParam::to_json();
      j["inner_M"] = inner_M;
      j["build_ef"] = build_ef;
      j["search_ef"] = search_ef;
      j["max_elements"] = max_elements;
      return j;
    }

    void HnswBuildParam::from_json(const json &j)
    {
      BuildParam::from_json(j);
      inner_M = j.at("inner_M").get<uint32_t>();
      build_ef = j.at("build_ef").get<uint32_t>();
      search_ef = j.at("search_ef").get<uint32_t>();
      max_elements = j.at("max_elements").get<uint64_t>();
    }

    json MiniHyperVecConstBuildParam::to_json() const
    {
      json j = BuildParam::to_json();
      j["centroid_num"] = centroid_num;
      j["cluster_size"] = cluster_size;
      j["max_elements"] = max_elements;
      j["centroid_index_type"] =
          collection::stringFromIndexType(centroid_index_type);
      if (centroid_build_param)
      {
        j["centroid_build_param"] = centroid_build_param->to_json();
      }
      return j;
    }

    void MiniHyperVecConstBuildParam::from_json(const json &j)
    {
      BuildParam::from_json(j);
      centroid_num = j.at("centroid_num").get<uint32_t>();
      cluster_size = j.at("cluster_size").get<uint32_t>();
      max_elements = j.at("max_elements").get<uint64_t>();
      centroid_index_type = collection::indexTypeFromString(
          j.at("centroid_index_type").get<std::string>());
      if (j.contains("centroid_build_param"))
      {
        switch (centroid_index_type)
        {
        case collection::IndexType::HNSW:
          centroid_build_param = std::make_unique<HnswBuildParam>();
          centroid_build_param->from_json(j.at("centroid_build_param"));
          break;
        default:
          std::cerr << "Unsupported centroid index type in from_json: "
                    << static_cast<uint32_t>(centroid_index_type) << std::endl;
          break;
        }
      }
      else
      {
        std::cerr << "No centroid_build_param found in from_json." << std::endl;
      }
    }

    std::unique_ptr<BuildParam> BuildParam::clone() const
    {
      return std::make_unique<BuildParam>(*this);
    }

    std::unique_ptr<BuildParam> HnswBuildParam::clone() const
    {
      return std::make_unique<HnswBuildParam>(*this);
    }

    MiniHyperVecConstBuildParam::MiniHyperVecConstBuildParam(
        const MiniHyperVecConstBuildParam &o)
        : BuildParam(o),
          centroid_num(o.centroid_num),
          cluster_size(o.cluster_size),
          max_elements(o.max_elements),
          centroid_index_type(o.centroid_index_type)
    {
      centroid_build_param =
          o.centroid_build_param ? o.centroid_build_param->clone() : nullptr;
    }
    MiniHyperVecConstBuildParam &MiniHyperVecConstBuildParam::operator=(
        const MiniHyperVecConstBuildParam &o)
    {
      if (this != &o)
      {
        BuildParam::operator=(o);
        centroid_num = o.centroid_num;
        cluster_size = o.cluster_size;
        max_elements = o.max_elements;
        centroid_index_type = o.centroid_index_type;
        centroid_build_param =
            o.centroid_build_param ? o.centroid_build_param->clone() : nullptr;
      }
      return *this;
    }
    std::unique_ptr<BuildParam> MiniHyperVecConstBuildParam::clone() const
    {
      return std::make_unique<MiniHyperVecConstBuildParam>(*this);
    }

  } // namespace index
} // namespace minihypervec