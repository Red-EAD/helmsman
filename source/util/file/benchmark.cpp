#include "util/file/benchmark.hpp"

namespace minihypervec
{
  namespace util
  {
    namespace
    {
      using json = nlohmann::json;

      std::string getStringWithFallback(const json &j,
                                        const std::string &primary_key,
                                        const std::string &fallback_key)
      {
        if (j.contains(primary_key))
        {
          return j.at(primary_key).get<std::string>();
        }
        if (!fallback_key.empty() && j.contains(fallback_key))
        {
          return j.at(fallback_key).get<std::string>();
        }
        throw std::runtime_error("missing required key: " + primary_key);
      }

      uint32_t getUInt32WithFallback(const json &j,
                                     const std::string &primary_key,
                                     const std::string &fallback_key)
      {
        if (j.contains(primary_key))
        {
          return j.at(primary_key).get<uint32_t>();
        }
        if (!fallback_key.empty() && j.contains(fallback_key))
        {
          return j.at(fallback_key).get<uint32_t>();
        }
        throw std::runtime_error("missing required key: " + primary_key);
      }

      void fillBaseSearchParamFromJson(const json &j,
                                       minihypervec::index::SearchParam &out)
      {
        if (j.contains("topk_value"))
        {
          out.topk_value = j.at("topk_value").get<uint32_t>();
        }
        else if (j.contains("topk"))
        {
          out.topk_value = j.at("topk").get<uint32_t>();
        }
        else if (j.contains("k"))
        {
          out.topk_value = j.at("k").get<uint32_t>();
        }
      }

      std::unique_ptr<minihypervec::index::SearchParam> parseSearchParamByIndexType(
          const json &j, minihypervec::collection::IndexType index_type)
      {
        using minihypervec::collection::IndexType;
        using minihypervec::index::HnswSearchParam;
        using minihypervec::index::MiniHyperVecConstSearchParam;
        using minihypervec::index::SearchParam;

        switch (index_type)
        {
        case IndexType::HNSW:
        {
          auto sp = std::make_unique<HnswSearchParam>();
          fillBaseSearchParamFromJson(j, *sp);
          if (j.contains("max_visits"))
          {
            sp->max_visits = j.at("max_visits").get<uint32_t>();
          }
          return sp;
        }
        case IndexType::HV_CONST:
        {
          auto sp = std::make_unique<MiniHyperVecConstSearchParam>();
          fillBaseSearchParamFromJson(j, *sp);
          sp->cluster_nprobe =
              getUInt32WithFallback(j, "cluster_nprobe", "nprobe");

          minihypervec::collection::IndexType centroid_index_type = IndexType::UNKNOWN;
          if (j.contains("centroid_index_type"))
          {
            centroid_index_type = minihypervec::collection::indexTypeFromString(
                j.at("centroid_index_type").get<std::string>());
          }

          if (j.contains("centroid_search_param"))
          {
            auto ct_type = centroid_index_type == IndexType::UNKNOWN
                               ? IndexType::HNSW
                               : centroid_index_type;
            sp->centroid_search_param =
                parseSearchParamByIndexType(j.at("centroid_search_param"),
                                            ct_type);
          }
          else
          {
            sp->centroid_search_param = std::make_unique<SearchParam>();
            sp->centroid_search_param->topk_value = sp->cluster_nprobe;
          }

          if (sp->centroid_search_param &&
              sp->centroid_search_param->topk_value == 10ul)
          {
            sp->centroid_search_param->topk_value = sp->cluster_nprobe;
          }
          return sp;
        }
        default:
          std::cerr << "Unsupported index type in benchmark search_param: "
                    << static_cast<uint32_t>(index_type) << std::endl;
          return nullptr;
        }
      }

    } // namespace

    int32_t loadQueryInt8(const std::string &file_path,
                          std::vector<std::vector<int8_t>> &out_vecs)
    {
      util::Dataset<int8_t> query_dataset(file_path);
      if (query_dataset.getDataBase() == nullptr)
      {
        std::cerr << "Error: failed to map query file " << file_path << std::endl;
        return -1;
      }
      uint32_t dim = query_dataset.dim;
      uint64_t nq = query_dataset.total_cnt;
      out_vecs.resize(nq, std::vector<int8_t>(dim));
      const int8_t *query_base = query_dataset.getDataBase();
      for (uint64_t i = 0; i < nq; i++)
      {
        const int8_t *cur_query = query_base + i * dim;
        std::memcpy(out_vecs[i].data(), cur_query, dim * sizeof(int8_t));
      }
      return 0;
    }

    int32_t loadGroundTruth(const std::string &file_path,
                            std::vector<std::vector<uint64_t>> &out_ids,
                            std::vector<std::vector<float>> &out_dists)
    {
      minihypervec::util::GtReader gt_hdr(file_path);
      const uint32_t max_k_truth = gt_hdr.k();
      std::vector<uint32_t> queries_ids;
      queries_ids.resize(gt_hdr.num_queries());
      std::iota(queries_ids.begin(), queries_ids.end(), 0);
      return gt_hdr.gettGroundTruthBatch(queries_ids, max_k_truth, out_ids,
                                         out_dists);
    }

    int32_t loadBenchMarkConfig(const std::string &file_path,
                                benchmark::benchmark_param &param)
    {
      try
      {
        std::string content;
        int32_t rc = util::read_file_to_string(file_path, content);
        if (rc != 0)
        {
          std::cerr << "Failed to read benchmark config: " << file_path
                    << ", rc=" << rc << std::endl;
          return rc;
        }
        json j = json::parse(content);

        param.test_collection_name =
            getStringWithFallback(j, "test_collection_name", "collection_name");
        param.test_query_path =
            getStringWithFallback(j, "test_query_path", "query_path");
        param.test_groundtruth_path = getStringWithFallback(
            j, "test_groundtruth_path", "test_ground_truth_path");
        if (param.test_groundtruth_path.empty() && j.contains("groundtruth_path"))
        {
          param.test_groundtruth_path = j.at("groundtruth_path").get<std::string>();
        }

        const std::string index_type_str =
            getStringWithFallback(j, "index_type", "index");
        param.index_type = minihypervec::collection::indexTypeFromString(index_type_str);
        if (param.index_type == minihypervec::collection::IndexType::UNKNOWN)
        {
          std::cerr << "Unknown index_type in benchmark config: " << index_type_str
                    << std::endl;
          return -1;
        }

        const std::string vec_type_str =
            getStringWithFallback(j, "vec_type", "data_type");
        param.data_type = minihypervec::collection::vecTypeFromString(vec_type_str);
        if (param.data_type == minihypervec::collection::VecType::UNKNOWN)
        {
          std::cerr << "Unknown vec_type/data_type in benchmark config: "
                    << vec_type_str << std::endl;
          return -1;
        }

        if (!j.contains("search_param"))
        {
          std::cerr << "No search_param found in benchmark config: " << file_path
                    << std::endl;
          return -1;
        }
        param.search_param =
            parseSearchParamByIndexType(j.at("search_param"), param.index_type);
        if (!param.search_param)
        {
          std::cerr << "Failed to parse search_param in benchmark config: "
                    << file_path << std::endl;
          return -1;
        }
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error---loadBenchMarkConfig: JSON parse/load failed: "
                  << e.what() << std::endl;
        return -1;
      }
      return 0;
    }
  } // namespace util
} // namespace minihypervec