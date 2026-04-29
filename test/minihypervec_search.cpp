#include "runtime/env/minihypervec_env.hpp"
#include "util/file/benchmark.hpp"
#include "util/file/files_rw.hpp"

using namespace minihypervec;

namespace
{
  using json = nlohmann::json;
  enum class RecallMode : uint32_t
  {
    kId = 0,
    kDist = 1,
  };

  RecallMode parseRecallMode(std::string mode)
  {
    for (auto &c : mode)
      c = static_cast<char>(std::tolower(c));
    if (mode == "id")
      return RecallMode::kId;
    if (mode == "dist" || mode == "distance")
      return RecallMode::kDist;
    throw std::runtime_error("unknown recall_mode: " + mode);
  }

  template <typename DistT>
  uint32_t countHitsById(const std::vector<std::pair<uint64_t, DistT>> &res,
                         const std::vector<uint64_t> &gt_ids, uint32_t topk)
  {
    const uint32_t rk =
        std::min<uint32_t>(topk, static_cast<uint32_t>(res.size()));
    const uint32_t gk =
        std::min<uint32_t>(topk, static_cast<uint32_t>(gt_ids.size()));
    uint32_t hits = 0;
    for (uint32_t i = 0; i < rk; ++i)
    {
      const uint64_t rid = res[i].first;
      for (uint32_t j = 0; j < gk; ++j)
      {
        if (rid == gt_ids[j])
        {
          ++hits;
          break;
        }
      }
    }
    return hits;
  }

  template <typename DistT>
  uint32_t countHitsByDistEps(const std::vector<std::pair<uint64_t, DistT>> &res,
                              const std::vector<float> &gt_dists, uint32_t topk,
                              float eps)
  {
    const uint32_t rk =
        std::min<uint32_t>(topk, static_cast<uint32_t>(res.size()));
    const uint32_t gk =
        std::min<uint32_t>(topk, static_cast<uint32_t>(gt_dists.size()));
    if (gk == 0)
      return 0;
    const float threshold = gt_dists[gk - 1] + eps;
    uint32_t hits = 0;
    for (uint32_t i = 0; i < rk; ++i)
    {
      const float rdist = static_cast<float>(res[i].second);
      if (rdist <= threshold)
        ++hits;
    }
    return hits;
  }

  void printGroundTruthTopK(uint64_t q,
                            const std::vector<std::vector<uint64_t>> &gt_ids,
                            const std::vector<std::vector<float>> &gt_dists,
                            uint32_t topk)
  {
    const bool has_ids = q < gt_ids.size();
    const bool has_dists = q < gt_dists.size();
    if (!has_ids && !has_dists)
    {
      std::cout << "GT topk: missing for q=" << q << "\n";
      return;
    }

    const std::vector<uint64_t> empty_ids;
    const std::vector<float> empty_dists;
    const auto &ids = has_ids ? gt_ids[q] : empty_ids;
    const auto &dists = has_dists ? gt_dists[q] : empty_dists;

    const uint32_t k_ids =
        std::min<uint32_t>(topk, static_cast<uint32_t>(ids.size()));
    const uint32_t k_dists =
        std::min<uint32_t>(topk, static_cast<uint32_t>(dists.size()));
    const uint32_t k = std::min<uint32_t>(k_ids, k_dists);

    std::cout << "GT top" << topk << " for q=" << q << " (print " << k
              << "):\n";
    for (uint32_t i = 0; i < k; ++i)
    {
      std::cout << "  [" << i << "] id=" << ids[i] << ", dist=" << dists[i]
                << "\n";
    }
    if (k == 0)
    {
      std::cout << "  (no gt entries)\n";
    }
    else if (k_ids != k_dists)
    {
      std::cout << "  (warning: gt_ids.size=" << ids.size()
                << ", gt_dists.size=" << dists.size() << ")\n";
    }
  }
}

int main(int argc, char **argv)
{
  if (argc < 2 || argc > 4)
  {
    std::cerr << "Usage: " << std::filesystem::path(argv[0]).filename().string()
              << " <benchmark_config.json> [recall_mode=id|dist] [dist_eps]\n";
    return 1;
  }
  const std::string config_path = argv[1];

  util::benchmark::benchmark_param param;
  if (util::loadBenchMarkConfig(config_path, param) != 0)
  {
    std::cerr << "Error: loadBenchMarkConfig failed\n";
    return 1;
  }
  if (!param.search_param || param.search_param->topk_value == 0)
  {
    std::cerr << "Error: invalid search_param.topk_value\n";
    return 1;
  }

  RecallMode recall_mode = RecallMode::kId;
  float dist_eps = 1e-5f;
  try
  {
    std::string raw;
    if (util::read_file_to_string(config_path, raw) == 0)
    {
      json cfg = json::parse(raw);
      if (cfg.contains("recall_mode"))
      {
        recall_mode = parseRecallMode(cfg.at("recall_mode").get<std::string>());
      }
      if (cfg.contains("dist_eps"))
      {
        dist_eps = cfg.at("dist_eps").get<float>();
      }
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "Warning: failed to parse optional recall fields: " << e.what()
              << "\n";
  }

  try
  {
    if (argc >= 3)
    {
      recall_mode = parseRecallMode(argv[2]);
    }
    if (argc >= 4)
    {
      dist_eps = std::stof(argv[3]);
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error: parse argv recall args failed: " << e.what() << "\n";
    return 1;
  }

  auto *env = runtime::MiniHyperVecEnv::getInstance();
  if (env->initForSearch(param.test_collection_name) != 0)
  {
    std::cerr << "Error: initMiniHyperVec failed\n";
    return 1;
  }

  std::vector<std::vector<uint64_t>> gt_ids;
  std::vector<std::vector<float>> gt_dists;
  if (util::loadGroundTruth(param.test_groundtruth_path, gt_ids, gt_dists) !=
      0)
  {
    std::cerr << "Error: loadGroundTruth failed\n";
    env->shutdownForSearch();
    return 1;
  }

  auto worker_handle =
      runtime::ServingWorkerPool::getInstance()->acquireHandle();
  if (!worker_handle)
  {
    std::cerr << "Error: acquire serving worker failed\n";
    env->shutdownForSearch();
    return 1;
  }
  worker_handle->bindToCPU();

  const uint32_t topk = param.search_param->topk_value;

  std::cout << "recall_mode="
            << (recall_mode == RecallMode::kId ? "id" : "dist")
            << ", dist_eps=" << dist_eps << "\n";

  uint64_t total_hits = 0;
  uint64_t total_returned = 0;

  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();

  if (param.data_type == minihypervec::collection::VecType::INT8)
  {
    util::Dataset<int8_t> query_ds(param.test_query_path);
    if (query_ds.getDataBase() == nullptr)
    {
      std::cerr << "Error: failed to map query file: " << param.test_query_path
                << "\n";
      env->shutdownForSearch();
      return 1;
    }
    const uint32_t dim = query_ds.dim;
    const uint64_t nq = query_ds.total_cnt;
    std::vector<int8_t> query(dim);
    std::vector<std::pair<uint64_t, int32_t>> res;

    for (uint64_t q = 0; q < nq; ++q)
    {
      std::memcpy(query.data(), query_ds.getDataBase() + q * dim, dim * sizeof(int8_t));
      res.clear();
      if (worker_handle->searchKnn(param.test_collection_name, query,
                                   *param.search_param, res) != 0)
      {
        std::cerr << "Error: searchKnn(INT8) failed at q=" << q << "\n";
        env->shutdownForSearch();
        return 1;
      }
      if (recall_mode == RecallMode::kId)
      {
        if (q < gt_ids.size())
        {
          total_hits += countHitsById(res, gt_ids[q], topk);
        }
      }
      else
      {
        if (q < gt_dists.size())
        {
          total_hits += countHitsByDistEps(res, gt_dists[q], topk, dist_eps);
        }
      }
      total_returned += topk;
    }

    std::cout << "nq=" << nq << " dim=" << dim << "\n";
  }
  else
  {
    std::cerr << "Error: unsupported vec_type in benchmark config\n";
    env->shutdownForSearch();
    return 1;
  }

  const auto t1 = clock::now();
  const double sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
          .count();
  const double recall = total_returned == 0
                            ? 0.0
                            : static_cast<double>(total_hits) /
                                  static_cast<double>(total_returned);

  std::cout << "elapsed=" << sec << " s, QPS="
            << (sec > 0.0 ? (static_cast<double>(total_returned) / topk) / sec
                          : 0.0)
            << "\n";
  std::cout << "Recall@" << topk << " = " << (recall * 100.0) << " %"
            << " (hits=" << total_hits << ", returned=" << total_returned
            << ")\n";

  if (env->shutdownForSearch() != 0)
  {
    std::cerr << "Warning: shutdownMiniHyperVec failed\n";
  }
  return 0;
}
