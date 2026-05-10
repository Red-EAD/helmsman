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
    uint32_t getMaxHits(const std::vector<std::pair<uint64_t, int32_t>> &res,
                        const std::vector<uint64_t> &gt_ids,
                        const std::vector<float> gt_dis, uint32_t topk)
    {
        uint32_t max_hits = 0;
        float threshold_topk = gt_dis[topk - 1] + 1e-5f;
        for (size_t i = 0; i < res.size() && i < topk; ++i)
        {
            const float rdist = static_cast<float>(res[i].second);
            if (rdist <= threshold_topk)
                ++max_hits;
        }
        return max_hits;
    }

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
    util::benchmark::benchmark_param param;
    std::string index_type_str;
    std::string vec_type_str;
    uint32_t topk = 0;
    uint32_t cluster_nprobe = 0;
    std::vector<uint32_t> topk_list;
    std::string memory_index_type;
    uint32_t memory_search_max_visits = 0;
    RecallMode recall_mode = RecallMode::kId;
    float dist_eps = 1e-5f;
    int32_t thread_num = 1;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--collection_name" && i + 1 < argc)
        {
            param.test_collection_name = argv[++i];
        }
        else if (arg == "--query_path" && i + 1 < argc)
        {
            param.test_query_path = argv[++i];
        }
        else if (arg == "--groundtruth_path" && i + 1 < argc)
        {
            param.test_groundtruth_path = argv[++i];
        }
        else if (arg == "--index_type" && i + 1 < argc)
        {
            index_type_str = argv[++i];
        }
        else if (arg == "--vec_type" && i + 1 < argc)
        {
            vec_type_str = argv[++i];
        }
        else if (arg == "--topk" && i + 1 < argc)
        {
            topk = static_cast<uint32_t>(std::stoi(argv[++i]));
        }
        else if (arg == "--nprobe" && i + 1 < argc)
        {
            cluster_nprobe = static_cast<uint32_t>(std::stoi(argv[++i]));
        }
        else if (arg == "--topk" && i + 1 < argc)
        {
            while (i + 1 < argc && argv[i + 1][0] != '-')
            {
                topk_list.push_back(static_cast<uint32_t>(std::stoi(argv[++i])));
            }
        }
        else if (arg == "--memory_index_type" && i + 1 < argc)
        {
            memory_index_type = argv[++i];
        }
        else if (arg == "--memory_search_max_visits" && i + 1 < argc)
        {
            memory_search_max_visits = static_cast<uint32_t>(std::stoi(argv[++i]));
        }
        else if (arg == "--T" && i + 1 < argc)
        {
            thread_num = std::stoi(argv[++i]);
        }
        else if (arg == "--recall_mode" && i + 1 < argc)
        {
            recall_mode = parseRecallMode(argv[++i]);
        }
        else if (arg == "--dist_eps" && i + 1 < argc)
        {
            dist_eps = std::stof(argv[++i]);
        }
    }

    if (index_type_str == "HV_CONST")
    {
        param.index_type = minihypervec::collection::IndexType::HV_CONST;
    }
    else if (index_type_str == "HNSW")
    {
        param.index_type = minihypervec::collection::IndexType::HNSW;
    }
    else
    {
        std::cerr << "Error: unsupported index_type: " << index_type_str << "\n";
        return 1;
    }

    if (vec_type_str == "INT8")
    {
        param.data_type = minihypervec::collection::VecType::INT8;
    }
    else
    {
        std::cerr << "Error: unsupported vec_type: " << vec_type_str << "\n";
        return 1;
    }

    if (param.test_collection_name.empty() || param.test_query_path.empty() || param.test_groundtruth_path.empty() || topk == 0)
    {
        std::cerr << "Error: missing required arguments.\n";
        std::cerr << "Usage: " << argv[0] << " --collection_name ... --query_path ... --groundtruth_path ... --index_type ... --vec_type ... --topk ... [--nprobe ...] [--memory_index_type ...] [--memory_search_max_visits ...] [--recall_mode ...] [--dist_eps ...]\n";
        return 1;
    }

    if (topk_list.empty())
        topk_list.push_back(topk);

    for (auto topk_run : topk_list)
    {
        if (param.index_type == minihypervec::collection::IndexType::HV_CONST)
        {
            auto sp = std::make_unique<minihypervec::index::MiniHyperVecConstSearchParam>();
            sp->topk_value = topk_run;
            sp->cluster_nprobe = cluster_nprobe;
            if (!memory_index_type.empty() || memory_search_max_visits > 0)
            {
                auto hnsw_sp = std::make_unique<minihypervec::index::HnswSearchParam>();
                if (memory_search_max_visits > 0)
                    hnsw_sp->max_visits = memory_search_max_visits;
                hnsw_sp->topk_value = cluster_nprobe;
                sp->centroid_search_param = std::move(hnsw_sp);
            }
            param.search_param = std::move(sp);
        }
        else if (param.index_type == minihypervec::collection::IndexType::HNSW)
        {
            auto sp = std::make_unique<minihypervec::index::HnswSearchParam>();
            sp->topk_value = topk_run;
            if (memory_search_max_visits > 0)
                sp->max_visits = memory_search_max_visits;
            param.search_param = std::move(sp);
        }
        else
        {
            std::cerr << "Error: unknown index_type for search_param\n";
            return 1;
        }

        std::cout << "==== Running with topk=" << topk_run << " ====" << std::endl;

        auto *env = runtime::MiniHyperVecEnv::getInstance();
        if (env->initForSearch(param.test_collection_name) != 0)
        {
            std::cerr << "Error: initMiniHyperVec failed\n";
            return 1;
        }

        std::vector<std::vector<uint64_t>> gt_ids;
        std::vector<std::vector<float>> gt_dists;
        if (util::loadGroundTruth(param.test_groundtruth_path, gt_ids, gt_dists) != 0)
        {
            std::cerr << "Error: loadGroundTruth failed\n";
            env->shutdownForSearch();
            return 1;
        }

        const std::string QueryPath = param.test_query_path;
        util::Dataset<int8_t> query_dataset(QueryPath);
        uint32_t dim = query_dataset.dim;
        uint64_t nq = query_dataset.total_cnt;
        std::vector<std::vector<int8_t>> query_vecs(query_dataset.total_cnt,
                                                    std::vector<int8_t>(dim));

        const int8_t *query_base = query_dataset.getDataBase();
        for (uint64_t i = 0; i < query_dataset.total_cnt; i++)
        {
            const int8_t *cur_query = query_base + i * dim;
            std::memcpy(query_vecs[i].data(), cur_query, dim * sizeof(int8_t));
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

        std::cout << "recall_mode="
                  << (recall_mode == RecallMode::kId ? "id" : "dist")
                  << ", dist_eps=" << dist_eps << "\n";

        std::vector<std::vector<std::pair<uint64_t, int32_t>>> res(nq);

        using clock = std::chrono::steady_clock;
        using usec = std::chrono::microseconds;

        const uint32_t repeats = 3;
        const uint64_t total_ops = nq * repeats;
        std::vector<double> latencies_ms(total_ops, 0.0);

        int32_t active_threads = thread_num > 0 ? thread_num : 1;
        int32_t chunk = (static_cast<int32_t>(nq) + active_threads - 1) / active_threads;
        std::vector<std::thread> threads;
        threads.reserve(active_threads);

        std::mutex io_mu;
        clock::time_point t0, t1;

        std::cout << "Start search " << nq << " queries with " << active_threads
                  << " threads (each repeats " << repeats << "x)\n";

        t0 = clock::now();

        auto worker_fn = [&](uint32_t tid)
        {
            const uint32_t start = tid * chunk;
            const uint32_t end = std::min<uint32_t>(nq, start + chunk);
            if (start >= end)
                return;
            auto worker_handle = runtime::ServingWorkerPool::getInstance()->acquireHandle();
            worker_handle->bindToCPU();

            for (uint32_t r = 0; r < repeats; ++r)
            {
                for (uint32_t q_id = start; q_id < end; ++q_id)
                {
                    res[q_id].clear();
                    std::vector<int8_t> &cur_query = query_vecs[q_id];
                    auto q_t0 = clock::now();
                    worker_handle->searchKnn(param.test_collection_name,
                                             cur_query, *param.search_param, res[q_id]);
                    auto q_t1 = clock::now();
                    uint64_t us = std::chrono::duration_cast<usec>(q_t1 - q_t0).count();
                    latencies_ms[static_cast<size_t>(r) * nq + q_id] = static_cast<double>(us) / 1000.0;
                }
            }
        };

        for (uint32_t t = 0; t < static_cast<uint32_t>(active_threads); ++t)
        {
            threads.emplace_back(worker_fn, t);
        }

        for (auto &th : threads)
            th.join();

        t1 = clock::now();

        std::cout << "nq=" << nq << " dim=" << dim << ", threads=" << active_threads << "\n";

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "Elapsed (search-only, " << repeats << "x): " << ms << " ms\n";
        std::cout << "Avg per pass: " << (ms / static_cast<double>(repeats)) << " ms\n";

        std::vector<double> sorted_lat = latencies_ms;
        std::sort(sorted_lat.begin(), sorted_lat.end());
        auto percentileNearestRank = [](const std::vector<double> &sorted_ms, double p) -> double
        {
            if (sorted_ms.empty())
                return 0.0;
            const double N = static_cast<double>(sorted_ms.size());
            size_t rank = static_cast<size_t>(std::ceil(p / 100.0 * N));
            if (rank == 0)
                rank = 1;
            if (rank > sorted_ms.size())
                rank = sorted_ms.size();
            return sorted_ms[rank - 1];
        };
        const double p50 = percentileNearestRank(sorted_lat, 50.0);
        const double p99 = percentileNearestRank(sorted_lat, 99.0);
        const double p999 = percentileNearestRank(sorted_lat, 99.9);
        const double p9999 = percentileNearestRank(sorted_lat, 99.99);
        const double p99999 = percentileNearestRank(sorted_lat, 99.999);
        const double pmin = sorted_lat.empty() ? 0.0 : sorted_lat.front();
        const double pmax = sorted_lat.empty() ? 0.0 : sorted_lat.back();
        double sum = 0.0;
        for (double v : latencies_ms)
            sum += v;
        const double mean = latencies_ms.empty() ? 0.0 : sum / latencies_ms.size();

        std::cout << "Latency (searchKnn, per-query):\n"
                  << "  min = " << pmin << " ms\n"
                  << "  mean = " << mean << " ms\n"
                  << "  P50 = " << p50 << " ms\n"
                  << "  P99 = " << p99 << " ms\n"
                  << "  P99.9 = " << p999 << " ms\n"
                  << "  P99.99 = " << p9999 << " ms\n"
                  << "  P99.999 = " << p99999 << " ms\n";
        std::cout << " throughput = " << (nq * repeats) / (ms / 1000.0) << " QPS\n";

        uint64_t total_hits = 0;
        for (uint32_t q_id = 0; q_id < nq; ++q_id)
        {
            uint32_t max_hits =
                getMaxHits(res[q_id], gt_ids[q_id], gt_dists[q_id], topk_run);
            total_hits += max_hits;
        }
        const uint64_t total_returned = static_cast<uint64_t>(nq) * topk_run;
        std::cout << "total_hits: " << total_hits
                  << " total returned: " << total_returned << std::endl;
        const double recall = (total_returned == 0)
                                  ? 0.0
                                  : static_cast<double>(total_hits) /
                                        static_cast<double>(total_returned);
        std::cout << "Recall@" << topk_run << " = " << (recall * 100.0) << " %"
                  << "  (queries=" << nq << ")\n";

        if (env->shutdownForSearch() != 0)
        {
            std::cerr << "Warning: shutdownMiniHyperVec failed\n";
        }
    }
    return 0;
}
