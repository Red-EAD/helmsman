#include "spann_index.h"

namespace spann
{
    void SpannIndex::BuildHeadHNSW(int M, int ef_construction, int ef_search)
    {
        std::cout << "Building head HNSW index...\n";
        if (num_heads_ <= 0 || dim_ <= 0)
        {
            throw std::runtime_error("Invalid head data for HNSW build");
        }

        if (head_hnsw_ != nullptr)
        {
            std::cout << "Head HNSW already built, skip rebuild.\n";
            return;
        }

        head_space_ = std::make_unique<hnswlib::L2SpaceI_int8>(dim_);
        head_hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<int>>(
            head_space_.get(), num_heads_, M, ef_construction);
        head_hnsw_->ef_ = ef_search;

        int thread_cnt = static_cast<int>(std::thread::hardware_concurrency());
        if (thread_cnt <= 0)
        {
            thread_cnt = 1;
        }

        std::cout << "Start adding int8 head vectors to HNSW with "
                  << thread_cnt << " threads...\n";

        std::vector<std::thread> threads;
        threads.reserve(thread_cnt);

        const int total = num_heads_;
        const int batch_size = (total + thread_cnt - 1) / thread_cnt;

        for (int t = 0; t < thread_cnt; ++t)
        {
            const int start = t * batch_size;
            const int end = std::min((t + 1) * batch_size, total);
            if (start >= end)
            {
                continue;
            }
            threads.emplace_back([this, start, end]()
                                 {
                for (int i = start; i < end; ++i)
                {
                    const int8_t *vec = GetHeadVector(i);
                    head_hnsw_->addPoint((void *)vec, static_cast<hnswlib::labeltype>(i));
                } });
        }

        for (auto &th : threads)
        {
            th.join();
        }

        std::cout << "Head HNSW built: M=" << M
                  << ", ef_construction=" << ef_construction
                  << ", ef_search=" << ef_search
                  << ", thread_cnt=" << thread_cnt << "\n";
    }

    std::vector<int32_t> SpannIndex::QueryNeighborClustersByHead(int cluster_id, int topk) const
    {
        CheckClusterId(cluster_id);

        if (head_hnsw_ == nullptr)
        {
            throw std::runtime_error("Head HNSW has not been built");
        }

        if (topk <= 0)
        {
            return {};
        }

        const int8_t *query = GetHeadVector(cluster_id);
        auto result = head_hnsw_->searchKnn(query, static_cast<size_t>(topk));

        std::vector<int32_t> neighbors;
        neighbors.reserve(static_cast<size_t>(topk));

        while (!result.empty())
        {
            neighbors.push_back(static_cast<int32_t>(result.top().second));
            result.pop();
        }

        std::reverse(neighbors.begin(), neighbors.end());
        return neighbors;
    }

    void SpannIndex::BuildSnapshotForFilling()
    {
        if (!AllClustersLoaded())
        {
            throw std::runtime_error("LoadAllClusters() must be called before filling");
        }

        snapshot_cluster_ids_ = cluster_ids_;
        snapshot_cluster_vecs_ = cluster_vecs_;
    }

    void SpannIndex::SearchNeighborCandidates(FillContext &ctx) const
    {
        constexpr int MAX_RETRIES = 10;
        const int8_t *head_ptr = GetHeadVector(ctx.cluster_id);

        std::unordered_set<int32_t> visited_clusters;
        visited_clusters.reserve(static_cast<size_t>(ctx.topk) * 4);
        visited_clusters.insert(ctx.cluster_id);

        std::unordered_set<int64_t> existing_ids(
            snapshot_cluster_ids_[ctx.cluster_id].begin(),
            snapshot_cluster_ids_[ctx.cluster_id].end());

        std::unordered_map<int64_t, Candidate> best_by_id;
        best_by_id.reserve(static_cast<size_t>(ctx.needed) * 16);

        int current_k = ctx.topk;

        for (int retry = 0; retry < MAX_RETRIES; ++retry)
        {
            std::vector<int32_t> neighbors = QueryNeighborClustersByHead(ctx.cluster_id, current_k);

            for (int nbr : neighbors)
            {
                if (visited_clusters.find(nbr) != visited_clusters.end())
                {
                    continue;
                }
                visited_clusters.insert(nbr);

                const auto &nbr_ids = snapshot_cluster_ids_[nbr];
                const auto &nbr_vecs = snapshot_cluster_vecs_[nbr];
                const size_t nbr_cnt = nbr_ids.size();

                if (nbr_cnt == 0)
                {
                    continue;
                }

                for (size_t i = 0; i < nbr_cnt; ++i)
                {
                    const int64_t vid64 = static_cast<int64_t>(nbr_ids[i]);
                    if (existing_ids.find(vid64) != existing_ids.end())
                    {
                        continue;
                    }

                    const int8_t *vec_ptr = nbr_vecs.data() + i * dim_;
                    float dist = 0.0f;
                    for (int d = 0; d < dim_; ++d)
                    {
                        const float diff = static_cast<float>(vec_ptr[d]) - static_cast<float>(head_ptr[d]);
                        dist += diff * diff;
                    }

                    auto it = best_by_id.find(vid64);
                    if (it == best_by_id.end() || dist < it->second.dist)
                    {
                        Candidate cand;
                        cand.dist = dist;
                        cand.id = static_cast<int32_t>(vid64);
                        cand.vec.assign(vec_ptr, vec_ptr + dim_);
                        best_by_id[vid64] = std::move(cand);
                    }
                }
            }

            if (static_cast<int>(best_by_id.size()) >= ctx.needed)
            {
                break;
            }
            current_k += ctx.radius;
        }

        if (best_by_id.empty())
        {
            return;
        }

        std::vector<Candidate> candidates;
        candidates.reserve(best_by_id.size());
        for (auto &kv : best_by_id)
        {
            candidates.push_back(std::move(kv.second));
        }

        const int actual_fill = std::min(ctx.needed, static_cast<int>(candidates.size()));
        if (actual_fill <= 0)
        {
            return;
        }

        std::nth_element(
            candidates.begin(),
            candidates.begin() + actual_fill - 1,
            candidates.end(),
            [](const Candidate &a, const Candidate &b)
            {
                return a.dist < b.dist;
            });

        std::sort(
            candidates.begin(),
            candidates.begin() + actual_fill,
            [](const Candidate &a, const Candidate &b)
            {
                return a.dist < b.dist;
            });

        ctx.selected_ids.clear();
        ctx.selected_vecs.clear();
        ctx.selected_ids.reserve(actual_fill);
        ctx.selected_vecs.reserve(static_cast<size_t>(actual_fill) * dim_);

        for (int i = 0; i < actual_fill; ++i)
        {
            ctx.selected_ids.push_back(candidates[i].id);
            ctx.selected_vecs.insert(
                ctx.selected_vecs.end(),
                candidates[i].vec.begin(),
                candidates[i].vec.end());
        }
    }

    void SpannIndex::FillingWorker(int thread_id, int num_threads)
    {
        for (int cid = thread_id; cid < num_heads_; cid += num_threads)
        {
            auto &cur_ids = cluster_ids_[cid];
            auto &cur_vecs = cluster_vecs_[cid];
            int cur_cnt = static_cast<int>(cur_ids.size());

            if (cur_cnt < fill_target_size_)
            {
                FillContext ctx(cid, fill_target_size_ - cur_cnt, fill_neighbor_topk_, 5);
                SearchNeighborCandidates(ctx);
                cur_ids.insert(cur_ids.end(), ctx.selected_ids.begin(), ctx.selected_ids.end());
                cur_vecs.insert(cur_vecs.end(), ctx.selected_vecs.begin(), ctx.selected_vecs.end());
            }
            ++fill_processed_clusters_;
        }
    }

    void SpannIndex::PerformFilling(
        int target_size, int neighbor_topk, bool count_head, int num_threads)
    {
        if (target_size <= 0)
        {
            throw std::invalid_argument("target_size must be > 0");
        }
        if (neighbor_topk <= 0)
        {
            throw std::invalid_argument("neighbor_topk must be > 0");
        }
        if (!AllClustersLoaded())
        {
            throw std::runtime_error("Please call LoadAllClusters() before PerformFilling()");
        }
        if (head_hnsw_ == nullptr)
        {
            throw std::runtime_error("Please build head HNSW before PerformFilling()");
        }

        fill_target_size_ = target_size;
        fill_neighbor_topk_ = neighbor_topk;
        fill_count_head_ = count_head;
        BuildSnapshotForFilling();

        if (num_threads <= 0)
        {
            num_threads = static_cast<int>(std::thread::hardware_concurrency());
            if (num_threads <= 0)
            {
                num_threads = 1;
            }
        }

        std::cout << "===== Start Filling =====\n";
        std::cout << "target_size   : " << target_size << "\n";
        std::cout << "neighbor_topk : " << neighbor_topk << "\n";
        std::cout << "count_head    : " << (count_head ? "true" : "false") << "\n";
        std::cout << "num_threads   : " << num_threads << "\n";

        fill_processed_clusters_ = 0;
        fill_stop_progress_ = false;

        const auto start_time = std::chrono::steady_clock::now();

        std::thread progress_thread([this, start_time]()
                                    {
            using namespace std::chrono_literals;

            while (!fill_stop_progress_.load())
            {
                std::this_thread::sleep_for(500ms);

                const int done = fill_processed_clusters_.load();
                const double percent = 100.0 * done / std::max(1, num_heads_);

                const auto now = std::chrono::steady_clock::now();
                const double elapsed =
                    std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();

                const double speed = (elapsed > 0.0) ? (done / elapsed) : 0.0;
                const double eta = (speed > 0.0) ? ((num_heads_ - done) / speed) : 0.0;

                std::cout << "\r[fill progress] "
                          << done << "/" << num_heads_
                          << " (" << std::fixed << std::setprecision(2) << percent << "%)"
                          << ", speed=" << std::setprecision(1) << speed << " clusters/s"
                          << ", eta=" << std::setprecision(1) << eta << "s"
                          << std::flush;
            } });

        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        for (int tid = 0; tid < num_threads; ++tid)
        {
            workers.emplace_back([this, tid, num_threads]()
                                 { this->FillingWorker(tid, num_threads); });
        }

        for (auto &th : workers)
        {
            th.join();
        }

        fill_stop_progress_ = true;
        progress_thread.join();

        std::cout << "\r[fill progress] "
                  << num_heads_ << "/" << num_heads_
                  << " (100.00%)\n"
                  << "===== Filling Done =====\n";
    }

    int32_t SpannIndex::performNorms()
    {
        std::cout << ">>> Calculating cluster norms..." << std::endl;
        size_t num_clusters = head_ids_.size();
        norms_.clear();
        norms_.resize(num_clusters);

        std::vector<std::thread> threads;
        int thread_cnt = static_cast<int>(std::thread::hardware_concurrency());
        if (thread_cnt <= 0)
        {
            thread_cnt = 1;
        }
        threads.reserve(thread_cnt);
        const int batch_size = static_cast<int>((num_clusters + thread_cnt - 1) / thread_cnt);
        for (int t = 0; t < thread_cnt; ++t)
        {
            const int start = t * batch_size;
            const int end = std::min((t + 1) * batch_size, static_cast<int>(num_clusters));
            if (start >= end)
            {
                continue;
            }
            threads.emplace_back([this, start, end]()
                                 {
                for (int i = start; i < end; ++i)
                {
                    const std::vector<int32_t> &ids = GetClusterIds(i);
                    const std::vector<int8_t> &vecs = GetClusterVectors(i);
                    const int cnt = static_cast<int>(ids.size());
                    std::vector<int32_t> norms(cnt, 0);
                    for (int j = 0; j < cnt; ++j)
                    {
                        const int8_t *vec_ptr = vecs.data() + static_cast<size_t>(j) * dim_;
                        int32_t norm = 0;
                        for (int d = 0; d < dim_; ++d)
                        {
                            norm += static_cast<int32_t>(vec_ptr[d]) * vec_ptr[d];
                        }
                        norms[j] = norm;
                    }
                    norms_[i] = std::move(norms);
                } });
        }

        std::cout << "Waiting for norm calculation threads to finish..." << std::endl;
        for (auto &th : threads)
        {
            th.join();
        }
        std::cout << "All norm calculation threads finished." << std::endl;
        return 0;
    }
} // namespace spann
