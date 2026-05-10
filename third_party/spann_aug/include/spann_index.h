#pragma once

#include "root.h"
#include "hnswlib/hnswlib.hpp"

namespace spann
{
    struct FillContext
    {
        int cluster_id = -1;
        int needed = 0;
        int topk = 0;
        int radius = 5;

        std::vector<int32_t> selected_ids;
        std::vector<int8_t> selected_vecs;

        FillContext() = default;
        FillContext(int cluster_id, int needed, int topk, int radius)
            : cluster_id(cluster_id), needed(needed), topk(topk), radius(radius) {}
    };
    struct Candidate
    {
        float dist;
        int32_t id;
        std::vector<int8_t> vec;
    };
    struct ClusterData
    {
        int64_t head_id = -1;
        int dim = 0;
        std::vector<int32_t> ids;
        std::vector<int8_t> vectors;
        int Count() const
        {
            return static_cast<int>(ids.size());
        }
    };

    class SpannIndex
    {
    public:
#pragma pack(push, 1)
        struct MetaRecord
        {
            int32_t pg;
            uint16_t off;
            int32_t cnt;
            uint16_t pc;
        };
#pragma pack(pop)

        static constexpr int64_t PAGE_SIZE = 4096;

    private:
        std::string head_path_;
        std::string head_id_path_;
        std::string disk_path_;

        int32_t num_heads_ = 0;
        int32_t dim_ = 0;
        int32_t total_docs_ = 0;
        int64_t base_offset_ = 0;

        std::vector<int8_t> head_vecs_;
        std::vector<int64_t> head_ids_;
        std::vector<MetaRecord> meta_;

        std::vector<std::vector<int32_t>> cluster_ids_;
        std::vector<std::vector<int8_t>> cluster_vecs_;

        std::vector<std::vector<int32_t>> snapshot_cluster_ids_;
        std::vector<std::vector<int8_t>> snapshot_cluster_vecs_;

        int fill_target_size_ = 64;
        int fill_neighbor_topk_ = 16;
        bool fill_count_head_ = true;

        std::atomic<int> fill_processed_clusters_{0};
        std::atomic<bool> fill_stop_progress_{false};

        int fd_ = -1;

        std::unique_ptr<hnswlib::L2SpaceI_int8> head_space_;
        std::unique_ptr<hnswlib::HierarchicalNSW<int>> head_hnsw_;

    private:
        void CheckClusterId(int cluster_id) const;
        void LoadHeads();
        void LoadHeadIds();
        void LoadMeta();

        void BuildSnapshotForFilling();
        void FillingWorker(int thread_id, int num_threads);
        std::vector<int32_t> QueryNeighborClustersByHead(int cluster_id, int topk) const;
        void SearchNeighborCandidates(FillContext &ctx) const;

    public:
        SpannIndex(const std::string &head_path, const std::string &head_id_path, const std::string &disk_path);
        ~SpannIndex();

        SpannIndex(const SpannIndex &) = delete;
        SpannIndex &operator=(const SpannIndex &) = delete;

        int GetNumHeads() const { return num_heads_; }
        int GetDim() const { return dim_; }
        int GetTotalDocs() const { return total_docs_; }
        int64_t GetBaseOffset() const { return base_offset_; }

        const int8_t *GetHeadVector(int cluster_id) const;
        int64_t GetHeadId(int cluster_id) const;
        int GetClusterSize(int cluster_id) const;

        int FetchCluster(int32_t cluster_id, std::vector<int8_t> &vecs, std::vector<int32_t> &ids) const;

        void LoadAllClusters();
        bool AllClustersLoaded() const;

        const std::vector<int32_t> &GetClusterIds(int cluster_id) const;
        const std::vector<int8_t> &GetClusterVectors(int cluster_id) const;

        void BuildHeadHNSW(int M, int ef_construction, int ef_search);
        void PerformFilling(int target_size, int neighbor_topk, bool count_head, int num_threads);

        void Summary() const;

    private:
        std::vector<std::vector<int32_t>> norms_;

    public:
        int32_t performNorms();
        void saveNorms(const std::string &norm_path) const;
        void saveHeadIndex(const std::string &head_index_path) const;
        void saveClusterIds(const std::string &cluster_ids_path) const;
    };

} // namespace spann
